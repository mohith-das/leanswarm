import asyncio
import datetime
import json
import secrets
import sqlite3
import time
from collections import deque
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from leanswarm.engine.config import RuntimeSettings
from leanswarm.engine.llm import LiteLLMRouter
from leanswarm.engine.models import TaskType
from leanswarm.engine.pricing import estimate_run
from leanswarm.webui.auth import get_current_user, hash_password, require_user, verify_password
from leanswarm.webui.config import WebUISettings
from leanswarm.webui.db import connect, init_db
from leanswarm.webui.email import send_password_reset_email
from leanswarm.webui.runs import RunManager
from leanswarm.webui.schemas import (
    AuthRequest,
    ChatRequest,
    DoctorRequest,
    EstimateRequest,
    ForgotPasswordRequest,
    PublishRequest,
    ResetPasswordRequest,
    StartRunRequest,
)


def create_webui_app() -> FastAPI:
    app = FastAPI(title="LeanSwarm Web UI")
    settings = WebUISettings.from_env()
    settings.ensure_dirs()
    
    db_path = settings.data_dir / "app.sqlite3"
    conn = connect(db_path)
    init_db(conn)
    app.state.db = conn
    
    run_manager = RunManager(settings)
    app.state.run_manager = run_manager
    
    # Rate limiting: separate buckets for run starts and chat/report calls.
    ip_history: dict[str, deque[float]] = {}
    chat_ip_history: dict[str, deque[float]] = {}

    def check_rate_limit(
        bucket: dict[str, deque[float]], ip: str, limit: int, label: str
    ) -> None:
        if limit <= 0:
            return
        q = bucket.setdefault(ip, deque())
        now = time.time()
        while q and q[0] < now - 3600:
            q.popleft()
        if len(q) >= limit:
            raise HTTPException(429, f"{label} rate limit exceeded")
        q.append(now)


    @app.on_event("startup")
    async def startup() -> None:
        asyncio.create_task(run_manager.purge_loop())

    @app.post("/api/auth/register")
    def register(req: AuthRequest, request: Request) -> JSONResponse:
        if not settings.allow_signup:
            raise HTTPException(403, "Signup disabled")
        if len(req.password) < 8 or len(req.password) > 128:
            raise HTTPException(422, "Password must be 8-128 chars")
        if "@" not in req.email:
            raise HTTPException(422, "Invalid email")
            
        c: sqlite3.Connection = request.app.state.db
        try:
            cur = c.execute(
                "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
                (req.email, hash_password(req.password), datetime.datetime.now(datetime.UTC).isoformat())
            )
            user_id = cur.lastrowid
            c.commit()
        except sqlite3.IntegrityError as err:
            raise HTTPException(409, "Email taken") from err
            
        token = secrets.token_urlsafe(32)
        token_hash = __import__('hashlib').sha256(token.encode()).hexdigest()
        expires = datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=30)
        c.execute(
            "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
            (token_hash, user_id, expires.isoformat())
        )
        c.commit()
        
        resp = JSONResponse({"email": req.email})
        resp.set_cookie("leanswarm_session", token, httponly=True, samesite="lax", secure=settings.secure_cookies, max_age=30*86400)
        return resp

    @app.post("/api/auth/login")
    def login(req: AuthRequest, request: Request) -> JSONResponse:
        c: sqlite3.Connection = request.app.state.db
        row = c.execute("SELECT id, email, password_hash FROM users WHERE email = ?", (req.email,)).fetchone()
        if not row or not verify_password(req.password, row["password_hash"]):
            raise HTTPException(401, "Invalid credentials")
            
        token = secrets.token_urlsafe(32)
        token_hash = __import__('hashlib').sha256(token.encode()).hexdigest()
        expires = datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=30)
        c.execute(
            "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
            (token_hash, row["id"], expires.isoformat())
        )
        c.commit()
        
        resp = JSONResponse({"email": row["email"]})
        resp.set_cookie("leanswarm_session", token, httponly=True, samesite="lax", secure=settings.secure_cookies, max_age=30*86400)
        return resp

    @app.post("/api/auth/logout")
    def logout(request: Request) -> JSONResponse:
        token = request.cookies.get("leanswarm_session")
        if token:
            token_hash = __import__('hashlib').sha256(token.encode()).hexdigest()
            c: sqlite3.Connection = request.app.state.db
            c.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
            c.commit()
        resp = JSONResponse({"ok": True})
        resp.delete_cookie("leanswarm_session")
        return resp

    @app.get("/api/auth/me")
    def get_me(user: sqlite3.Row | None = Depends(get_current_user)) -> dict[str, Any]:  # noqa: B008
        return {"email": user["email"] if user else None}

    @app.post("/api/auth/forgot-password")
    def forgot_password(req: ForgotPasswordRequest, request: Request) -> dict[str, Any]:
        if not settings.smtp_host:
            raise HTTPException(500, "SMTP not configured")
        c: sqlite3.Connection = request.app.state.db
        row = c.execute("SELECT id FROM users WHERE email = ?", (req.email,)).fetchone()
        if row:
            token = secrets.token_urlsafe(32)
            token_hash = __import__('hashlib').sha256(token.encode()).hexdigest()
            expires = datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1)
            c.execute("DELETE FROM password_resets WHERE user_id = ?", (row["id"],))
            c.execute(
                "INSERT INTO password_resets (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
                (token_hash, row["id"], expires.isoformat()),
            )
            reset_url = str(request.base_url).rstrip("/") + "/reset-password?token=" + token
            send_password_reset_email(req.email, reset_url, settings)
            c.commit()
        return {"ok": True}

    @app.post("/api/auth/reset-password")
    def reset_password(req: ResetPasswordRequest, request: Request) -> dict[str, Any]:
        c: sqlite3.Connection = request.app.state.db
        token_hash = __import__('hashlib').sha256(req.token.encode()).hexdigest()
        row = c.execute(
            "SELECT user_id FROM password_resets WHERE token_hash = ? AND expires_at > ?",
            (token_hash, datetime.datetime.now(datetime.UTC).isoformat()),
        ).fetchone()
        if not row:
            raise HTTPException(400, "Invalid or expired token")
        if len(req.password) < 8 or len(req.password) > 128:
            raise HTTPException(422, "Password must be 8-128 chars")
        c.execute("DELETE FROM password_resets WHERE token_hash = ?", (token_hash,))
        c.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (hash_password(req.password), row["user_id"]),
        )
        c.commit()
        return {"ok": True}

    @app.post("/api/runs")
    async def start_run(req: StartRunRequest, request: Request, user: sqlite3.Row | None = Depends(get_current_user)) -> dict[str, Any]:  # noqa: B008
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        if settings.runs_per_hour_per_ip > 0:
            history = ip_history.setdefault(ip, deque())
            while history and history[0] < now - 3600:
                history.popleft()
            if len(history) >= settings.runs_per_hour_per_ip:
                raise HTTPException(429, "Rate limit exceeded")
            history.append(now)
            
        if req.rounds > settings.max_rounds:
            raise HTTPException(422, f"rounds exceeds max {settings.max_rounds}")
        if req.max_agents > settings.max_agents:
            raise HTTPException(422, f"max_agents exceeds max {settings.max_agents}")
        if len(req.seed_document) > settings.max_seed_chars:
            raise HTTPException(422, f"seed_document exceeds {settings.max_seed_chars} chars")
        if len(req.source_urls) > 6:
            raise HTTPException(422, "max 6 source URLs")
        for url in req.source_urls:
            if len(url) > 2000:
                raise HTTPException(422, "each source URL must be ≤ 2000 chars")
            
        run_id = await run_manager.start(req, user["id"] if user else None)
        return {"id": run_id}

    @app.get("/api/runs/{id}/events")
    async def stream_events(id: str, request: Request) -> StreamingResponse:
        job = run_manager.jobs.get(id)
        if not job:
            raise HTTPException(404, "Run not found or purged")
            
        async def gen() -> AsyncIterator[str]:
            idx = 0
            while True:
                if await request.is_disconnected():
                    break
                async with job.cond:
                    if idx < len(job.events):
                        evt = job.events[idx]
                        idx += 1
                        yield f"data: {json.dumps(evt)}\n\n"
                        if evt.get("type") in ("complete", "error"):
                            break
                    else:
                        try:
                            await asyncio.wait_for(job.cond.wait(), timeout=15)
                        except TimeoutError:
                            yield ": keepalive\n\n"
        
        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/runs/{id}")
    def get_run(id: str, request: Request, user: sqlite3.Row | None = Depends(get_current_user)) -> dict[str, Any]:  # noqa: B008
        c: sqlite3.Connection = request.app.state.db
        row = c.execute("SELECT * FROM runs WHERE id = ?", (id,)).fetchone()
        
        # If it's in the DB, enforce privacy
        if row:
            if not row["is_public"] and (user is None or row["user_id"] != user["id"]):
                raise HTTPException(404, "Run not found")
            return {
                "id": row["id"],
                "status": "complete",
                "request": json.loads(row["request_json"]),
                "models": json.loads(row["models_json"]),
                "result": json.loads(row["result_json"]),
                "is_public": bool(row["is_public"])
            }
            
        # If it's not in the DB, check memory
        job = run_manager.jobs.get(id)
        if job:
            if job.owner_user_id is not None and (user is None or job.owner_user_id != user["id"]):
                raise HTTPException(404, "Run not found")
            return {
                "id": job.id,
                "status": job.status,
                "request": job.request_sanitized,
                "models": job.models,
                "result": job.result,
                "error": job.error
            }
            
        raise HTTPException(404, "Run not found")

    @app.post("/api/runs/{id}/save")
    def save_run(id: str, req: PublishRequest, request: Request, user: sqlite3.Row = Depends(require_user)) -> dict[str, Any]:  # noqa: B008
        c: sqlite3.Connection = request.app.state.db
        
        row = c.execute("SELECT id, user_id FROM runs WHERE id = ?", (id,)).fetchone()
        if row:
            if row["user_id"] != user["id"]:
                raise HTTPException(403, "Not your run")
            if req.title:
                c.execute("UPDATE runs SET title = ? WHERE id = ?", (req.title, id))
                c.commit()
            return {"id": id}
            
        job = run_manager.jobs.get(id)
        if not job:
            raise HTTPException(410, "Run purged")
        if job.status != "complete" or not job.result:
            raise HTTPException(400, "Run not complete")
            
        title = req.title or job.request_sanitized.get("title") or "Saved Run"
        seed_excerpt = job.request_sanitized.get("seed_document", "")[:500]
        
        evts = job.events
        complete_evt = next((e for e in evts if e.get("type") == "complete"), {})
        
        c.execute(
            """INSERT INTO runs 
            (id, user_id, title, question, seed_excerpt, request_json, result_json, models_json, 
             prompt_tokens, completion_tokens, cost_usd, is_public, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
            (
                job.id, user["id"], title, job.request_sanitized.get("question", ""), seed_excerpt,
                json.dumps(job.request_sanitized), json.dumps(job.result), json.dumps(job.models),
                complete_evt.get("prompt_tokens_total", 0), complete_evt.get("completion_tokens_total", 0),
                complete_evt.get("cost_usd"), datetime.datetime.now(datetime.UTC).isoformat()
            )
        )
        c.commit()
        return {"id": job.id}

    @app.post("/api/runs/{id}/publish")
    def publish_run(id: str, req: PublishRequest, request: Request, user: sqlite3.Row = Depends(require_user)) -> dict[str, Any]:  # noqa: B008
        c: sqlite3.Connection = request.app.state.db
        
        row = c.execute("SELECT id, user_id, is_public FROM runs WHERE id = ?", (id,)).fetchone()
        if row:
            if row["user_id"] != user["id"]:
                raise HTTPException(403, "Not your run")
            c.execute("UPDATE runs SET is_public = 1, title = COALESCE(?, title) WHERE id = ?", (req.title, id))
            c.commit()
            return {"id": id}
            
        job = run_manager.jobs.get(id)
        if not job:
            raise HTTPException(410, "Run purged")
        if job.status != "complete" or not job.result:
            raise HTTPException(400, "Run not complete")
            
        title = req.title or job.request_sanitized.get("title") or "Published Run"
        seed_excerpt = job.request_sanitized.get("seed_document", "")[:500]
        
        evts = job.events
        complete_evt = next((e for e in evts if e.get("type") == "complete"), {})
        
        c.execute(
            """INSERT INTO runs 
            (id, user_id, title, question, seed_excerpt, request_json, result_json, models_json, 
             prompt_tokens, completion_tokens, cost_usd, is_public, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)""",
            (
                job.id, user["id"], title, job.request_sanitized.get("question", ""), seed_excerpt,
                json.dumps(job.request_sanitized), json.dumps(job.result), json.dumps(job.models),
                complete_evt.get("prompt_tokens_total", 0), complete_evt.get("completion_tokens_total", 0),
                complete_evt.get("cost_usd"), datetime.datetime.now(datetime.UTC).isoformat()
            )
        )
        c.commit()
        return {"id": job.id}

    @app.get("/api/runs")
    def list_runs(request: Request, user: sqlite3.Row = Depends(require_user)) -> list[dict[str, Any]]:  # noqa: B008
        c: sqlite3.Connection = request.app.state.db
        rows = c.execute("SELECT id, title, question, created_at, is_public, models_json, cost_usd FROM runs WHERE user_id = ? ORDER BY created_at DESC", (user["id"],)).fetchall()
        return [
            {
                "id": r["id"], "title": r["title"], "question": r["question"],
                "created_at": r["created_at"], "is_public": bool(r["is_public"]),
                "models_json": json.loads(r["models_json"]), "cost_usd": r["cost_usd"]
            } for r in rows
        ]

    @app.delete("/api/runs/{id}")
    def delete_run(id: str, request: Request, user: sqlite3.Row = Depends(require_user)) -> dict[str, Any]:  # noqa: B008
        c: sqlite3.Connection = request.app.state.db
        row = c.execute("SELECT user_id FROM runs WHERE id = ?", (id,)).fetchone()
        if not row or row["user_id"] != user["id"]:
            raise HTTPException(404, "Run not found")
        c.execute("DELETE FROM runs WHERE id = ?", (id,))
        c.commit()
        return {"ok": True}

    @app.post("/api/estimate")
    def estimate(req: EstimateRequest) -> dict[str, Any]:
        return estimate_run(
            rounds=req.rounds,
            max_agents=req.max_agents,
            active_agent_fraction=req.active_agent_fraction,
            group_size=req.group_size,
            flagship_model=req.models.flagship,
            standard_model=req.models.standard,
            cheap_model=req.models.cheap,
            seed_chars=req.seed_chars,
        )

    @app.post("/api/doctor")
    def doctor(req: DoctorRequest) -> list[dict[str, Any]]:
        eng_settings = RuntimeSettings.from_env()
        eng_settings.dry_run = False
        eng_settings.credentials = req.credentials.copy()
        if req.api_base:
            eng_settings.api_base = req.api_base
        if req.api_key:
            eng_settings.api_key = req.api_key
            
        router = LiteLLMRouter(eng_settings)
        models = list(dict.fromkeys([req.models.flagship, req.models.standard, req.models.cheap]))
        
        results = []
        for model in models:
            ready, missing = router._live_ready(model)
            res = {"model": model, "ready": ready, "missing": missing}
            if req.ping:
                try:
                    from litellm import acompletion
                    kwargs = {
                        "model": model,
                        "max_tokens": 16,
                        "messages": [
                            {"role": "system", "content": 'Reply with exactly {"ok": true}'},
                            {"role": "user", "content": "ping"}
                        ]
                    }
                    if eng_settings.api_base:
                        kwargs["api_base"] = eng_settings.api_base
                    api_key = router._resolve_api_key(model)
                    if api_key:
                        kwargs["api_key"] = api_key
                        
                    start_t = time.perf_counter()
                    asyncio.run(acompletion(**kwargs))
                    latency = (time.perf_counter() - start_t) * 1000
                    res["ping_ms"] = round(latency, 1)
                except Exception as e:
                    res["ping_error"] = str(e).split("\n", 1)[0]
            results.append(res)
        return results

    @app.get("/api/gallery")
    def gallery(request: Request, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        limit = min(50, max(1, limit))
        c: sqlite3.Connection = request.app.state.db
        rows = c.execute(
            "SELECT id, title, question, result_json, models_json, cost_usd, created_at FROM runs WHERE is_public = 1 ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
        
        out = []
        for r in rows:
            res_json = json.loads(r["result_json"])
            report = res_json.get("report", {})
            out.append({
                "id": r["id"],
                "title": r["title"],
                "question": r["question"],
                "prediction": str(report.get("prediction", ""))[:140],
                "confidence": report.get("confidence", 0.5),
                "models_json": json.loads(r["models_json"]),
                "cost_usd": r["cost_usd"],
                "created_at": r["created_at"],
                "tick_count": report.get("tick_count", 0),
                "agent_count": len(res_json.get("world", {}).get("agents", []))
            })
        return out

    @app.get("/api/gallery/{id}")
    def get_gallery_item(id: str, request: Request) -> dict[str, Any]:
        c: sqlite3.Connection = request.app.state.db
        row = c.execute("SELECT * FROM runs WHERE id = ? AND is_public = 1", (id,)).fetchone()
        if not row:
            raise HTTPException(404, "Not found")
            
        return {
            "id": row["id"],
            "status": "complete",
            "request": json.loads(row["request_json"]),
            "models": json.loads(row["models_json"]),
            "result": json.loads(row["result_json"]),
            "is_public": True,
            "created_at": row["created_at"]
        }

    @app.post("/api/runs/{id}/chat")
    async def chat(id: str, req: ChatRequest, request: Request, user: sqlite3.Row | None = Depends(get_current_user)) -> dict[str, Any]:  # noqa: B008
        from leanswarm.engine.llm import LiveCredentialsError
        from leanswarm.webui.chat import run_chat

        if not req.message.strip():
            raise HTTPException(422, "message is required")
        ip = request.client.host if request.client else "unknown"
        check_rate_limit(chat_ip_history, ip, settings.chats_per_hour_per_ip, "Chat")

        # Resolve the run result
        job = run_manager.jobs.get(id)
        result: dict[str, Any] | None = None
        if job:
            if job.owner_user_id is not None and (user is None or job.owner_user_id != user["id"]):
                raise HTTPException(404, "Run not found")
            result = job.result
        else:
            c: sqlite3.Connection = request.app.state.db
            row = c.execute("SELECT * FROM runs WHERE id = ?", (id,)).fetchone()
            if not row:
                raise HTTPException(404, "Run not found")
            if not row["is_public"] and (user is None or row["user_id"] != user["id"]):
                raise HTTPException(404, "Run not found")
            result = json.loads(row["result_json"])
        if result is None:
            raise HTTPException(404, "Run not complete")

        # Build a per-request router
        eng_settings = RuntimeSettings.from_env()
        eng_settings.dry_run = not req.live
        if req.models is not None:
            eng_settings.flagship_model = req.models.flagship
            eng_settings.standard_model = req.models.standard
            eng_settings.cheap_model = req.models.cheap
        eng_settings.credentials = req.credentials.copy()
        if req.api_base:
            eng_settings.api_base = req.api_base
        if req.api_key:
            eng_settings.api_key = req.api_key
        eng_settings.log_dir = settings.data_dir / "chat-logs"
        router = LiteLLMRouter(eng_settings)
        try:
            return await run_chat(router, result, req.agent_id, req.message, req.history, req.live)
        except KeyError as exc:
            raise HTTPException(404, "Agent not found in this run") from exc
        except LiveCredentialsError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.post("/api/runs/{id}/report")
    async def generate_report(id: str, req: ChatRequest, request: Request, user: sqlite3.Row | None = Depends(get_current_user)) -> dict[str, Any]:  # noqa: B008
        from leanswarm.engine.llm import LiveCredentialsError

        ip = request.client.host if request.client else "unknown"
        check_rate_limit(chat_ip_history, ip, settings.chats_per_hour_per_ip, "Report")

        job = run_manager.jobs.get(id)
        result: dict[str, Any] | None = None
        if job:
            if job.owner_user_id is not None and (user is None or job.owner_user_id != user["id"]):
                raise HTTPException(404, "Run not found")
            result = job.result
        else:
            c: sqlite3.Connection = request.app.state.db
            row = c.execute("SELECT * FROM runs WHERE id = ?", (id,)).fetchone()
            if not row:
                raise HTTPException(404, "Run not found")
            if not row["is_public"] and (user is None or row["user_id"] != user["id"]):
                raise HTTPException(404, "Run not found")
            result = json.loads(row["result_json"])
        if result is None:
            raise HTTPException(404, "Run not complete")

        report = result.get("report", {})
        ticks = result.get("ticks", [])
        world = result.get("world", {})
        profile = world.get("profile", {})

        eng_settings = RuntimeSettings.from_env()
        eng_settings.dry_run = not req.live
        if req.models is not None:
            eng_settings.flagship_model = req.models.flagship
            eng_settings.standard_model = req.models.standard
            eng_settings.cheap_model = req.models.cheap
        eng_settings.credentials = req.credentials.copy()
        if req.api_base:
            eng_settings.api_base = req.api_base
        if req.api_key:
            eng_settings.api_key = req.api_key
        eng_settings.log_dir = settings.data_dir / "chat-logs"
        router = LiteLLMRouter(eng_settings)

        before_p = router.prompt_tokens_total
        before_c = router.completion_tokens_total
        try:
            response = await router.route(
                TaskType.FULL_REPORT,
                {
                    "use_llm": req.live,
                    "question": report.get("question", ""),
                    "prediction": report.get("prediction", ""),
                    "confidence": report.get("confidence", 0.5),
                    "rationale": report.get("rationale", []),
                    "key_events": report.get("key_events", []),
                    "tick_count": report.get("tick_count", 0),
                    "tick_events": [
                        event
                        for tick in ticks
                        for event in tick.get("events", [])[:4]
                    ],
                    "world_summary": profile.get("summary", ""),
                    "world_topics": [t.get("label", "") for t in (profile.get("topics", []) or [])[:6]],
                    "world_entities": [e.get("label", "") for e in (profile.get("entities", []) or [])[:8]],
                    "agent_names": [
                        (a.get("name", "") + (": " + a.get("stance", "")) if a.get("stance") else a.get("name", ""))
                        for a in world.get("agents", [])[:16]
                    ],
                },
            )
        except LiveCredentialsError as exc:
            raise HTTPException(422, str(exc)) from None
        return {
            "title": str(response.get("title", "")),
            "sections": response.get("sections", []),
            "prompt_tokens": router.prompt_tokens_total - before_p,
            "completion_tokens": router.completion_tokens_total - before_c,
        }

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}
        
    static_dir = Path(__file__).parent / "static"
    if (static_dir / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")
        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str) -> FileResponse:
            if full_path.startswith("api/"):
                raise HTTPException(404)
            path = static_dir / full_path
            if path.is_file():
                return FileResponse(path)
            return FileResponse(static_dir / "index.html")
    else:
        @app.get("/")
        def fallback_root() -> JSONResponse:
            return JSONResponse({"status": "frontend not built"})

    return app
