import asyncio
import datetime
import json
import secrets
import sqlite3
import time
from collections import deque
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from leanswarm.engine.config import RuntimeSettings
from leanswarm.engine.llm import LiteLLMRouter
from leanswarm.engine.pricing import estimate_run
from leanswarm.webui.auth import get_current_user, hash_password, require_user, verify_password
from leanswarm.webui.config import WebUISettings
from leanswarm.webui.db import connect, init_db
from leanswarm.webui.runs import RunManager
from leanswarm.webui.schemas import (
    AuthRequest,
    DoctorRequest,
    EstimateRequest,
    PublishRequest,
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
    
    # Rate limiting
    ip_history: dict[str, deque[float]] = {}
    
    @app.on_event("startup")
    async def startup():
        asyncio.create_task(run_manager.purge_loop())

    @app.post("/api/auth/register")
    def register(req: AuthRequest, request: Request):
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
    def login(req: AuthRequest, request: Request):
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
    def logout(request: Request):
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
    def get_me(user: sqlite3.Row | None = Depends(get_current_user)):  # noqa: B008
        return {"email": user["email"] if user else None}

    @app.post("/api/runs")
    async def start_run(req: StartRunRequest, request: Request, user: sqlite3.Row | None = Depends(get_current_user)):  # noqa: B008
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
            
        run_id = await run_manager.start(req, user["id"] if user else None)
        return {"id": run_id}

    @app.get("/api/runs/{id}/events")
    async def stream_events(id: str, request: Request):
        job = run_manager.jobs.get(id)
        if not job:
            raise HTTPException(404, "Run not found or purged")
            
        async def gen():
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
    def get_run(id: str, request: Request, user: sqlite3.Row | None = Depends(get_current_user)):  # noqa: B008
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
    def save_run(id: str, req: PublishRequest, request: Request, user: sqlite3.Row = Depends(require_user)):  # noqa: B008
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
    def publish_run(id: str, req: PublishRequest, request: Request, user: sqlite3.Row | None = Depends(get_current_user)):  # noqa: B008
        c: sqlite3.Connection = request.app.state.db
        
        row = c.execute("SELECT id, user_id, is_public FROM runs WHERE id = ?", (id,)).fetchone()
        if row:
            if row["user_id"] and (not user or row["user_id"] != user["id"]):
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
                job.id, user["id"] if user else None, title, job.request_sanitized.get("question", ""), seed_excerpt,
                json.dumps(job.request_sanitized), json.dumps(job.result), json.dumps(job.models),
                complete_evt.get("prompt_tokens_total", 0), complete_evt.get("completion_tokens_total", 0),
                complete_evt.get("cost_usd"), datetime.datetime.now(datetime.UTC).isoformat()
            )
        )
        c.commit()
        return {"id": job.id}

    @app.get("/api/runs")
    def list_runs(request: Request, user: sqlite3.Row = Depends(require_user)):  # noqa: B008
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
    def delete_run(id: str, request: Request, user: sqlite3.Row = Depends(require_user)):  # noqa: B008
        c: sqlite3.Connection = request.app.state.db
        row = c.execute("SELECT user_id FROM runs WHERE id = ?", (id,)).fetchone()
        if not row or row["user_id"] != user["id"]:
            raise HTTPException(404, "Run not found")
        c.execute("DELETE FROM runs WHERE id = ?", (id,))
        c.commit()
        return {"ok": True}

    @app.post("/api/estimate")
    def estimate(req: EstimateRequest):
        return estimate_run(
            rounds=req.rounds,
            max_agents=req.max_agents,
            active_agent_fraction=req.active_agent_fraction,
            group_size=req.group_size,
            flagship_model=req.models.flagship,
            standard_model=req.models.standard,
            cheap_model=req.models.cheap
        )

    @app.post("/api/doctor")
    def doctor(req: DoctorRequest):
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
    def gallery(request: Request, limit: int = 20, offset: int = 0):
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
    def get_gallery_item(id: str, request: Request):
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

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}
        
    static_dir = Path(__file__).parent / "static"
    if (static_dir / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")
        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str):
            if full_path.startswith("api/"):
                raise HTTPException(404)
            path = static_dir / full_path
            if path.is_file():
                return FileResponse(path)
            return FileResponse(static_dir / "index.html")
    else:
        @app.get("/")
        def fallback_root():
            return JSONResponse({"status": "frontend not built"})

    return app
