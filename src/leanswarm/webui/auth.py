import datetime
import hashlib
import hmac
import secrets
import sqlite3

from fastapi import Depends, HTTPException, Request


def hash_password(pw: str) -> str:
    salt = secrets.token_bytes(16)
    hash_bytes = hashlib.scrypt(pw.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${hash_bytes.hex()}"

def verify_password(pw: str, stored: str) -> bool:
    try:
        parts = stored.split("$")
        if len(parts) != 3 or parts[0] != "scrypt":
            return False
        salt = bytes.fromhex(parts[1])
        stored_hash = bytes.fromhex(parts[2])
        hash_bytes = hashlib.scrypt(pw.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
        return hmac.compare_digest(hash_bytes, stored_hash)
    except Exception:
        return False

def get_current_user(request: Request) -> sqlite3.Row | None:
    token = request.cookies.get("leanswarm_session")
    if not token:
        return None
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    conn: sqlite3.Connection = request.app.state.db
    row = conn.execute(
        "SELECT users.* FROM users JOIN sessions ON users.id = sessions.user_id WHERE sessions.token_hash = ? AND sessions.expires_at > ?",
        (token_hash, datetime.datetime.now(datetime.UTC).isoformat())
    ).fetchone()
    return row

def require_user(user: sqlite3.Row | None = Depends(get_current_user)) -> sqlite3.Row:  # noqa: B008
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user
