"""
auth.py - a simple access code in front of the scribe.

Set SCRIBE_ACCESS_CODE (in .env). Doctors type it once per browser; the browser then keeps a
signed cookie for 12 hours. Scripts can send the code as "Authorization: Bearer <code>".
Without SCRIBE_ACCESS_CODE the scribe is open (fine on a laptop, NOT on a hospital network).
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse

COOKIE = "scribe_session"
MAX_AGE = 12 * 3600
OPEN_PATHS = {"/", "/healthz", "/api/login", "/api/logout", "/favicon.ico"}


class AccessGate:
    def __init__(self, code: Optional[str], machine_tokens: tuple[str, ...] = ()):
        self.code = (code or "").strip()
        # extra bearer tokens for other systems (e.g. the scribe posting to the ERP)
        self.machine_tokens = tuple(t.strip() for t in machine_tokens if t and t.strip())
        # a fresh key per start-up: restarting the scribe logs everyone out
        self.key = secrets.token_bytes(32)

    @property
    def enabled(self) -> bool:
        return bool(self.code or self.machine_tokens)

    def code_ok(self, given: str) -> bool:
        return bool(self.code) and hmac.compare_digest(given.strip().encode(), self.code.encode())

    def token_ok(self, given: str) -> bool:
        return self.code_ok(given) or any(hmac.compare_digest(given.strip().encode(), t.encode())
                                          for t in self.machine_tokens)

    def make_cookie(self) -> str:
        expires = str(int(time.time()) + MAX_AGE)
        sig = hmac.new(self.key, expires.encode(), hashlib.sha256).hexdigest()
        return f"{expires}.{sig}"

    def cookie_ok(self, value: Optional[str]) -> bool:
        if not value or "." not in value:
            return False
        expires, sig = value.split(".", 1)
        good = hmac.new(self.key, expires.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig, good) and expires.isdigit() and int(expires) > time.time()

    def allowed(self, request: Request) -> bool:
        if not self.enabled or request.url.path in OPEN_PATHS:
            return True
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer ") and self.token_ok(auth[7:]):
            return True
        return self.cookie_ok(request.cookies.get(COOKIE))

    async def middleware(self, request: Request, call_next):
        if not self.allowed(request):
            return JSONResponse({"detail": "Enter the access code"}, status_code=401)
        return await call_next(request)
