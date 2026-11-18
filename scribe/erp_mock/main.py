"""
erp_mock - a small stand-in for the hospital's management system (HMS / ERP).

It accepts the FHIR transaction bundles the scribe sends, stores every resource, and gives
management two things on top:
  * Follow-up tracker: every follow-up the doctor asked for, as upcoming / due / overdue / done
  * Analytics: visits per day, top diagnoses, medicines, tests, red flags, follow-up adherence
A real deployment replaces this with the hospital's own system (any FHIR R4 server, or a webhook).

Run:  uvicorn erp_mock.main:app --port 8091
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from urllib.parse import quote
from zoneinfo import ZoneInfo

import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.auth import COOKIE, MAX_AGE, AccessGate

DUE_WINDOW_DAYS = 2      # a follow-up within the next 2 days counts as "due"
STATIC = os.path.join(os.path.dirname(__file__), "static")


class Db:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS resources (
            id INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT NOT NULL, body TEXT NOT NULL,
            identifier TEXT, created_at TEXT NOT NULL)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_type ON resources(type)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_ident ON resources(type, identifier)")
        self.db.commit()

    def find_by_identifier(self, rtype: str, ident: str) -> Optional[int]:
        r = self.db.execute("SELECT id FROM resources WHERE type = ? AND identifier = ?",
                            (rtype, ident)).fetchone()
        return r[0] if r else None

    def insert(self, rtype: str, body: dict, ident: Optional[str]) -> int:
        cur = self.db.execute("INSERT INTO resources (type, body, identifier, created_at) VALUES (?,?,?,?)",
                              (rtype, "{}", ident, datetime.now(timezone.utc).isoformat(timespec="seconds")))
        return cur.lastrowid

    def save_body(self, rid: int, body: dict) -> None:
        self.db.execute("UPDATE resources SET body = ? WHERE id = ?", (json.dumps(body), rid))

    def all(self, rtype: str) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT body FROM resources WHERE type = ? ORDER BY id", (rtype,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def get(self, rtype: str, rid: int) -> Optional[dict]:
        with self.lock:
            r = self.db.execute("SELECT body FROM resources WHERE type = ? AND id = ?", (rtype, rid)).fetchone()
        return json.loads(r[0]) if r else None


def _first_identifier(res: dict) -> Optional[str]:
    ids = res.get("identifier") or []
    return f"{ids[0].get('system')}|{ids[0].get('value')}" if ids else None


def _rewrite_refs(obj, mapping: dict):
    """Replace urn:uuid references inside a bundle with the stored Type/id."""
    if isinstance(obj, dict):
        return {k: (mapping.get(v, v) if k == "reference" and isinstance(v, str) else _rewrite_refs(v, mapping))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rewrite_refs(v, mapping) for v in obj]
    return obj


# ---------------- follow-up reminders ----------------
def pk_whatsapp_number(phone: Optional[str]) -> Optional[str]:
    """Pakistani mobile number -> international digits for wa.me (03001234567 -> 923001234567)."""
    if not phone:
        return None
    digits = "".join(ch for ch in phone if ch.isdigit())
    if digits.startswith("0092"):
        digits = digits[2:]
    if digits.startswith("03") and len(digits) == 11:
        digits = "92" + digits[1:]
    return digits if digits.startswith("923") and len(digits) == 12 else None


def reminder(f: dict) -> dict:
    """A short Urdu reminder the staff can send. Nothing is sent automatically."""
    if f["state"] not in ("overdue", "due", "upcoming") or not f["due_date"]:
        return {"reminder_ur": None, "whatsapp_url": None}
    d = date.fromisoformat(f["due_date"]).strftime("%d/%m/%Y")
    name = f["patient"] or "محترم مریض"
    doctor = f["doctor"] or "ڈاکٹر"
    if f["state"] == "overdue":
        text = (f"السلام علیکم {name}، {doctor} نے آپ کو {d} کو دوبارہ معائنے کے لیے بلایا تھا۔ "
                "آپ ابھی تک نہیں آئے۔ براہ کرم جلد از جلد ہسپتال تشریف لائیں۔")
    else:
        text = (f"السلام علیکم {name}، یاد دہانی: {doctor} نے آپ کو {d} کو دوبارہ معائنے کے لیے بلایا ہے۔ "
                "براہ کرم اپنی رپورٹس ساتھ لائیں۔")
    number = pk_whatsapp_number(f.get("phone"))
    url = f"https://wa.me/{number}?text={quote(text, safe='')}" if number else None
    return {"reminder_ur": text, "whatsapp_url": url}


class Login(BaseModel):
    code: str


def create_app(db_path: Optional[str] = None, today=None) -> FastAPI:
    db = Db(db_path or os.environ.get("ERP_DB", "data/erp.db"))
    tz = ZoneInfo(os.environ.get("ERP_TIMEZONE", "Asia/Karachi"))
    today_fn = today or (lambda: datetime.now(tz).date())
    app = FastAPI(title="Hospital ERP (mock)")
    # ERP_TOKEN: what the scribe sends (SCRIBE_ERP_TOKEN). ERP_ACCESS_CODE: staff login for the dashboard.
    gate = AccessGate(os.environ.get("ERP_ACCESS_CODE", ""), (os.environ.get("ERP_TOKEN", ""),))
    app.middleware("http")(gate.middleware)

    @app.post("/api/login")
    def login(body: Login, request: Request):
        if not gate.code_ok(body.code):
            if gate.enabled:
                time.sleep(1.0)
                raise HTTPException(401, "Wrong access code")
            return {"ok": True}
        r = JSONResponse({"ok": True})
        r.set_cookie(COOKIE, gate.make_cookie(), max_age=MAX_AGE, httponly=True, samesite="strict",
                     secure=request.url.scheme == "https")
        return r

    # ---------------- FHIR ----------------
    @app.post("/fhir")
    async def transaction(request: Request):
        try:
            bundle = await request.json()
        except ValueError:
            raise HTTPException(400, "Body is not JSON")
        if bundle.get("resourceType") != "Bundle" or bundle.get("type") != "transaction":
            raise HTTPException(400, "Expected a FHIR Bundle of type transaction")
        entries = bundle.get("entry") or []
        mapping, results, plan = {}, [], []
        with db.lock:
            # pass 1: give every entry a real id (or reuse the existing Patient/Practitioner)
            for e in entries:
                res = e.get("resource") or {}
                rtype = res.get("resourceType")
                if not rtype or e.get("request", {}).get("method") != "POST":
                    db.db.rollback()
                    raise HTTPException(400, "Every entry needs a resource and request.method POST")
                ident = _first_identifier(res)
                existing = db.find_by_identifier(rtype, ident) if e["request"].get("ifNoneExist") and ident else None
                rid = existing or db.insert(rtype, res, ident)
                mapping[e.get("fullUrl")] = f"{rtype}/{rid}"
                plan.append((rid, rtype, res, existing is None))
            # pass 2: store bodies with references pointing at the real ids
            for rid, rtype, res, created in plan:
                if created:
                    db.save_body(rid, {**_rewrite_refs(res, mapping), "id": str(rid)})
                results.append({"response": {"status": "201 Created" if created else "200 OK",
                                             "location": f"{rtype}/{rid}/_history/1"}})
            db.db.commit()
        return JSONResponse({"resourceType": "Bundle", "type": "transaction-response", "entry": results})

    @app.get("/fhir/{rtype}")
    def search(rtype: str):
        items = db.all(rtype)
        return {"resourceType": "Bundle", "type": "searchset", "total": len(items),
                "entry": [{"resource": r} for r in items]}

    @app.get("/fhir/{rtype}/{rid}")
    def read(rtype: str, rid: int):
        r = db.get(rtype, rid)
        if r is None:
            raise HTTPException(404, f"{rtype}/{rid} not found")
        return r

    # ---------------- follow-ups ----------------
    def _resource(ref: str) -> dict:
        try:
            rtype, rid = ref.split("/")
            return db.get(rtype, int(rid)) or {}
        except (ValueError, AttributeError):
            return {}

    def followups() -> list[dict]:
        out, t = [], today_fn()
        for a in db.all("Appointment"):
            start = (a.get("requestedPeriod") or [{}])[0].get("start")
            pref = next((p["actor"]["reference"] for p in a.get("participant", [])
                         if p.get("actor", {}).get("reference", "").startswith("Patient/")), "")
            dref = next((p["actor"]["reference"] for p in a.get("participant", [])
                         if p.get("actor", {}).get("reference", "").startswith("Practitioner/")), "")
            pat, doc = _resource(pref), _resource(dref)
            due = date.fromisoformat(start) if start else None
            if a.get("status") == "fulfilled":
                state = "done"
            elif a.get("status") == "cancelled":
                state = "cancelled"
            elif due is None:
                state = "upcoming"
            elif due < t:
                state = "overdue"
            elif due <= t + timedelta(days=DUE_WINDOW_DAYS):
                state = "due"
            else:
                state = "upcoming"
            out.append({
                "id": a["id"], "due_date": start, "state": state, "reason": a.get("description"),
                "patient": (pat.get("name") or [{}])[0].get("text"),
                "mrn": (pat.get("identifier") or [{}])[0].get("value"),
                "phone": next((x["value"] for x in pat.get("telecom", []) if x.get("system") == "phone"), None),
                "doctor": (doc.get("name") or [{}])[0].get("text"),
                "days_overdue": (t - due).days if state == "overdue" else 0,
            })
            out[-1].update(reminder(out[-1]))
        order = {"overdue": 0, "due": 1, "upcoming": 2, "done": 3, "cancelled": 4}
        return sorted(out, key=lambda f: (order[f["state"]], f["due_date"] or ""))

    @app.get("/api/followups")
    def list_followups(state: Optional[str] = None):
        items = followups()
        return [f for f in items if f["state"] == state] if state else items

    def _set_status(rid: int, status: str):
        with db.lock:
            row = db.db.execute("SELECT body FROM resources WHERE type='Appointment' AND id=?", (rid,)).fetchone()
            if row is None:
                raise HTTPException(404, "No such follow-up")
            body = json.loads(row[0])
            body["status"] = status
            db.save_body(rid, body)
            db.db.commit()
        return {"id": str(rid), "status": status}

    @app.post("/api/followups/{rid}/done")
    def mark_done(rid: int):
        return _set_status(rid, "fulfilled")

    @app.post("/api/followups/{rid}/cancel")
    def cancel(rid: int):
        return _set_status(rid, "cancelled")

    # ---------------- analytics ----------------
    @app.get("/api/analytics")
    def analytics(days: int = 30):
        t = today_fn()
        since = t - timedelta(days=max(days, 1) - 1)
        encounters = db.all("Encounter")
        per_day = Counter()
        per_doctor = Counter()
        for e in encounters:
            d = (e.get("period", {}).get("start") or "")[:10]
            if d and d >= since.isoformat():
                per_day[d] += 1
            per_doctor[(_resource(e["participant"][0]["individual"]["reference"]).get("name") or [{}])[0]
                       .get("text", "unknown") if e.get("participant") else "unknown"] += 1
        diag = Counter(c.get("code", {}).get("text", "?") for c in db.all("Condition")
                       if c.get("verificationStatus", {}).get("coding", [{}])[0].get("code") != "refuted")
        meds = Counter(m.get("medicationCodeableConcept", {}).get("text", "?") for m in db.all("MedicationRequest"))
        tests = Counter(s.get("code", {}).get("text", "?") for s in db.all("ServiceRequest"))
        flags = Counter(f.get("code", {}).get("text", "?") for f in db.all("Flag"))
        fu = followups()
        fu_count = Counter(f["state"] for f in fu)
        closed = fu_count["done"] + fu_count["overdue"]
        return {
            "visits_total": len(encounters),
            "patients_total": len(db.all("Patient")),
            "visits_per_day": [{"date": (since + timedelta(days=i)).isoformat(),
                                "visits": per_day.get((since + timedelta(days=i)).isoformat(), 0)}
                               for i in range((t - since).days + 1)],
            "visits_per_doctor": [{"doctor": k, "visits": v} for k, v in per_doctor.most_common()],
            "top_diagnoses": [{"name": k, "count": v} for k, v in diag.most_common(10)],
            "top_medications": [{"name": k, "count": v} for k, v in meds.most_common(10)],
            "top_tests": [{"name": k, "count": v} for k, v in tests.most_common(10)],
            "red_flags": [{"name": k, "count": v} for k, v in flags.most_common(10)],
            "followups": {s: fu_count.get(s, 0) for s in ("overdue", "due", "upcoming", "done", "cancelled")},
            # of follow-ups whose date has passed, how many came back
            "followup_adherence_pct": round(100 * fu_count["done"] / closed, 1) if closed else None,
        }

    @app.get("/")
    def dashboard():
        return FileResponse(os.path.join(STATIC, "index.html"))

    return app


app = create_app() if os.environ.get("ERP_NO_APP") != "1" else None
