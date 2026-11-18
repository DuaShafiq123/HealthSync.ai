"""Access code on the scribe, token + access code on the ERP."""
from fastapi.testclient import TestClient

from app.auth import AccessGate
from erp_mock.main import create_app as create_erp
from conftest import TODAY, make_scribe, start_visit, upload


def test_scribe_locked_without_code(tmp_path, erp):
    c = make_scribe(tmp_path, erp, access_code="clinic-2026")
    assert c.get("/").status_code == 200                      # page loads, shows the login box
    assert c.get("/healthz").json()["access_code"] is True
    assert c.get("/api/sessions").status_code == 401
    assert start_visit(c).status_code == 401
    assert c.post("/api/login", json={"code": "wrong"}).status_code == 401
    assert c.get("/api/sessions").status_code == 401

    r = c.post("/api/login", json={"code": " clinic-2026 "})
    assert r.status_code == 200
    cookie = r.cookies.get("scribe_session")
    assert cookie and "clinic-2026" not in cookie              # the code itself is never stored
    sid = start_visit(c).json()["id"]
    assert upload(c, sid).status_code == 202

    c.post("/api/logout")
    c.cookies.clear()
    assert c.get("/api/sessions").status_code == 401
    assert c.get("/api/sessions", headers={"Authorization": "Bearer clinic-2026"}).status_code == 200


def test_forged_or_expired_cookie_is_rejected():
    g = AccessGate("x")
    good = g.make_cookie()
    assert g.cookie_ok(good)
    assert not g.cookie_ok("9999999999." + "0" * 64)
    assert not g.cookie_ok(good.split(".")[0] + ".bad")
    assert not AccessGate("x").cookie_ok(good)                 # signed by another start-up
    expired = "1." + __import__("hmac").new(g.key, b"1", "sha256").hexdigest()
    assert not g.cookie_ok(expired)


def test_erp_token_and_staff_code(tmp_path, monkeypatch):
    monkeypatch.setenv("ERP_TOKEN", "machine-secret")
    monkeypatch.setenv("ERP_ACCESS_CODE", "staff")
    erp = TestClient(create_erp(str(tmp_path / "erp.db"), today=lambda: TODAY))
    assert erp.get("/").status_code == 200
    assert erp.get("/api/followups").status_code == 401
    assert erp.post("/fhir", json={}).status_code == 401

    # scribe without the token cannot deliver; with it, it can
    no_token = make_scribe(tmp_path / "a", erp)
    sid = start_visit(no_token).json()["id"]
    upload(no_token, sid)
    s = no_token.post(f"/api/sessions/{sid}/approve").json()
    assert s["status"] == "send_failed" and "401" in s["error"]

    with_token = make_scribe(tmp_path / "b", erp, erp_token="machine-secret")
    sid = start_visit(with_token).json()["id"]
    upload(with_token, sid)
    assert with_token.post(f"/api/sessions/{sid}/approve").json()["status"] == "sent"

    # staff log in to the dashboard with the code, not the machine token
    assert erp.post("/api/login", json={"code": "machine-secret"}).status_code == 401
    assert erp.post("/api/login", json={"code": "staff"}).status_code == 200
    assert len(erp.get("/api/followups").json()) == 1
