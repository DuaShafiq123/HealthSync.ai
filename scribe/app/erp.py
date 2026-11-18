"""
erp.py - sends an approved consultation to the hospital's system.

SCRIBE_ERP_MODE:
  fhir     POST the FHIR transaction Bundle to SCRIBE_ERP_URL (any FHIR R4 server, or our erp_mock)
  webhook  POST our own JSON {session, patient, note} to SCRIBE_ERP_URL (for an ERP with a custom API)
  none     do not send (notes stay in the scribe only)
SCRIBE_ERP_TOKEN, when set, is sent as "Authorization: Bearer <token>".
"""
from __future__ import annotations

from typing import Optional

import httpx

from .config import Settings


class ErpError(RuntimeError):
    pass


class ErpClient:
    def __init__(self, settings: Settings, http: Optional[httpx.Client] = None):
        self.mode = settings.erp_mode
        self.url = settings.erp_url
        self.http = http or httpx.Client(timeout=30.0)
        self.headers = {"Authorization": f"Bearer {settings.erp_token}"} if settings.erp_token else {}
        if self.mode not in ("fhir", "webhook", "none"):
            raise ValueError(f"SCRIBE_ERP_MODE must be fhir, webhook or none, got {self.mode!r}")

    def send(self, bundle: dict, payload: dict) -> str:
        """Returns a reference the ERP gave back (e.g. 'Encounter/17'). Raises ErpError on failure."""
        if self.mode == "none":
            return "not sent (SCRIBE_ERP_MODE=none)"
        body, ctype = (bundle, "application/fhir+json") if self.mode == "fhir" else (payload, "application/json")
        try:
            r = self.http.post(self.url, json=body, headers={**self.headers, "Content-Type": ctype})
        except httpx.HTTPError as e:
            raise ErpError(f"Could not reach the hospital system at {self.url}: {e}") from e
        if r.status_code >= 300:
            raise ErpError(f"Hospital system answered {r.status_code}: {r.text[:300]}")
        if self.mode == "webhook":
            try:
                body = r.json()
            except ValueError:
                return "accepted"
            return str(body.get("id", "accepted")) if isinstance(body, dict) else "accepted"
        return _encounter_location(bundle, r)


def _encounter_location(bundle: dict, r: httpx.Response) -> str:
    """In a FHIR transaction-response, entry i answers request i. Return the Encounter's location."""
    try:
        resp = r.json()
        for sent, got in zip(bundle["entry"], resp.get("entry", []), strict=False):
            if sent["resource"]["resourceType"] == "Encounter":
                loc = got.get("response", {}).get("location", "")
                return loc.split("/_history")[0] or "accepted"
    except (ValueError, KeyError, AttributeError):
        pass
    return "accepted"
