import os
import sys
from datetime import date

import pytest

os.environ["SCRIBE_NO_APP"] = "1"     # do not build the module-level apps on import
os.environ["ERP_NO_APP"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.erp import ErpClient  # noqa: E402
from app.main import create_app  # noqa: E402
from app.note import FakeNoteWriter  # noqa: E402
from app.transcribe import FakeTranscriber  # noqa: E402
from erp_mock.main import create_app as create_erp  # noqa: E402

TODAY = date(2026, 11, 18)


@pytest.fixture
def erp(tmp_path):
    return TestClient(create_erp(str(tmp_path / "erp.db"), today=lambda: TODAY))


def make_scribe(tmp_path, erp_http, transcriber=None, note_writer=None, **settings):
    s = Settings(data_dir=str(tmp_path / "scribe"), transcriber="fake", note_writer="fake",
                 erp_mode=settings.pop("erp_mode", "fhir"), erp_url="http://erp/fhir", **settings)
    return TestClient(create_app(s, transcriber or FakeTranscriber(), note_writer or FakeNoteWriter(),
                                 ErpClient(s, http=erp_http)))


@pytest.fixture
def scribe(tmp_path, erp):
    return make_scribe(tmp_path, erp)


PATIENT = {"name": "Ayesha Khan", "mrn": "MR-1001", "age": 34, "sex": "female", "phone": "0300-1234567"}


def start_visit(client, patient=None, consent=True):
    return client.post("/api/sessions", json={"patient": patient or PATIENT, "doctor": "Dr. Sana Malik",
                                              "department": "Medicine OPD", "consent": consent})


def upload(client, sid, data=b"fake-webm-bytes", ctype="audio/webm", name="visit.webm"):
    return client.post(f"/api/sessions/{sid}/audio", files={"audio": (name, data, ctype)})
