# HealthSync Scribe: consultation recorder → doctor's note → hospital system

One feature of HealthSync. The doctor presses **Start** at the beginning of a consultation and **Stop** at the end. The scribe:

1. **Transcribes** the conversation (Urdu, English or mixed) with Whisper, running on the hospital's own computer.
2. **Drafts the note** with Claude: complaint, history, symptoms (including denied ones), examination, diagnoses, medicines, tests, advice, follow-up, red flags, and a short Urdu summary for the patient. Anything unclear goes in a **"Please check"** list.
3. **Waits for the doctor.** The doctor edits and approves. Nothing reaches the hospital system without approval.
4. **Sends** the approved note to the hospital system as **FHIR R4** records (Patient, Encounter, Condition, MedicationRequest, ServiceRequest, Flag, Appointment, Composition). FHIR is the standard most HMS / ERP / EMR systems can import. A plain JSON webhook mode exists for systems without FHIR.

The included **demo hospital system** (`erp_mock/`) receives those records and gives management:

- a **follow-up tracker**: overdue, due in the next 2 days, upcoming, came back
- **analytics**: visits per day and per doctor, top diagnoses, medicines and tests, red flags, and follow-up attendance

## Run it

You need Docker Desktop and a Claude API key.

```bash
cd scribe
cp .env.example .env          # put your ANTHROPIC_API_KEY in .env
docker compose up -d --build
```

- **Doctor's screen:** http://localhost:8090
- **Hospital dashboard (demo):** http://localhost:8091

Both containers restart by themselves whenever Docker starts, so the scribe is ready before the first patient of the day. The first recording downloads the Whisper model (about 1.6 GB). It is kept in the `models` volume, so this happens only once.

Browsers allow the microphone only on `localhost` or over HTTPS. If doctors use other computers to reach the scribe, put it behind HTTPS (for example a hospital reverse proxy).

## Connect a real hospital system

In `.env`:

| Setting | Meaning |
|---|---|
| `SCRIBE_ERP_MODE=fhir` | POST a FHIR transaction Bundle to `SCRIBE_ERP_URL` (any FHIR R4 server: HAPI FHIR, OpenMRS FHIR module, Epic/Cerner gateways, most modern HMS) |
| `SCRIBE_ERP_MODE=webhook` | POST our own JSON (`session`, `patient`, `doctor`, `note`) to `SCRIBE_ERP_URL`, for an ERP with a custom API |
| `SCRIBE_ERP_MODE=none` | Keep notes in the scribe only |
| `SCRIBE_ERP_TOKEN` | Sent as `Authorization: Bearer …` |

With a real system, run only the scribe: `docker compose up -d scribe`.

## Privacy and safety

- **Consent first.** A recording cannot start until the doctor confirms that the patient and doctor agreed.
- **Identifiers stay local.** Only the transcript, age, sex and department go to Claude. The patient's name, MR number and phone never leave the hospital computer.
- **Audio is deleted** as soon as it is transcribed (`SCRIBE_KEEP_AUDIO=false`).
- **The doctor approves every note.** The AI drafts and never decides. Once a note is sent, it is locked.
- **No guessing.** The AI writes only what was said. Unclear words, doses or drug names are listed under "Please check". If speech recognition confidence is low, the doctor is warned at the top of the note.
- If the hospital system is down, the note waits as "not sent" and can be sent again. Nothing is lost.

## Speech-to-text model

Set with `SCRIBE_WHISPER_MODEL`. From the Hugging Face survey of 18 Nov 2026:

| Model | Licence | Notes |
|---|---|---|
| `large-v3-turbo` (default) | MIT | Best with Urdu and English mixed in one sentence |
| `kingabzpro/whisper-large-v3-urdu-ct2` | Apache-2.0 | Fine-tuned on Urdu (Common Voice WER 26%). Better for pure Urdu; English words may come out in Urdu script |

No Urdu medical speech model exists yet. Before using this with patients, compare models on our own recorded consultations.

## Develop and test

```bash
cd scribe
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest                                               # no model, API key or network needed
```

To work on the screens without Whisper or Claude:

```bash
SCRIBE_TRANSCRIBER=fake SCRIBE_NOTE_WRITER=fake uvicorn app.main:app --port 8090
uvicorn erp_mock.main:app --port 8091
```

| Path | What it is |
|---|---|
| `app/main.py` | Web service and consultation flow |
| `app/transcribe.py` | Whisper speech-to-text |
| `app/note.py` | Claude note writing (prompt and safety rules) |
| `app/fhir.py` | Approved note → FHIR bundle |
| `app/erp.py` | Sending to the hospital system |
| `app/store.py` | SQLite storage of consultations |
| `app/static/index.html` | Doctor's screen |
| `erp_mock/` | Demo hospital system: FHIR store, follow-ups, analytics, dashboard |
