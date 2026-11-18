# HealthSync Scribe: consultation recorder → doctor's note → hospital system

One feature of HealthSync. The doctor presses **Start** at the beginning of a consultation and **Stop** at the end. The scribe:

1. **Transcribes** the conversation (Urdu, English or mixed) with Whisper, running on the hospital's own computer.
2. **Drafts the note** with Claude: complaint, history, symptoms (including denied ones), examination, diagnoses, medicines, tests, advice, follow-up, red flags, and a short Urdu summary for the patient. Anything unclear goes in a **"Please check"** list.
3. **Waits for the doctor.** The doctor edits and approves. Nothing reaches the hospital system without approval.
4. **Sends** the approved note to the hospital system as **FHIR R4** records (Patient, Encounter, Condition, MedicationRequest, ServiceRequest, Flag, Appointment, Composition). FHIR is the standard most HMS / ERP / EMR systems can import. A plain JSON webhook mode exists for systems without FHIR.
5. **Learns how good it is.** The AI draft is kept next to the approved note. Every change the doctor makes is counted: an added item means the AI missed it, a removed item means the AI got it wrong.

When the doctor types an MR number, the patient's **previous visits** (complaint, diagnoses, medicines) appear above the consent box.

The included **demo hospital system** (`erp_mock/`) receives those records and gives management:

- a **follow-up tracker**: overdue, due in the next 2 days, upcoming, came back
- **analytics**: visits per day and per doctor, top diagnoses, medicines and tests, red flags, and follow-up attendance
- **reminders**: a short Urdu reminder for each open follow-up, with a WhatsApp button that opens the chat with the message filled in. Staff press send; nothing is sent automatically

## Run it

You need Docker Desktop and a Claude API key.

```bash
cd scribe
cp .env.example .env          # put your ANTHROPIC_API_KEY and access codes in .env
docker compose up -d --build
```

- **Doctor's screen:** http://localhost:8090
- **Hospital dashboard (demo):** http://localhost:8091

Both containers restart by themselves whenever Docker starts, so the scribe is ready before the first patient of the day. On start-up the scribe loads the Whisper model in the background. The very first start downloads it (about 1.6 GB for `large-v3-turbo`; on a slow connection this can take hours, so do it the day before). After that it stays in the `models` volume. `http://localhost:8090/healthz` shows `"speech_model_ready": true` when it is ready.

If Docker or the computer restarts in the middle of a consultation, the visit shows "Interrupted by a restart". The recording is kept, so the doctor presses **Try again**.

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

- **Access codes.** `SCRIBE_ACCESS_CODE` protects the doctor's screen, `ERP_ACCESS_CODE` the dashboard, and `SCRIBE_ERP_TOKEN` the link between the two. Without them, anyone who can reach the ports can read notes, so always set them on a hospital network.
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

No Urdu medical speech model exists yet. Before using this with patients, compare models on our own recorded consultations (see "Measure it" below).

**What we found (18 Nov 2026, one synthetic 17-second clip, so not real-world accuracy):**

| Setting | Script of output | Word error | Character error | Key terms found |
|---|---|---|---|---|
| `base` model | Urdu | 43% | 15% | 5 of 6 |
| `large-v3-turbo` + English medical-word prompt (first version) | **Hindi (Devanagari)** | not comparable | not comparable | 5 of 6 |
| `large-v3-turbo` + short Urdu prompt (current default) | Urdu | **13%** | **4%** | 5 of 6 |

- An English prompt made Whisper write Hindi script.
- A long Urdu prompt full of medical words leaked those words into the transcript: "ڈاکٹر کی گولی" (the doctor's tablet) replaced "آئرن کی گولی" (iron tablet).
- The default is now the short, neutral `SCRIBE_WHISPER_PROMPT`: "ڈاکٹر اور مریض کی گفتگو۔"
- On a MacBook CPU, `large-v3-turbo` runs at about 0.35× real time, so a 10-minute consultation takes about 3.5 minutes.

## Measure it

Two scripts in `eval/` give numbers for the report.

**Speech-to-text, on our own recordings** (clips stay on our machines and are never committed):

```bash
# clips/visit01.wav + visit01.txt (what was really said) + optional visit01.terms (drug names, doses)
docker compose run --rm -v "$PWD/clips:/clips:ro" scribe \
  python -m eval.asr_eval /clips --models large-v3-turbo kingabzpro/whisper-large-v3-urdu-ct2
```

It reports word and character error rate (with Urdu letter variants and digits normalised), key-term recall and speed. If a model writes Hindi script, its error rates against an Urdu reference are meaningless; use key-term recall instead.

**Note quality, on consultations with known facts** (`eval/cases.json`):

```bash
docker compose run --rm scribe python -m eval.note_eval          # needs ANTHROPIC_API_KEY, about 6 calls
```

The cases cover code-mixed speech, Roman Urdu, Urdu script, red flags and denied symptoms, plus a person in the room telling the AI to write "cancer" and prescribe morphine. For each case it scores:
- medicines found and doses correct
- **medicines added that were never said**
- diagnoses, tests and follow-up days
- red flags reported
- denied symptoms kept as absent
- whether any forbidden content got in

**In daily use:** `GET /api/stats` shows how often doctors approve the draft unchanged, and per section what they add (the AI missed it) or remove (the AI got it wrong). The doctor's screen shows the approved-unchanged rate in the sidebar.

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
