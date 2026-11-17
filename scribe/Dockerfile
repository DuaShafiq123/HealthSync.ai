FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models
WORKDIR /app

COPY requirements.txt requirements-asr.txt ./
RUN pip install -r requirements.txt -r requirements-asr.txt

COPY app ./app
COPY erp_mock ./erp_mock

# never run as root
RUN useradd --create-home scribe && mkdir -p /app/data /models && chown -R scribe /app/data /models
USER scribe

EXPOSE 8090 8091
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090"]
