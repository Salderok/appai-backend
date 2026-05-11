# appAi backend

FastAPI + SQLAlchemy + async Postgres backend for the appAi personal AI assistant.

## Quick start

```bash
uv sync                                    # install deps
cp .env.example .env                       # fill in API keys + DEVICE_KEY
uv run uvicorn app.main:app --reload       # http://localhost:8000/docs
```

## Tests

```bash
uv run pytest -q
```

See the repo root `README.md` for the full architecture and roadmap.
