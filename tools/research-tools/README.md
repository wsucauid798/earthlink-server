# empirical-tests

A small empirical test harness (plural tests) that runs multiple EarthLink scenarios against a running `earthlink-server` and produces real artifacts (CSV + charts) for the paper.

## Prereqs
- `earthlink-server` running locally (default: `http://localhost:8000`, websocket `ws://localhost:8000/ws/world`).
- Python 3.10+.

## Install
From this folder:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the suite

```bash
python run_suite.py
```

Artifacts are written under `artifacts/<timestamp>/...`.

## Notes
- This harness does **not** modify the server codebase.
- If the server is not reachable, runs will fail fast with a clear error.
