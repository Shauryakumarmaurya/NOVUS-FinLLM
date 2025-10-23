# giga-finanalytix

MVP backend that ingests earnings call PDFs, scrapes financial statement data, calls Gemini for qualitative analysis, and assembles a background-generated report using Redis + RQ.

## Components

- `app.py`: Flask API (enqueue report job, check status, health)
- `tasks.py`: Long-running report generation with progress meta
- `redis_config.py`: Lazy Redis + Queue factory (supports `REDIS_URL` or host/port)
- `logic.py`: Financial parsing, scraping, Gemini prompt logic
- `static/`: Minimal frontend placeholder

## Why RQ / background job?
Render / typical PaaS impose ~30s request timeouts. Offloading the heavy multi-step Gemini + scraping workflow to a worker avoids request timeout failures and gives progress visibility.

## Quick Start (Local)

```cmd
REM 1. (Optional) Create virtualenv
python -m venv .venv
call .venv\Scripts\activate

REM 2. Install deps
pip install -r requirements.txt

REM 3. Run Redis (Docker example)
docker run -p 6379:6379 redis:7-alpine

REM 4. Set env vars
set GEMINI_API_KEY=your_key
set FMP_API_KEY=your_key
set REDIS_HOST=localhost

REM 5. Start Flask API
python app.py

REM 6. In another terminal start an RQ worker
rq worker financial_analysis
```

## API

### POST /generate_report
Multipart form-data:
- `ticker`: (string) Screener.in ticker (e.g. TCS)
- `files`: 1..N PDF uploads

Response:
```json
{ "job_id": "<uuid>", "status": "queued" }
```

### GET /job_status/<job_id>
Returns job state and (on completion) the report payload. Includes `progress.stage` while running.

### GET /health
Simple Redis connectivity probe.

## Progress Meta
`tasks.py` updates `job.meta['stage']` through stages:
```
extract_pdfs → fetch_financials → business_model → quarterly_updates → management_commentary → risks → prompt_set → assumptions → projections → assemble
```

If JSON parsing of assumptions fails, an `assumptions_parse_error` is stored.

## Deployment Tips
- Set `REDIS_URL` in production if using a hosted Redis.
- Use `WEB_CONCURRENCY=1` (or more) for gunicorn, but keep worker separate.
- Scale workers based on throughput: each job runs multiple Gemini calls + scraping, so start small.

## Future Enhancements (ideas)
- Stream partial markdown sections as they finish (Server-Sent Events)
- Add caching layer for financial statement HTML
- Persist final report to a database (for retrieval without Redis job retention)
- Add authentication / API keys for external use
- Replace sleeps with rate-limit aware wrapper + exponential backoff

## License
Internal MVP (add a proper license if open-sourcing).

