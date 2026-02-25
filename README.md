# Azure RAG Flask App (Real Azure AI Search)

Flask-based RAG evaluation app wired to a **real Azure AI Search service** and NVIDIA-hosted OpenAI-compatible LLM inference.

Key capabilities:
- Real Azure Search index creation, upload, and retrieval (no mock DB/search)
- Document management UI (add/list/delete documents)
- Chat with retrieved context + NVIDIA LLM
- Free-tier firewall (`AZURE_TIER=FREE`) that blocks quota-breaking operations before SDK execution
- Per-operation cost and telemetry summaries

## Architecture

- `app.py`: Flask routes and runtime config
- `services/document_store.py`: Azure Search SDK integration
- `services/azure_simulator.py`: capacity monitor + cost evaluator
- `services/llm_service.py`: NVIDIA OpenAI-compatible client
- `templates/index.html`, `static/app.js`, `static/styles.css`: frontend

## Free Tier Firewall Rules

When `AZURE_TIER=FREE`, the app enforces:
- Max indexes: `3`
- Max documents per index: `1000`
- Max storage per index: `50 MB`
- Max semantic ranker queries: `1000 / month`

Pre-flight checks happen before:
- document uploads (`upload_documents`)
- semantic/agentic queries

If a limit would be exceeded, `AzureQuotaExceededError` is raised and the operation is blocked.

## Prerequisites

- Python 3.10+
- Azure CLI (`az`) logged in
- Azure AI Search service endpoint + admin key
- NVIDIA API key for LLM calls

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Create `.env` from `.env.example` and set:
   - `NVIDIA_API_KEY`
   - `AZURE_SEARCH_ENDPOINT`
   - `AZURE_SEARCH_ADMIN_KEY`
4. Run:
   - `python app.py`
5. Open:
   - `http://localhost:5000`

## Environment Variables

- `NVIDIA_API_KEY`: NVIDIA inference API key
- `NVIDIA_MODEL`: model name for chat (default from `.env.example`)
- `AZURE_SEARCH_ENDPOINT`: `https://<service>.search.windows.net`
- `AZURE_SEARCH_ADMIN_KEY`: admin key for Azure Search
- `AZURE_TIER`: `FREE` or `BASIC`
- `AZURE_LLM_INPUT_COST_PER_1K`: input token price
- `AZURE_LLM_OUTPUT_COST_PER_1K`: output token price
- `AZURE_SEMANTIC_QUERY_COST`: semantic query price per request
- `PORT`: Flask port

## UI Workflow

### Documents
- Add document (upload file to an index)
- Refresh and view available uploaded documents (grouped by `source_name`)
- Select one or more documents and delete

### Chat
- Query an index with lexical/vector search
- Optional semantic ranker toggle for agentic mode
- Uses retrieved Azure Search chunks as context for NVIDIA LLM

### Configuration
- Switch `AZURE_TIER` (`FREE`/`BASIC`)
- Tune chunking, embedding dimensions, model, and pricing inputs

## API Endpoints

- `GET /api/health`: status + runtime config + store stats
- `GET /api/config`: get current runtime config
- `POST /api/config`: update runtime config/tier/pricing
- `POST /api/upload`: add document (`multipart/form-data`, `file`, optional `index_name`)
- `GET /api/documents`: list documents by index (`index_name` query param)
- `DELETE /api/documents`: delete selected documents by source names
- `POST /api/chat`: run RAG query + LLM response
- `POST /api/reset`: reset session cost/counters/history

## Cost & Telemetry Output

Each successful ingestion/chat operation returns:

```json
{
  "query_cost_usd": 0.0,
  "total_session_cost_usd": 0.0,
  "semantic_queries_used": 0,
  "storage_used_mb": 0.0,
  "limit_status": "FREE_OK"
}
```

## Notes

- A local file `.semantic_query_counter.json` is used for monthly semantic query counting.
- Current vector embedding is deterministic/local for evaluation of Azure vector retrieval flow; retrieval still runs against real Azure Search.
