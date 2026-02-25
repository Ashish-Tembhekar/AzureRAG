# Azure RAG Flask Simulator

Dummy RAG app to evaluate Azure AI Search migration behavior with:
- Document upload and chunk ingestion
- Chat with NVIDIA-hosted OpenAI-compatible LLM
- Azure FREE/BASIC tier simulation and cost telemetry

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Create `.env` from `.env.example` and set `NVIDIA_API_KEY`.
4. Run:
   - `python app.py`
5. Open:
   - `http://localhost:5000`

## API Endpoints

- `GET /api/health`: status + config + document store stats
- `GET /api/config`: read active runtime config
- `POST /api/config`: update runtime config, pricing, and tier
- `POST /api/upload`: upload document (`multipart/form-data`, field `file`, optional `index_name`)
- `POST /api/chat`: ask a RAG question
- `POST /api/reset`: reset session costs/counters/history

## Cost Summary Format

Every successful ingestion or chat operation includes:

```json
{
  "query_cost_usd": 0.0,
  "total_session_cost_usd": 0.0,
  "semantic_queries_used": 0,
  "storage_used_mb": 0.0,
  "limit_status": "FREE_OK"
}
```
