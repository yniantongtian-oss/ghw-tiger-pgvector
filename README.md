# ghw-tiger-pgvector — hybrid semantic + time-series search in one Postgres

MLH Global Hack Week: Builders Week · Tiger Data Challenge 5 *"Implement AI-Powered Vector Search with pgvector"*.

**Data:** the top 100 Hacker News stories of every month from Jan 2025 to Oct 2026 (2,200 stories, `data/hn_stories_2025_2026.jsonl.gz`,
from the public HN Algolia API) — text plus time-series metadata (timestamp, points, comments).

**Embeddings:** computed locally with the open-source `sentence-transformers/all-MiniLM-L6-v2` model (384-d, CPU, no paid API key).

**Storage:** one TimescaleDB **hypertable** `ghw26.hn_stories (time, title, points, …, embedding vector(384))` with monthly chunks. The
script enables `pgvector` and `pgvectorscale`; it builds a **StreamingDiskANN** index (pgvectorscale) when the extension is available
(Tiger Cloud) and falls back to pgvector **HNSW** elsewhere.

`vector_search.py` then runs, entirely inside PostgreSQL:

1. Pure semantic search (cosine `<=>`) + query plans (exact scan while chunks are tiny; per-chunk ANN index scans merged in distance order at scale).
2. **Hybrid search**: semantic similarity + time-range filter + metadata filter (`points >= 300`) — same query, two eras, different answers;
   the plan shows chunk exclusion pruning months before any distance is computed.
3. Distance metrics compared: cosine `<=>`, Euclidean `<->`, negative inner product `<#>`.
4. **Semantic trend over time**: `time_bucket('1 month')` × vector similarity — what share of top stories each month is about LLMs.
5. RAG retrieval step: top-k context block for an LLM prompt.
6. ANN vs exact-scan latency.

Results: [`results/RESULTS.md`](results/RESULTS.md). Ad-hoc search:
`python vector_search.py --ask "postgres for AI agents" --since 2026-01-01 --min-points 200`

No separate vector database — vectors, time-series and metadata live in the same table and the same SQL query.

## Run it
```bash
cp .env.example .env && export $(grep -v '^#' .env | xargs)
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: small CPU-only torch
pip install -r requirements.txt
python vector_search.py
```
`ci/run-live.yml` runs it in GitHub Actions against the live service (secret: `DATABASE_URL`). Copy it to `.github/workflows/` to enable it.

## Stack
Tiger Cloud (TimescaleDB) · pgvector · pgvectorscale (StreamingDiskANN) · hypertables · sentence-transformers all-MiniLM-L6-v2 · Python · psycopg 3
