"""Challenge: AI-powered vector search with pgvector (+ pgvectorscale) on Tiger Cloud.

2,200 top Hacker News stories (Jan 2025 - Oct 2026) are embedded locally with the
open-source all-MiniLM-L6-v2 model (384 dims, no API key) and stored in a
TimescaleDB hypertable next to their time-series metadata (time, points,
comments). We then run hybrid searches that combine semantic similarity with
time-range and metadata filters, compare distance metrics, and show a
semantic trend over time - all inside PostgreSQL.

  python vector_search.py            # load + index + run the demo queries
  python vector_search.py --ask "postgres for AI agents" --since 2026-01-01
"""
import argparse, gzip, json, os, time
from datetime import datetime
from pathlib import Path
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent
OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)
DATA = ROOT / "data" / "hn_stories_2025_2026.jsonl.gz"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
L = []
def log(s=""): print(s, flush=True); L.append(s)

def table(cur, sql, params=None, title=None):
    t = time.perf_counter(); cur.execute(sql, params); rows = cur.fetchall(); ms = (time.perf_counter() - t) * 1000
    cols = [d.name for d in cur.description]
    if title: log(f"\n### {title}  _({ms:.0f} ms)_\n")
    log("| " + " | ".join(cols) + " |"); log("|" + "---|" * len(cols))
    for r in rows:
        log("| " + " | ".join("" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v).replace("|", "/")) for v in r) + " |")
    return rows

def setup(cur, model):
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE"); vs = True
    except psycopg.Error:
        vs = False
    cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname IN ('timescaledb','vector','vectorscale') ORDER BY 1")
    exts = cur.fetchall()
    cur.execute("DROP SCHEMA IF EXISTS ghw26 CASCADE; CREATE SCHEMA ghw26")
    cur.execute("""CREATE TABLE ghw26.hn_stories (
        time timestamptz NOT NULL, id bigint NOT NULL, title text NOT NULL, url text, author text,
        points int, num_comments int, embedding vector(384) NOT NULL)""")
    cur.execute("SELECT create_hypertable('ghw26.hn_stories', by_range('time', INTERVAL '1 month'))")
    docs = [json.loads(l) for l in gzip.open(DATA, "rt")]
    t = time.perf_counter()
    embs = model.encode([d["title"] for d in docs], batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    emb_s = time.perf_counter() - t
    t = time.perf_counter()
    with cur.copy("COPY ghw26.hn_stories (time, id, title, url, author, points, num_comments, embedding) FROM STDIN WITH (FORMAT BINARY)") as cp:
        cp.set_types(["timestamptz", "int8", "text", "text", "text", "int4", "int4", "vector"])
        for d, e in zip(docs, embs):
            cp.write_row([datetime.fromisoformat(d["time"].replace("Z", "+00:00")), d["id"], d["title"], d["url"], d["author"], d["points"], d["num_comments"], e])
    load_s = time.perf_counter() - t
    t = time.perf_counter()
    if vs:
        cur.execute("CREATE INDEX hn_emb_diskann ON ghw26.hn_stories USING diskann (embedding vector_cosine_ops)")
        idx = "StreamingDiskANN (pgvectorscale, cosine)"
    else:
        cur.execute("CREATE INDEX hn_emb_hnsw ON ghw26.hn_stories USING hnsw (embedding vector_cosine_ops)")
        idx = "HNSW (pgvector, cosine)"
    idx_s = time.perf_counter() - t
    cur.execute("CREATE INDEX ON ghw26.hn_stories (points)")
    cur.execute("ANALYZE ghw26.hn_stories")
    cur.execute("SELECT count(*), min(time)::date, max(time)::date, (SELECT count(*) FROM show_chunks('ghw26.hn_stories')) FROM ghw26.hn_stories")
    n, a, b, chunks = cur.fetchone()
    log(f"Extensions: {', '.join(f'{e} {v}' for e, v in exts)}  ")
    log(f"Loaded **{n:,}** stories ({a} .. {b}) into hypertable `ghw26.hn_stories` ({chunks} monthly chunks).  ")
    log(f"Embedding with `{MODEL}` (local CPU): {emb_s:.1f}s · binary COPY: {load_s:.1f}s · ANN index **{idx}** built in {idx_s:.1f}s.")
    return idx

def search(cur, model, q, since=None, until=None, min_points=0, k=5, metric="<=>"):
    """Build a hybrid query: ANN ordering on the embedding + optional time-range / metadata filters."""
    where, p = [], {"e": model.encode(q, normalize_embeddings=True), "k": k}
    if since: where.append("time >= %(since)s"); p["since"] = since
    if until: where.append("time < %(until)s"); p["until"] = until
    if min_points: where.append("points >= %(mp)s"); p["mp"] = min_points
    sql = f"""SELECT time::date AS day, points, title, round((embedding {metric} %(e)s)::numeric, 3) AS distance
              FROM ghw26.hn_stories {('WHERE ' + ' AND '.join(where)) if where else ''}
              ORDER BY embedding {metric} %(e)s LIMIT %(k)s"""
    return sql, p

def short_plan(cur, sql, p, n=10):
    cur.execute("EXPLAIN " + sql, p)
    import re
    lines = [re.sub(r"'\[[^']*\]'::vector", "$query_embedding", r[0]) for r in cur.fetchall()]
    return "\n".join(lines[:n]) + ("\n..." if len(lines) > n else "")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ask"); ap.add_argument("--since"); ap.add_argument("--until"); ap.add_argument("--min-points", type=int, default=0)
    a = ap.parse_args()
    model = SentenceTransformer(MODEL)
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        run(conn, model, a)

def run(conn, model, a):
    with conn.cursor() as cur:
        if a.ask:
            sql, p = search(cur, model, a.ask, a.since, a.until, a.min_points, k=8)
            table(cur, sql, p, title=f"Results for: {a.ask}"); return
        cur.execute("SET TimeZone='UTC'")
        log(f"# Hybrid vector + time-series search on Tiger Cloud\n\nRun at {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n\n## 1. Load\n")
        idx = setup(cur, model)
        log("\n## 2. Semantic search (cosine distance `<=>`, ANN index)\n")
        sql, p = search(cur, model, "open-source databases for time-series data")
        table(cur, sql, p, title="Query: \"open-source databases for time-series data\" (all time)")
        log("\nDefault plan (each monthly chunk holds only ~100 rows, so the planner rightly picks an exact scan):\n\n```\n" + short_plan(cur, sql, p, 6) + "\n```")
        cur.execute("SET enable_seqscan = off")
        log("\nWith `enable_seqscan = off` the per-chunk ANN indexes are used and merged in distance order (what happens at scale):\n\n```\n" + short_plan(cur, sql, p, 8) + "\n```")
        cur.execute("RESET enable_seqscan")

        log("\n## 3. Hybrid: semantic + time range + metadata filter\n")
        if idx.startswith("HNSW"):
            cur.execute("SET hnsw.iterative_scan = relaxed_order")  # pgvector 0.8: keep scanning the index until filtered LIMIT is met
            log("`SET hnsw.iterative_scan = relaxed_order` (pgvector 0.8 filtered ANN).")
        else:
            log("pgvectorscale StreamingDiskANN streams candidates, so WHERE filters + LIMIT work without post-filter recall loss.")
        sql, p = search(cur, model, "AI coding agents writing software", since="2026-06-01", min_points=300)
        table(cur, sql, p, title="\"AI coding agents writing software\" · since 2026-06-01 · points >= 300")
        sql, p = search(cur, model, "AI coding agents writing software", until="2025-04-01", min_points=300)
        table(cur, sql, p, title="Same query · before 2025-04-01 · points >= 300")
        log("\nPlan - chunk exclusion on `time` prunes the hypertable to the chunks overlapping Jan-Mar 2025 before any distance is computed:\n\n```\n" + short_plan(cur, sql, p, 12) + "\n```")

        log("\n## 4. Distance metrics compared (same query, top 3 each)\n")
        for name, op in [("cosine `<=>`", "<=>"), ("L2 / Euclidean `<->`", "<->"), ("negative inner product `<#>`", "<#>")]:
            sql, p = search(cur, model, "privacy and government surveillance", k=3, metric=op)
            table(cur, sql, p, title=name)
        log("\n(Embeddings are L2-normalised, so all three metrics rank identically - a good sanity check.)")

        log("\n## 5. Semantic trend over time (vector similarity + time_bucket)\n")
        e = model.encode("ChatGPT, Claude, LLM", normalize_embeddings=True)
        table(cur, """SELECT time_bucket('1 month', time)::date AS month, count(*) AS stories,
                             count(*) FILTER (WHERE embedding <=> %(e)s < 0.70) AS about_llms,
                             round(100.0 * count(*) FILTER (WHERE embedding <=> %(e)s < 0.70) / count(*), 1) AS pct
                      FROM ghw26.hn_stories GROUP BY 1 ORDER BY 1""", {"e": e},
              title="Share of top HN stories semantically close to 'ChatGPT, Claude, LLM' (cosine distance < 0.70)")

        log("\n## 6. RAG retrieval step\n")
        q = "Self-hosting PostgreSQL databases"
        sql, p = search(cur, model, q, since="2025-01-01", k=5)
        rows = table(cur, sql, p, title=f"Context retrieved for: \"{q}\"")
        log("\nContext block that would be passed to an LLM:\n\n```\n" + "\n".join(f"[{r[0]}] ({r[1]} pts) {r[2]}" for r in rows) + "\n```")

        log("\n## 7. Exact vs ANN latency\n")
        e = model.encode("rust programming language", normalize_embeddings=True)
        def lat(setting):
            cur.execute(setting); ts = []
            for _ in range(7):
                t = time.perf_counter(); cur.execute("SELECT id FROM ghw26.hn_stories ORDER BY embedding <=> %s LIMIT 10", (e,)); cur.fetchall(); ts.append((time.perf_counter()-t)*1000)
            return sorted(ts)[3]
        ann = lat("SET enable_seqscan = off"); cur.execute("RESET enable_seqscan")
        exact = lat("SET enable_indexscan = off"); cur.execute("RESET enable_indexscan")
        log(f"| search | median latency (client round-trip) |\n|---|---|\n| ANN index ({idx}) | {ann:.1f} ms |\n| exact scan | {exact:.1f} ms |\n\nAt 2,200 rows both are ~milliseconds; the index matters once the table grows to millions of rows.")
    (OUT / "RESULTS.md").write_text("\n".join(L) + "\n")

if __name__ == "__main__":
    main()
