# Hybrid vector + time-series search on Tiger Cloud

Run at 2026-11-09 08:00:44 UTC

## 1. Load

Extensions: timescaledb 2.31.0, vector 0.8.7, vectorscale 0.9.0  
Loaded **2,200** stories (2025-01-01 .. 2026-10-31) into hypertable `ghw26.hn_stories` (23 monthly chunks).  
Embedding with `sentence-transformers/all-MiniLM-L6-v2` (local CPU): 3.8s · binary COPY: 1.9s · ANN index **StreamingDiskANN (pgvectorscale, cosine)** built in 1.4s.

## 2. Semantic search (cosine distance `<=>`, ANN index)


### Query: "open-source databases for time-series data" (all time)  _(118 ms)_

| day | points | title | distance |
|---|---|---|---|
| 2025-01-01 | 625 | Databases in 2024: A Year in Review | 0.480 |
| 2025-10-07 | 630 | Show HN: Timelinize – Privately organize your own data from everywhere, locally | 0.496 |
| 2026-01-05 | 717 | Databases in 2025: A Year in Review | 0.501 |
| 2026-01-12 | 737 | TimeCapsuleLLM: LLM trained only on data from 1800-1875 | 0.589 |
| 2025-01-21 | 1021 | Stargate Project: SoftBank, OpenAI, Oracle, MGX to build data centers | 0.612 |

Default plan (each monthly chunk holds only ~100 rows, so the planner rightly picks an exact scan):

```
Limit  (cost=46.70..49.69 rows=5 width=94)
  ->  Result  (cost=46.70..1364.96 rows=2200 width=94)
        ->  Merge Append  (cost=46.70..1315.46 rows=2200 width=1602)
              Sort Key: ((hn_stories.embedding <=> $query_embedding))
              ->  Index Scan using _hyper_6_213_chunk_hn_emb_diskann on _hyper_6_213_chunk  (cost=2.01..59.11 rows=110 width=1601)
                    Order By: (embedding <=> $query_embedding)
...
```

With `enable_seqscan = off` the per-chunk ANN indexes are used and merged in distance order (what happens at scale):

```
Limit  (cost=46.70..49.69 rows=5 width=94)
  ->  Result  (cost=46.70..1364.96 rows=2200 width=94)
        ->  Merge Append  (cost=46.70..1315.46 rows=2200 width=1602)
              Sort Key: ((hn_stories.embedding <=> $query_embedding))
              ->  Index Scan using _hyper_6_213_chunk_hn_emb_diskann on _hyper_6_213_chunk  (cost=2.01..59.11 rows=110 width=1601)
                    Order By: (embedding <=> $query_embedding)
              ->  Index Scan using _hyper_6_214_chunk_hn_emb_diskann on _hyper_6_214_chunk  (cost=2.01..16.29 rows=28 width=1603)
                    Order By: (embedding <=> $query_embedding)
...
```

## 3. Hybrid: semantic + time range + metadata filter

pgvectorscale StreamingDiskANN streams candidates, so WHERE filters + LIMIT work without post-filter recall loss.

### "AI coding agents writing software" · since 2026-06-01 · points >= 300  _(112 ms)_

| day | points | title | distance |
|---|---|---|---|
| 2026-06-13 | 1603 | Open source AI must win | 0.420 |
| 2026-10-12 | 808 | AI is a tool and using it well takes skill | 0.431 |
| 2026-08-20 | 1061 | Don't paste the AI, please | 0.485 |
| 2026-08-12 | 1014 | AI is removing the middle class of software engineering? | 0.490 |
| 2026-08-27 | 1035 | CEO fired developers to make room for AI. Developers create open source AI CEO | 0.494 |

### Same query · before 2025-04-01 · points >= 300  _(111 ms)_

| day | points | title | distance |
|---|---|---|---|
| 2025-03-11 | 645 | RubyLLM: A delightful Ruby way to work with AI | 0.384 |
| 2025-02-11 | 741 | Firing programmers for AI is a mistake | 0.430 |
| 2025-03-11 | 749 | Show HN: Factorio Learning Environment – Agents Build Factories | 0.543 |
| 2025-03-19 | 577 | AI Blindspots – Blindspots in LLMs I've noticed while AI coding | 0.562 |
| 2025-03-07 | 601 | AI tools are spotting errors in research papers | 0.570 |

Plan - chunk exclusion on `time` prunes the hypertable to the chunks overlapping Jan-Mar 2025 before any distance is computed:

```
Limit  (cost=8.07..11.14 rows=5 width=94)
  ->  Result  (cost=8.07..191.65 rows=299 width=94)
        ->  Merge Append  (cost=8.07..184.92 rows=299 width=1602)
              Sort Key: ((hn_stories.embedding <=> $query_embedding))
              ->  Index Scan using _hyper_6_213_chunk_hn_emb_diskann on _hyper_6_213_chunk  (cost=2.01..59.38 rows=110 width=1601)
                    Order By: (embedding <=> $query_embedding)
                    Filter: (points >= '300'::smallint)
              ->  Index Scan using _hyper_6_214_chunk_hn_emb_diskann on _hyper_6_214_chunk  (cost=2.01..16.36 rows=28 width=1603)
                    Order By: (embedding <=> $query_embedding)
                    Filter: (points >= '300'::smallint)
              ->  Index Scan using _hyper_6_215_chunk_hn_emb_diskann on _hyper_6_215_chunk  (cost=2.01..55.27 rows=101 width=1603)
                    Order By: (embedding <=> $query_embedding)
...
```

## 4. Distance metrics compared (same query, top 3 each)


### cosine `<=>`  _(117 ms)_

| day | points | title | distance |
|---|---|---|---|
| 2026-02-15 | 937 | Amazon's Ring and Google's Nest reveal the severity of U.S. surveillance state | 0.403 |
| 2026-09-15 | 964 | 25 years of mass surveillance is enough | 0.406 |
| 2026-03-15 | 1005 | Bill C-22, the Lawful Access Act: Dangerous backdoor surveillance risks remain | 0.411 |

### L2 / Euclidean `<->`  _(113 ms)_

| day | points | title | distance |
|---|---|---|---|
| 2026-02-15 | 937 | Amazon's Ring and Google's Nest reveal the severity of U.S. surveillance state | 0.897 |
| 2026-09-15 | 964 | 25 years of mass surveillance is enough | 0.901 |
| 2026-03-15 | 1005 | Bill C-22, the Lawful Access Act: Dangerous backdoor surveillance risks remain | 0.907 |

### negative inner product `<#>`  _(113 ms)_

| day | points | title | distance |
|---|---|---|---|
| 2026-02-15 | 937 | Amazon's Ring and Google's Nest reveal the severity of U.S. surveillance state | -0.597 |
| 2026-09-15 | 964 | 25 years of mass surveillance is enough | -0.594 |
| 2026-03-15 | 1005 | Bill C-22, the Lawful Access Act: Dangerous backdoor surveillance risks remain | -0.589 |

(Embeddings are L2-normalised, so all three metrics rank identically - a good sanity check.)

## 5. Semantic trend over time (vector similarity + time_bucket)


### Share of top HN stories semantically close to 'ChatGPT, Claude, LLM' (cosine distance < 0.70)  _(113 ms)_

| month | stories | about_llms | pct |
|---|---|---|---|
| 2025-01-01 | 100 | 1 | 1.0 |
| 2025-02-01 | 100 | 2 | 2.0 |
| 2025-03-01 | 100 | 4 | 4.0 |
| 2025-04-01 | 100 | 0 | 0.0 |
| 2025-05-01 | 100 | 6 | 6.0 |
| 2025-06-01 | 100 | 2 | 2.0 |
| 2025-07-01 | 100 | 7 | 7.0 |
| 2025-08-01 | 100 | 9 | 9.0 |
| 2025-09-01 | 100 | 9 | 9.0 |
| 2025-10-01 | 100 | 10 | 10.0 |
| 2025-11-01 | 100 | 4 | 4.0 |
| 2025-12-01 | 100 | 3 | 3.0 |
| 2026-01-01 | 100 | 4 | 4.0 |
| 2026-02-01 | 100 | 8 | 8.0 |
| 2026-03-01 | 100 | 5 | 5.0 |
| 2026-04-01 | 100 | 9 | 9.0 |
| 2026-05-01 | 100 | 4 | 4.0 |
| 2026-06-01 | 100 | 11 | 11.0 |
| 2026-07-01 | 100 | 6 | 6.0 |
| 2026-08-01 | 100 | 5 | 5.0 |
| 2026-09-01 | 100 | 5 | 5.0 |
| 2026-10-01 | 100 | 2 | 2.0 |

## 6. RAG retrieval step


### Context retrieved for: "Self-hosting PostgreSQL databases"  _(117 ms)_

| day | points | title | distance |
|---|---|---|---|
| 2025-12-20 | 683 | Go ahead, self-host Postgres | 0.207 |
| 2025-07-19 | 612 | My Self-Hosting Setup | 0.412 |
| 2025-04-29 | 608 | Jepsen: Amazon RDS for PostgreSQL 17.4 | 0.591 |
| 2025-11-30 | 662 | Self-hosting my photos with Immich | 0.603 |
| 2025-06-06 | 1634 | Self-hosting your own media considered harmful according to YouTube | 0.614 |

Context block that would be passed to an LLM:

```
[2025-12-20] (683 pts) Go ahead, self-host Postgres
[2025-07-19] (612 pts) My Self-Hosting Setup
[2025-04-29] (608 pts) Jepsen: Amazon RDS for PostgreSQL 17.4
[2025-11-30] (662 pts) Self-hosting my photos with Immich
[2025-06-06] (1634 pts) Self-hosting your own media considered harmful according to YouTube
```

## 7. Exact vs ANN latency

| search | median latency (client round-trip) |
|---|---|
| ANN index (StreamingDiskANN (pgvectorscale, cosine)) | 117.0 ms |
| exact scan | 111.1 ms |

At 2,200 rows both are ~milliseconds; the index matters once the table grows to millions of rows.
