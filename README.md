# B2B Partnership Recommendation Engine

A RAG-based system that matches businesses with strategic partners. Given a business profile, it uses an LLM to reason about what an ideal partner looks like, embeds that reasoning, runs a cosine similarity search over a pgvector index, then uses the LLM again to explain each match.

All inference runs locally via Ollama — no external API calls.

---

## Architecture

```
Business Profile
      │
      ▼
 llama3.1:8b
 reasons about ideal partner characteristics
      │
      ▼
 qwen3-embedding (4096-dim)
 embeds the LLM's partner description
      │
      ▼
 pgvector HNSW cosine search
 + hard metadata filters (trade_type, category, roles)
      │
      ▼
 llama3.1:8b
 explains why each match is a good fit
      │
      ▼
 Ranked matches with reasoning
```

The LLM does not embed raw profiles directly. Each business profile is stored as a composite text block (`description + industry + roles + tags + trade_regions + partner_goals`), embedded at ingestion time. At query time, the LLM reasons about what an ideal partner would look like — that reasoning is what gets embedded and searched. This separates "what this business is" from "what it needs."

**Stack:** Python 3.11, PostgreSQL + pgvector, Ollama

---

## Project Structure

```
recommendation/
  common.py                 shared connect_db() and get_embedding()
  constants/
    constants.py            all config: DB, Ollama URLs, model names
  logic/
    embed_service.py        build text blocks, embed profiles, insert into DB
    recommendations.py      RAG recommendation engine (the full 5-step flow)
  data_gen/
    transform.py            CSV → Business profile pipeline (orchestrator)
    llm_gen.py              LLM prompt construction and text generation
    mappings.py             industry tag map, country → trade region map
    schemas.py              Pydantic Business model, enums, sub-industry map
    validate.py             validate profiles.json against the Business schema
  evaluation/
    eval.py                 evaluation orchestrator: run_eval, filter tests
    eval_scoring.py         heuristic scorers, precision@k, recall@k
    eval_report.py          print_summary, dump_results, thresholds
sql/
  schema.sql                businesses table, HNSW vector index, GIN/B-tree indexes
data/
  companies_sorted.csv      input dataset (not in repo — see Setup)
  profiles.json             generated profiles (not in repo — generated locally)
```

---

## Prerequisites

- Python 3.11+
- Docker
- [Ollama](https://ollama.com) running locally with both models pulled:

```bash
ollama pull llama3.1:8b
ollama pull qwen3-embedding
```

---

## Models

| Role | Model |
|---|---|
| Reasoning + explanations | `llama3.1:8b` |
| Embeddings | `qwen3-embedding` (4096-dim) |

Both are heavy models chosen for quality during development. For faster iteration, you can swap in a smaller LLM (e.g. `llama3.2:3b`) or a lower-dimension embedding model. If you change the embedding model, update the vector dimension in `sql/schema.sql` and re-run `embed_service.py` to re-embed all profiles.

---

## Setup

**1. Clone and create a virtual environment**

```bash
git clone <repo>
cd rag-recommendation-system
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Start PostgreSQL with pgvector**

```bash
docker run -d \
  --name rag-postgres \
  -e POSTGRES_USER=raguser \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=rag_recommendation \
  -p 5433:5432 \
  pgvector/pgvector:pg16
```

**3. Run the schema**

```bash
psql -h localhost -p 5433 -U raguser -d rag_recommendation -f sql/schema.sql
```

**4. Get the dataset**

Download the [People Data Labs Company Dataset](https://www.kaggle.com/datasets/peopledatalabssf/free-7-million-company-dataset) from Kaggle and place it at:

```
data/companies_sorted.csv
```

---

## Usage

Run each step in order from the repo root.

**Generate business profiles from the CSV**

```bash
python -m recommendation.data_gen.transform \
  --csv data/companies_sorted.csv \
  --count 200 \
  --output data/profiles.json
```

**Validate generated profiles**

```bash
python -m recommendation.data_gen.validate \
  --input data/profiles.json \
  --output data/profiles_validated.json
```

**Embed profiles and insert into the database**

```bash
python -m recommendation.logic.embed_service
```

**Get recommendations for a business**

```bash
python -m recommendation.logic.recommendations <business_id>
```

Optional filters: `--trade_type domestic|international|both`, `--category manufacturer|distributor|...`

**Run evaluation**

```bash
python -m recommendation.evaluation.eval
```

Results are printed to stdout and written to `recommendation/evaluation/eval_results.json`.

---

## Eval Results

Heuristic metrics averaged over 20 queries (top-5 results each):

| Metric | Score | Threshold | Result |
|---|---|---|---|
| Industry match rate | 0.84 | ≥ 0.70 | PASS |
| Role compatibility | 0.91 | ≥ 0.80 | PASS |
| Trade type compatibility | 0.76 | ≥ 0.70 | PASS |
| Category match rate | 0.17 | ≤ 0.30 | PASS |
| Trade region overlap | — | neutral | — |

Filter correctness: **4/4 passing** (`trade_type=domestic`, `trade_type=international`, `category=manufacturer`, `category=distributor`).

Category match is intentionally low — same-category matches indicate the system is finding complementary partners (e.g. a supplier paired with a distributor), not clones.

---

## Roadmap

- **Hybrid search** — BM25 + vector with RRF fusion
- **Cross-encoder reranking** — re-score top-N candidates with a dedicated reranker
- **Agentic loop** — runtime decision-making over search strategy and filter selection
- **Go API layer** — Echo-based HTTP API wrapping the recommendation engine
- **Latency optimization** — profile the end-to-end pipeline and cut inference time
