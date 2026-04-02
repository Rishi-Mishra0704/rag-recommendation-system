# B2B Partnership Recommendation Engine

A RAG-style recommendation system for matching businesses with strategic partners. Given a business profile, the system asks an LLM to describe the ideal partner, then runs hybrid retrieval with semantic vector search and lexical BM25-style full-text search, fuses both ranked lists with Reciprocal Rank Fusion (RRF), and optionally uses the LLM again to explain why each match is a fit.

All inference runs locally through Ollama.

---

## Architecture

```text
Business profile
      |
      v
llama3.1:8b
generates an ideal-partner description
      |
      +------------------------------+
      |                              |
      v                              v
qwen3-embedding                 PostgreSQL full-text query
embeds query text               from the same ideal-partner text
      |                              |
      v                              v
pgvector cosine search          BM25-like ranking with ts_rank
      |                              |
      +--------------+---------------+
                     |
                     v
      Reciprocal Rank Fusion (RRF)
                     |
                     v
     Optional LLM explanations per match
                     |
                     v
          Ranked partner recommendations
```

The retrieval query is not the raw source business profile. During recommendation, the LLM first rewrites the source business into a short description of the ideal partner. That generated text is used in both retrieval branches:

- Vector search over `profile_embedding`
- Full-text search over a generated `search_text` `tsvector`

Each branch applies the same hard filters when provided, and the two ranked lists are merged with RRF (`k=60`). In the current implementation, each branch fetches up to 20 candidates before fusion.

---

## How Profiles Are Represented

At ingestion time, each business is converted into a composite text block and embedded once. The text block is built from:

- `description`
- `industry`
- `sub_industry`
- `roles`
- `tags`
- `trade_regions`
- `partner_goals`

This keeps the stored representation focused on business identity and context, while query-time retrieval is driven by the LLM's description of partner intent.

---

## Stack

- Python 3.11
- PostgreSQL + pgvector
- PostgreSQL full-text search (`tsvector` + `ts_rank`)
- Ollama

---

## Project Structure

```text
recommendation/
  common.py                 shared DB connection and embedding helper
  constants/
    constants.py            config, model names, SQL constants
  logic/
    embed_service.py        build profile text blocks, embed, insert into DB
    recommendations.py      end-to-end recommendation orchestration
    search.py               vector search, BM25 search, RRF merge
  data_gen/
    transform.py            CSV -> business profile pipeline
    llm_gen.py              LLM prompt construction and text generation
    mappings.py             industry and trade-region mapping helpers
    schemas.py              Pydantic business schema and enums
    validate.py             validate generated profiles
  evaluation/
    eval.py                 evaluation runner and filter tests
    eval_scoring.py         heuristic metrics and precision/recall helpers
    eval_report.py          summary printer and JSON report writer
sql/
  schema.sql                businesses table and supporting indexes
api/
  main.go                   placeholder for a future API layer
data/
  profiles.json             generated business profiles
  profiles_validated.json   validated business profiles
```

---

## Prerequisites

- Python 3.11+
- Docker
- [Ollama](https://ollama.com) running locally with both models pulled

```bash
ollama pull llama3.1:8b
ollama pull qwen3-embedding
```

---

## Models

| Role | Model |
|---|---|
| Query generation + match explanations | `llama3.1:8b` |
| Embeddings | `qwen3-embedding` (4096-dim) |

If you change the embedding model, update the vector dimension in `sql/schema.sql` and re-run the embedding pipeline so all stored business vectors are regenerated.

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

**3. Apply the schema**

```bash
psql -h localhost -p 5433 -U raguser -d rag_recommendation -f sql/schema.sql
```

**4. Get the dataset**

Download the [People Data Labs Company Dataset](https://www.kaggle.com/datasets/peopledatalabssf/free-7-million-company-dataset) and place it at:

```text
data/companies_sorted.csv
```

---

## Usage

Run each step from the repo root.

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

**Embed profiles and insert them into PostgreSQL**

```bash
python -m recommendation.logic.embed_service
```

**Generate recommendations for a business**

```bash
python -m recommendation.logic.recommendations <business_id>
```

The current CLI accepts a business ID and returns fused hybrid-search recommendations. Filtering by `trade_type`, `category`, and `roles` is supported inside `recommend(...)` programmatically, even though those flags are not yet exposed on the command line.

**Run evaluation**

```bash
python -m recommendation.evaluation.eval
```

This runs recommendation-quality heuristics, simple precision/recall where ground truth is available, and filter-correctness checks. Results are printed to stdout and written to `recommendation/evaluation/eval_results.json`.

---

## Retrieval Details

- `vector_search(...)` ranks businesses by cosine distance on `profile_embedding`
- `bm25_search(...)` ranks businesses with PostgreSQL full-text search using `ts_rank`
- `rrf_merge(...)` combines both lists with Reciprocal Rank Fusion
- Source businesses are excluded from their own result set
- Optional hard filters currently supported in the search layer:
  - `trade_type`
  - `category`
  - `roles` overlap

The database schema also defines a generated `search_text` column so the same stored business record can participate in both semantic and lexical retrieval.

---

## Roadmap

- **Cross-encoder reranking** — re-score top-N candidates with a dedicated reranker
- **Agentic loop** — runtime decision-making over search strategy and filter selection
- **Go API layer** — Echo-based HTTP API wrapping the recommendation engine
- **Latency optimization** — profile the end-to-end pipeline and cut inference time