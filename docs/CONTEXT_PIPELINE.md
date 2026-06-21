# Advanced Context Engineering Pipeline

This document explains the **modular RAG pipeline** under `context_pipeline/`, how it improves retrieval and prompting, and how to run it for demos.

## Folder structure

```text
RAG model/
├── context_pipeline/          # Advanced pipeline (this feature)
│   ├── __init__.py            # Public exports
│   ├── __main__.py            # python -m context_pipeline → demo CLI
│   ├── config.py              # Env-driven PipelineConfig dataclass
│   ├── logging_utils.py       # JSON-line stderr logging per stage
│   ├── hybrid_retrieval.py    # Weaviate hybrid oversampling
│   ├── deduplication.py       # Jaccard near-dup removal
│   ├── reranking.py           # Bi-encoder cosine (or lexical fallback)
│   ├── memory_decay.py        # Exponential decay + history trimming
│   ├── compression.py         # Extractive sentence compression
│   ├── token_budget.py        # Token slices + Groq message assembly
│   ├── pipeline.py            # Orchestrates all stages
│   └── run_demo.py            # CLI entry for instructors
├── vector_store.py            # Weaviate + FastEmbed indexing & hybrid search
├── rag_groq.py                # Chunking, BM25, Groq helpers
├── web_app.py                 # Flask UI (set RAG_CONTEXT_PIPELINE=true)
├── docs/                      # Sample .md / .txt corpora
├── docker-compose.yml         # Local Weaviate
├── requirements.txt
└── .env                       # Secrets + pipeline toggles
```

## Pipeline flow (end-to-end)

1. **Query processing** — normalize whitespace (`pipeline._normalize_query`).
2. **Hybrid retrieval** — Weaviate `hybrid()` fuses **dense vectors** and **BM25**; we **oversample** (`RAG_RETRIEVE_OVERSAMPLE × RAG_TOP_K`) so later stages have more to work with.
3. **Deduplication** — Jaccard similarity on word bags; drops near-duplicate chunks (saves tokens, reduces repetition bias).
4. **Re-ranking** — Re-score candidates with **query vs snippet embeddings** (same FastEmbed model as indexing). BM25-only mode falls back to lexical overlap.
5. **Memory decay** — Older chat turns get lower exponential weights; lowest-weight turns are dropped first until history fits its token slice.
6. **Context compression** — Keeps the best sentences per chunk vs query overlap (extractive, fast).
7. **Token budget allocation** — Splits the **input** window across system / history / documents / query using `tiktoken` (or `len/4` fallback).
8. **Groq** — `answer_with_groq_messages` sends the assembled chat payload.

## Why hybrid search helps

- **Vectors** capture paraphrases (“disk full” ≈ “no free space”).
- **BM25** captures rare tokens (SKUs, error codes, product names).
- **Weighted hybrid** (`RAG_HYBRID_ALPHA`) lets you bias toward semantics vs keywords per domain.

## Why re-ranking helps

First-stage retriever optimizes **recall** (surface anything plausibly relevant). A reranker optimizes **precision at the top** so the LLM reads the *most* on-topic paragraphs first—critical with small `top_k`.

## Memory decay formula

For each prior turn (when `ts` is stored on the message):

`weight = exp(-λ × age_hours)`

Without timestamps, synthetic ages increase toward the **start** of the list so recent UI turns behave like “fresh memory.”

## Compression (before vs after)

- **Before:** long chunk with repeated boilerplate.
- **After:** fewer sentences, highest overlap with the user query — fewer tokens, similar factual coverage (extractive, not generative).

## Installation

```powershell
cd "c:\Users\rocks\Desktop\RAG model"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` → `.env` and set `GROQ_API_KEY` (and Weaviate variables if using cloud).

## Docker (Weaviate local)

```powershell
docker compose up -d
```

Health check: `http://localhost:8080/v1/.well-known/ready`

## Running the demo CLI

```powershell
$env:RAG_CONTEXT_PIPELINE="true"   # optional; demo always uses pipeline module
python -m context_pipeline.run_demo "Compare Windows and Linux for developers"
```

Stderr shows JSON **stage** logs when `RAG_PIPELINE_DEBUG=true`.

## Running the web UI with the pipeline

In `.env`:

```env
RAG_CONTEXT_PIPELINE=true
```

Then:

```powershell
python web_app.py
```

Open `http://127.0.0.1:7860`. Retrieval mode **vector** uses Weaviate hybrid; **bm25** uses the in-memory lexical path (reranker falls back to lexical scores).

## Example questions (sample `docs/`)

- “What is Weaviate used for?”
- “Who develops Zoho and what products are listed?”
- “Compare Linux and Windows security in the operating systems doc.”

## Environment variables (pipeline)

See `.env.example` — keys prefixed with `RAG_` for retrieval, `RAG_TOKEN_*` for budget fractions, `RAG_CONTEXT_PIPELINE` for Flask integration.
