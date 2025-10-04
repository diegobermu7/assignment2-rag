# Assignment 2 - Retrieval-Augmented Generation

## Overview
This repository explores the evolution of a retrieval-augmented generation (RAG) system from a baseline pipeline to a more production-ready stack. It pairs Milvus Lite vector search with Hugging Face models, adds re-ranking and confidence estimation, and evaluates outputs with classical metrics and RAGAS.

## Repository Layout
- `src/naive_rag.py` - embeds passages, indexes them in Milvus Lite, retrieves top-*k* contexts, and generates answers with a seq2seq model.
- `src/enhanced_rag.py` - extends the naive pipeline with candidate over-retrieval, cross-encoder re-ranking, confidence-based abstention, and richer output metadata.
- `src/evaluation.py` - computes per-question F1, unknown-rate statistics, confidence intervals, and Hugging Face SQuAD metrics.
- `src/ragas_evaluation.py` - prepares datasets for RAGAS, shards evaluations across threads, and aggregates weighted metric summaries.
- `src/utils.py` - shared utilities for loading data, embedding text, managing Milvus Lite collections, prompt templating, and generation.
- `notebooks/` - exploratory analysis and system evaluation notebooks referenced in the write-up.
- `data/` - processed passages and evaluation splits used by the pipelines.
- `results/` - default destination for JSON predictions, comparison tables, and RAGAS summaries.

## Setup Instructions
1. **Prerequisites**
   - Python 3.10+
   - `pip` and a C++ build toolchain (Milvus Lite and sentence-transformers rely on native extensions).
   - (Optional) GPU with CUDA for faster generation.
2. **Create an isolated environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   The list is intentionally exhaustive (notebook + RAG tooling). For a lighter runtime you can install only the core packages: `sentence-transformers`, `transformers`, `pymilvus`, `datasets`, `ragas`, `langchain-openai`, `tqdm`, `pandas`, `scipy`.
4. **Configure API keys**
   - Copy `.env.example` to `.env` (or edit `.env`) and set `OPENAI_API_KEY` for RAGAS evaluations that rely on `gpt-4o-mini`.
   - Export `OPENAI_API_KEY` in your shell when running RAGAS from the CLI: `export OPENAI_API_KEY=...`.
5. **Prepare Milvus Lite state (optional)**
   - Default runs store the collection locally in `rag_wikipedia_mini.db`. Delete the file or pass `--milvus_drop` to rebuild from scratch.

## Running the Pipelines
### Naive Baseline
```bash
python -m src.naive_rag \
  --passages_uri data/processed/passages.csv \
  --queries_uri data/evaluation/test_dataset.csv \
  --embed_model sentence-transformers/all-MiniLM-L6-v2 \
  --gen_model google/flan-t5-base \
  --top_k 3 \
  --output_file naive_results.json
```
- Results are written to `results/naive_results.json` (relative to repo root).
- Set `--milvus_drop` on the first run to ensure a fresh Milvus Lite collection.

### Enhanced Pipeline
```bash
python -m src.enhanced_rag \
  --retrieve_candidates 50 \
  --top_k 5 \
  --rerank_model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --abstain_on_low_conf \
  --output_file enhanced_results.json
```
Key additions:
- Over-retrieves up to `--retrieve_candidates` passages and re-ranks with a CrossEncoder.
- Computes per-question confidence and margin; abstains when scores fall below `--min_conf` or `--min_margin`.
- Stores confidence metadata alongside predictions in `results/enhanced_results.json`.

## Evaluation Workflows
- **Classical metrics**
  ```bash
  python -m src.evaluation --results_path results/enhanced_results.json --tag enhanced --comparison_csv results/comparison_analysis.csv --ci
  ```
  Produces per-question statistics in memory, appends aggregate rows to `results/comparison_analysis.csv`, and optionally reports Hugging Face SQuAD EM/F1 if `evaluate` is installed.

- **RAGAS metrics**
  ```bash
  python -m src.ragas_evaluation \
    results/enhanced_results.json \
    data/processed/passages.csv \
    --tag enhanced
  ```
  (The script exposes `eval_one`; invoke from Python or wrap in a thin CLI.) Results are logged under `results/ragas_summary.csv`, with optional per-example parquet dumps when supported.

## Configuration Reference
### CLI Parameters
| Script | Argument | Default | Explanation |
| --- | --- | --- | --- |
| `src.naive_rag` | `--passages_uri` | `../assignment2-rag/data/processed/passages.csv` | Source passages; change when using new corpora. |
|  | `--queries_uri` | `../assignment2-rag/data/evaluation/test_dataset.csv` | Questions + references for evaluation. |
|  | `--embed_model` | `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face embedding model used for passages and queries. |
|  | `--gen_model` | `google/flan-t5-base` | Seq2Seq model for answer generation. |
|  | `--top_k` | `3` | Number of contexts kept per question after retrieval. |
|  | `--milvus_db_path` | `rag_wikipedia_mini.db` | Local Milvus Lite SQLite file. Delete or override to reset state. |
|  | `--milvus_index_type` | `HNSW` | ANN index type (`HNSW` vs `IVF_FLAT`). |
| `src.enhanced_rag` | `--retrieve_candidates` | `50` | ANN hits to pull before re-ranking. |
|  | `--rerank_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CrossEncoder used to score question-context pairs. |
|  | `--min_conf` / `--min_margin` | `0.25` / `0.05` | Confidence gating thresholds feeding the abstention logic. |
|  | `--abstain_on_low_conf` | `False` | When set, predictions fall back to "I don't know." if confidence is low. |
|  | `--output_file` | `enhanced_results.json` | File (under `results/`) that captures predictions and metadata. |
| `src.evaluation` | `--ci` / `--ci_alpha` | `False` / `0.05` | Enable normal-approximation confidence intervals and choose alpha. |

### Environment Variables
| Variable | Used by | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | `src/ragas_evaluation.py` | Required by `langchain_openai.ChatOpenAI` when scoring with GPT-based judges. |
| `RAGAS_CTX_CAP` | `src/ragas_evaluation.py` | Caps the number of contexts included per example (default `5`). |
| `RAGAS_CTX_MAX_CHARS` | same | Truncates each context to the specified character length (default `0` for no cap). |
| `RAGAS_SAMPLE_N` | same | Optionally limits dataset size for faster experiments. |
| `RAGAS_BATCH` / `RAGAS_WORKERS` | same | Control shard size and parallel workers for threaded evaluation. |
| `RAGAS_LLM_TIMEOUT` / `RAGAS_LLM_RETRIES` | same | Configure request timeout and retry budget for the OpenAI judge. |

## Inline Code Comments & Documentation
- Complex logic is documented inline in `src/utils.py` (Milvus index creation and search) and `src/enhanced_rag.py` (re-ranking and confidence scoring). These comments explain the rationale for architectural choices such as ANN index parameters, CrossEncoder scoring, and abstention behavior.
- Each pipeline script starts with a module-level docstring summarizing the high-level flow, and helper functions carry docstrings where behavior is non-trivial.

## Error Handling and Logging
- `src/utils.get_logger` standardizes console logging with timestamps; every CLI script instantiates a module-specific logger (`rag_utils`, `enhanced-rag`, `eval`).
- Calls to external systems (Milvus stats, RAGAS shards) are wrapped in `try/except` blocks that log warnings rather than terminating the run.
- Confidence gating in `src/enhanced_rag` prevents low-quality answers by abstaining when metrics fall below thresholds.
- The evaluation stack validates JSON schemas, normalizes gold answers defensively, and clamps confidence interval bounds to `[0, 1]`.

## Notebooks & Reproduction Tips
- `notebooks/data_exploration.ipynb` investigates corpus characteristics, embedding distributions, and query coverage.
- `notebooks/system_evaluation.ipynb` walks through running the pipelines end-to-end, visualizing metrics, and comparing system variants.
- Execute notebooks with `jupyter lab` or `jupyter notebook` after activating the virtual environment and installing requirements.

## Troubleshooting
- **Milvus schema errors**: set `--milvus_drop` or delete `rag_wikipedia_mini.db` before re-running if the schema has changed.
- **GPU memory issues**: reduce `--max_new_tokens` or switch to a smaller generator model (`google/flan-t5-small`).
- **RAGAS throughput**: lower `RAGAS_WORKERS` or reduce `RAGAS_BATCH` when rate-limited; use `RAGAS_SAMPLE_N` for quick smoke tests.
