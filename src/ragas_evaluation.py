# ragas_evaluation.py
import os, json, pandas as pd, numpy as np
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# ----------------- helpers -----------------
def load_results(path: str) -> list[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of records.")
    return data

def load_passage_lookup(passages_csv: str) -> dict[int, str]:
    df = pd.read_csv(passages_csv)[["id", "passage"]]
    return {int(r.id): str(r.passage) for r in df.itertuples(index=False)}

def _norm_gold(g) -> str:
    if isinstance(g, list):
        return str(g[0]) if g else ""
    return "" if g is None else str(g)

def _trim_ctx(c: str, max_chars: int | None) -> str:
    if not isinstance(c, str): c = str(c or "")
    if max_chars and len(c) > max_chars:
        return c[:max_chars]
    return c

# ----------------- dataset builder -----------------
def build_ragas_dataset(results_json: str, passages_csv: str) -> Dataset:
    recs = load_results(results_json)
    id2text = load_passage_lookup(passages_csv)

    ctx_cap = int(os.getenv("RAGAS_CTX_CAP", "5"))
    ctx_max = int(os.getenv("RAGAS_CTX_MAX_CHARS", "0"))

    rows = []
    for r in recs:
        q = str(r.get("question", "") or "")
        a = str(r.get("prediction", "") or "")
        g = _norm_gold(r.get("gold"))
        if not q or not a or not g:
            continue

        cids = r.get("context_ids", []) or []
        ctxs = []
        for cid in cids:
            try:
                txt = id2text.get(int(cid), "")
            except Exception:
                txt = ""
            if txt:
                ctxs.append(txt)

        # dedupe in-order
        seen = set()
        ctxs = [c for c in ctxs if not (c in seen or seen.add(c))]
        if ctx_cap > 0: ctxs = ctxs[:ctx_cap]
        if ctx_max > 0: ctxs = [_trim_ctx(c, ctx_max) for c in ctxs]
        if not ctxs:
            continue

        rows.append({
            "question": q,
            "answer": a, "response": a,
            "contexts": ctxs,
            # gold aliases for ragas versions
            "reference": g,
            "references": [g],
            "ground_truth": g,
            "ground_truths": [g],
        })

    if not rows:
        raise ValueError("No valid rows for RAGAS (empty/invalid questions, answers, golds, or contexts).")

    sample_n = os.getenv("RAGAS_SAMPLE_N")
    if sample_n:
        n = min(int(sample_n), len(rows))
        rows = rows[:n]

    return Dataset.from_pandas(pd.DataFrame(rows, dtype=object), preserve_index=False)

# ----------------- judge factory -----------------
def _make_judge(model_name: str = "gpt-4o-mini") -> ChatOpenAI:
    timeout = int(os.getenv("RAGAS_LLM_TIMEOUT", "60"))
    retries = int(os.getenv("RAGAS_LLM_RETRIES", "10"))
    return ChatOpenAI(model=model_name, temperature=0, timeout=timeout, max_retries=retries)

# ----------------- shard worker (runs in a thread) -----------------
def _eval_shard(ds: Dataset,
                start_end: Tuple[int, int],
                metrics,
                model_name: str) -> Tuple[Dict[str, float], int, pd.DataFrame | None, Tuple[int, int]]:
    start, end = start_end
    sub = ds.select(range(start, end))
    judge = _make_judge(model_name)
    res = ragas_evaluate(sub, metrics=metrics, llm=judge)
    pdf = res.to_pandas()  # per-example scores
    names = [m.name for m in metrics]
    means = {n: float(pd.to_numeric(pdf.get(n), errors="coerce").mean()) for n in names}
    return means, len(sub), pdf, (start, end)

# ----------------- evaluation (multithreaded) -----------------
def eval_one(
    results_json: str,
    passages_csv: str,
    tag: str,
    out_dir: str = "results",
    model_name: str = "gpt-4o-mini",
) -> dict:
    ds = build_ragas_dataset(results_json, passages_csv)
    metrics = [faithfulness, context_precision, context_recall, answer_relevancy]
    names = [m.name for m in metrics]

    BATCH   = int(os.getenv("RAGAS_BATCH", "25"))
    WORKERS = int(os.getenv("RAGAS_WORKERS", "4"))

    # Build shard boundaries
    total = len(ds)
    shards = [(i, min(i + BATCH, total)) for i in range(0, total, BATCH)]

    batch_means: List[Dict[str, float]] = []
    batch_sizes: List[int] = []
    per_example_frames: List[pd.DataFrame] = []

    # Run shards concurrently with bounded workers
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(_eval_shard, ds, se, metrics, model_name): se for se in shards}
        for fut in as_completed(futures):
            start, end = futures[fut]
            try:
                means, size, pdf, _ = fut.result()
                batch_means.append(means)
                batch_sizes.append(size)
                if pdf is not None:
                    per_example_frames.append(pdf)
            except Exception as e:
                print(f"[WARN] RAGAs shard {start}-{end} failed: {e}; recording NaNs.")
                batch_means.append({n: np.nan for n in names})
                batch_sizes.append(end - start)

    if not batch_means:
        raise RuntimeError("RAGAS evaluation produced no successful shards.")

    # Weighted average across shards (ignore NaNs)
    def wavg(key: str) -> float:
        vals = np.array([bm[key] for bm in batch_means], dtype=float)
        wts  = np.array(batch_sizes, dtype=float)
        mask = ~np.isnan(vals)
        if not mask.any():
            return float("nan")
        return float(np.average(vals[mask], weights=wts[mask]))

    summary = {
        "tag": tag,
        "n": int(sum(batch_sizes)),
        "faithfulness":      wavg("faithfulness"),
        "context_precision": wavg("context_precision"),
        "context_recall":    wavg("context_recall"),
        "answer_relevancy":  wavg("answer_relevancy"),
        "ctx_cap": int(os.getenv("RAGAS_CTX_CAP", "5")),
        "ctx_char_cap": int(os.getenv("RAGAS_CTX_MAX_CHARS", "0")),
        "batch": BATCH,
        "workers": WORKERS,
        "judge": model_name,
    }

    os.makedirs(out_dir, exist_ok=True)
    # append summary
    summ_csv = os.path.join(out_dir, "ragas_summary.csv")
    row_df = pd.DataFrame([summary])
    if os.path.exists(summ_csv) and os.path.getsize(summ_csv) > 0:
        pd.concat([pd.read_csv(summ_csv), row_df], ignore_index=True).to_csv(summ_csv, index=False)
    else:
        row_df.to_csv(summ_csv, index=False)

    # save per-example concatenated distribution (optional)
    if per_example_frames:
        dist_path = os.path.join(out_dir, f"ragas_dists_{tag}.parquet")
        try:
            pd.concat(per_example_frames, ignore_index=True).to_parquet(dist_path, index=False)
        except Exception:
            pass

    return summary