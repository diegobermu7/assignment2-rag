# src/evaluation.py
try:
    import evaluate as hf_evaluate
    _HF_SQUAD = hf_evaluate.load("squad")
except Exception:
    _HF_SQUAD = None

"""
Slim evaluator:
- load_results(list of dicts)
- per_question_scores(DataFrame with f1/unknown)  # EM intentionally omitted
- normal_ci (z from alpha) + comparison row
- HF SQuAD for macro EM (and macro F1 for reference)
- CLI prints macro means and writes <results>_scores.csv
"""

import os, json, argparse
import numpy as np
import pandas as pd
from scipy import stats
from src.utils import  get_logger

logger = get_logger("eval")

def normalize_answer(s: str) -> str:
    """Lower, remove punctuation/articles/extra whitespace."""
    import re, string
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = set(pred_tokens) & set(truth_tokens)
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)
    if len(common) == 0:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * precision * recall / (precision + recall + 1e-12)

# ---------- I/O ----------
def load_results(path: str) -> list:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of records.")
    return data

def _normalize_gold(g):
    if isinstance(g, list):
        return g[0] if g else ""
    return "" if g is None else str(g)

# ---------- Per-question metrics (F1 only) ----------
def per_question_scores(records: list) -> pd.DataFrame:
    """
    Returns per-question F1 and unknown flag.
    NOTE: We deliberately DO NOT compute EM here. EM is computed once via HF SQuAD.
    """
    rows = []
    for i, r in enumerate(records):
        pred = str(r.get("prediction") or "")
        gold = _normalize_gold(r.get("gold"))
        qid  = r.get("qid", r.get("id", i))
        p_lo = pred.lower()
        unknown = int(("don't know" in p_lo) or ("do not know" in p_lo) or ("unknown" in p_lo))
        rows.append({
            "qid": qid,
            "f1": f1_score(pred, gold),
            "unknown": unknown,
        })
    return pd.DataFrame(rows)

# ---------- Normal-approx CI: m ± z*s/√n (z from alpha) ----------
def normal_ci(values: np.ndarray, alpha: float = 0.05):
    """
    Mean ± z * s / sqrt(n), where z = Φ^{-1}(1 - alpha/2).
    Bounds clamped to [0,1].
    Returns (mean, lo, hi).
    """
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    m = float(np.nanmean(vals)) if n else float("nan")
    if n <= 1:
        return m, m, m
    s = float(np.nanstd(vals, ddof=1))
    z = float(stats.norm.ppf(1.0 - alpha/2.0))
    half = z * s / (n ** 0.5)
    lo, hi = max(0.0, m - half), min(1.0, m + half)
    return m, lo, hi

# ---------- HF SQuAD macro metrics (EM/F1 in percent) ----------
def hf_squad_scores(results_path: str) -> dict | None:
    """Return {'exact_match': %, 'f1': %} or None if HF evaluate isn't available."""
    if _HF_SQUAD is None:
        logger.warning("HuggingFace 'evaluate' not available; skipping HF SQuAD metrics.")
        return None
    data = load_results(results_path)
    preds = [{"id": str(r.get("qid", r.get("id"))), "prediction_text": str(r.get("prediction",""))} for r in data]
    refs  = [{"id": str(r.get("qid", r.get("id"))),
              "answers": {"text": [_normalize_gold(r.get("gold"))], "answer_start": [0]}}
             for r in data]
    return _HF_SQUAD.compute(predictions=preds, references=refs)

# ---------- Comparison row ----------
def append_comparison_row(scores: pd.DataFrame, tag: str, out_csv: str,
                          cis: dict | None, hf: dict | None = None):
    """
    Stores macro F1 (and its CI if provided), unknown rate, and HF EM/F1 (scaled to 0..1).
    EM mean in this table is taken from HF only (no per-question EM here).
    """
    row = {
        "tag": tag,
        "n": int(len(scores)),
        "f1_mean": float(scores.f1.mean()),
        "unknown_rate": float(scores.unknown.mean()),
    }
    if cis and "f1" in cis:
        row.update({"f1_lo": cis["f1"][1], "f1_hi": cis["f1"][2]})

    if hf:
        # HF returns 0..100; convert to 0..1 for consistency
        row.update({
            "em_mean": float(hf.get("exact_match", 0.0)) / 100.0,
            "hf_f1":   float(hf.get("f1", 0.0)) / 100.0,
        })
    else:
        row.update({"em_mean": np.nan, "hf_f1": np.nan})

    out = pd.DataFrame([row])
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        pd.concat([pd.read_csv(out_csv), out], ignore_index=True).to_csv(out_csv, index=False)
    else:
        out.to_csv(out_csv, index=False)
    logger.info(f"Appended to {out_csv}: {row}")

# ---------- One-shot pipeline for notebooks ----------
def evaluate_once(results_path: str, 
                  tag: str = "run",
                  comparison_csv: str = "../results/comparison_analysis.csv",
                  ci: bool = True,
                  ci_alpha: float = 0.05) -> pd.DataFrame:
    records = load_results(results_path)
    scores  = per_question_scores(records)

    # CI only for F1 (we have per-question F1s; EM is HF-only macro)
    cis = None
    if ci:
        f1_ci = normal_ci(scores.f1.values.astype(float), alpha=ci_alpha)
        cis   = {"f1": f1_ci}
        logger.info(f"N={len(scores)} | F1={f1_ci[0]:.3f}[{f1_ci[1]:.3f},{f1_ci[2]:.3f}] "
                    f"| unknown_rate={scores.unknown.mean():.3f}")
    else:
        logger.info(f"N={len(scores)} | F1={scores.f1.mean():.3f} | unknown_rate={scores.unknown.mean():.3f}")

    # HF SQuAD (macro EM and macro F1, in percent)
    hf = hf_squad_scores(results_path)
    if hf:
        logger.info(f"HF SQuAD | EM={hf['exact_match']:.2f}% | F1={hf['f1']:.2f}%")

    # per-question CSV (no EM column by design)
    #out_csv = results_path.replace(".json", "_scores.csv")
    #os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    #scores.to_csv(out_csv, index=False)
    #logger.info(f"Wrote per-question scores: {out_csv}")

    # comparison row with HF EM and F1 CI
    append_comparison_row(scores, tag, comparison_csv, cis, hf)
    return scores

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_path", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--scores_out", default=None)  # kept for compat; we always write <results>_scores.csv
    ap.add_argument("--comparison_csv", default="results/comparison_analysis.csv")
    ap.add_argument("--ci", action="store_true")
    ap.add_argument("--ci_alpha", type=float, default=0.05)
    args = ap.parse_args()

    # reuse the one-shot
    evaluate_once(
        results_path=args.results_path,
        tag=args.tag,
        comparison_csv=args.comparison_csv,
        ci=args.ci,
        ci_alpha=args.ci_alpha,
    )

if __name__ == "__main__":
    main()