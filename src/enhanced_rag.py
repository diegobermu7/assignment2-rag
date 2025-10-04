# src/enhanced_rag.py
"""
Enhanced RAG:
- Retrieve N candidates from Milvus Lite
- Re-rank candidates with a CrossEncoder (robust)
- Confidence scoring with optional abstention (simple)
- Prompt & generate with HF seq2seq; write JSON with citations + confidence
"""
import os, json, argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import CrossEncoder

from src.utils import (
    set_seed, get_logger,
    load_passages, load_queries, embed_texts,
    MilvusIndex, build_prompt, generate_answers_transformers,
    DEFAULT_SYSTEM_PROMPT,
)

logger = get_logger("enhanced-rag")

# ---------- Index ----------
def build_index(
    passages: pd.DataFrame,
    model_name: str,
    batch_size: int = 128,
    db_path: str = "rag_wikipedia_mini.db",
    collection_name: str = "rag_mini",
    drop_if_exists: bool = False,
    index_type: str = "IVF_FLAT",   # default IVF_FLAT for Milvus Lite local mode
    metric_type: str = "IP",
) -> Tuple[MilvusIndex, np.ndarray]:
    texts = passages["passage"].tolist()
    logger.info(f"Encoding {len(texts)} passages with {model_name} ...")
    embs = embed_texts(texts, model_name=model_name, batch_size=batch_size,
                       normalize=True, show_progress_bar=True)
    dim = embs.shape[1]
    logger.info(f"Embeddings shape: {embs.shape}")

    store = MilvusIndex(
        db_path=db_path, collection_name=collection_name, dim=dim,
        drop_if_exists=drop_if_exists, index_type=index_type, metric_type=metric_type
    )
    store.add(passages["id"].tolist(), passages["passage"].tolist(), embs.astype("float32"))

    try:
        stats = store.client.get_collection_stats(collection_name)
        schema = store.client.describe_collection(collection_name)
        print("Entity count:", stats["row_count"])
        print("Collection schema:", schema)
    except Exception as e:
        logger.warning(f"Could not fetch Milvus stats: {e}")
    return store, embs

# ---------- Retrieval + Re-ranking ----------
def retrieve_candidates(
    index: MilvusIndex,
    query_texts: List[str],
    model_name: str,
    n_candidates: int = 50,
) -> List[List[Tuple[int, str]]]:
    qvecs = embed_texts(query_texts, model_name=model_name, batch_size=64, normalize=True)
    return index.search(qvecs.astype("float32"), k=n_candidates)

def rerank_with_cross_encoder(
    query: str, candidates: List[Tuple[int, str]], model: CrossEncoder
) -> List[Tuple[int, str, float]]:
    if not candidates:
        return []
    pairs = [(query, text) for _, text in candidates]
    scores = model.predict(pairs)  # higher is more relevant
    scored = [(pid, text, float(s)) for (pid, text), s in zip(candidates, scores)]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored

# ---------- Confidence ----------
def confidence_from_scores(scored: List[Tuple[int, str, float]], top_k: int) -> dict:
    if not scored:
        return {"conf": 0.0, "margin": 0.0}
    top = scored[0][2]
    second = scored[1][2] if len(scored) > 1 else -1e9
    margin = top - second
    return {"conf": float(top), "margin": float(margin)}

# ---------- Main pipeline ----------
def run(args):
    set_seed(args.seed)

    # Load data
    df = load_passages(args.passages_uri)
    qdf = load_queries(args.queries_uri).head(args.max_queries if args.max_queries > 0 else None)
    logger.info(f"Loaded passages={len(df)} queries={len(qdf)}")

    # Build index
    index, _ = build_index(
        df, args.embed_model, batch_size=args.embed_batch_size,
        db_path=args.milvus_db_path, collection_name=args.milvus_collection,
        drop_if_exists=args.milvus_drop, index_type=args.milvus_index_type,
        metric_type=args.milvus_metric,
    )

    # Cross-encoder for re-ranking
    logger.info(f"Loading CrossEncoder: {args.rerank_model}")
    reranker = CrossEncoder(args.rerank_model, max_length=args.rerank_max_length)

    # Retrieve + re-rank + generate
    out = []
    all_questions = qdf["question"].tolist()
    all_gold      = qdf["answer"].tolist()
    all_ids       = qdf["id"].tolist()

    # bulk candidate retrieval for efficiency
    candidates_per_q = retrieve_candidates(index, all_questions, args.embed_model, args.retrieve_candidates)

    for qid, question, gold, cands in tqdm(zip(all_ids, all_questions, all_gold, candidates_per_q), total=len(all_ids)):
        scored = rerank_with_cross_encoder(question, cands, reranker)
        conf = confidence_from_scores(scored, args.top_k)

        # pick top-k passages after re-ranking
        top = [(pid, text) for pid, text, _ in scored[:args.top_k]]

        # abstain if requested and confidence is low
        low_conf = (conf["conf"] < args.min_conf) or (conf["margin"] < args.min_margin)
        if args.abstain_on_low_conf and low_conf:
            pred = "I don't know."
        else:
            prompt = build_prompt(question, top, system=(args.prompt or DEFAULT_SYSTEM_PROMPT))
            pred = generate_answers_transformers(
                [prompt], model_name=args.gen_model, max_new_tokens=args.max_new_tokens,
                temperature=args.temperature
            )[0]

        out.append({
            "qid": int(qid),
            "question": question,
            "prediction": pred,
            "gold": gold,
            "context_ids": [pid for pid, _ in top],
            "conf": conf["conf"],
            "margin": conf["margin"],
            "low_conf": bool(low_conf),
        })

    # Persist
    out_path = os.path.join("../results", args.output_file)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info(f"Wrote enhanced results to {out_path}")

# ---------- CLI ----------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--passages_uri", type=str, default="../assignment2-rag/data/processed/passages.csv")
    p.add_argument("--queries_uri",  type=str, default="../assignment2-rag/data/evaluation/test_dataset.csv")
    # embeddings/gen
    p.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--gen_model",   type=str, default="google/flan-t5-base")
    p.add_argument("--embed_batch_size", type=int, default=128)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature",    type=float, default=0.0)
    p.add_argument("--prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    # retrieval
    p.add_argument("--retrieve_candidates", type=int, default=50, help="ANN candidates before re-ranking")
    p.add_argument("--top_k", type=int, default=5, help="contexts to keep after re-ranking")
    # reranker
    p.add_argument("--rerank_model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument("--rerank_max_length", type=int, default=512)
    # confidence
    p.add_argument("--min_conf", type=float, default=0.25)
    p.add_argument("--min_margin", type=float, default=0.05)
    p.add_argument("--abstain_on_low_conf", action="store_true")
    # milvus
    p.add_argument("--milvus_db_path", type=str, default="rag_wikipedia_mini.db")
    p.add_argument("--milvus_collection", type=str, default="rag_mini")
    p.add_argument("--milvus_drop", action="store_true")
    p.add_argument("--milvus_index_type", type=str, default="IVF_FLAT", choices=["IVF_FLAT"])
    p.add_argument("--milvus_metric", type=str, default="IP", choices=["IP", "L2"])
    # run control
    p.add_argument("--max_queries", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_file", type=str, default="enhanced_results.json")
    args = p.parse_args()
    run(args)