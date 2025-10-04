"""
Naive RAG pipeline ():
- Load HF passages & queries
- Embed passages (Sentence-Transformers)
- Build Milvus Lite collection and insert (id, passage, embedding)
- For each query: embed -> retrieve top-k -> prompt -> generate (flan-t5-base)
- Save results JSON with citations
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
from src.utils import *
from tqdm import tqdm
from typing import List, Tuple
from src.utils import DEFAULT_SYSTEM_PROMPT

logger = get_logger()

def build_index(
    passages: pd.DataFrame,
    model_name: str,
    batch_size: int = 128,
    db_path: str = "rag_wikipedia_mini.db",
    collection_name: str = "rag_mini",
    drop_if_exists: bool = False,
    index_type: str = "HNSW",
    metric_type: str = "IP",
) -> Tuple[MilvusIndex, np.ndarray]:
    """Embed passages, create Milvus Lite collection, and insert rows."""
    texts = passages["passage"].tolist()
    logger.info(f"Encoding {len(texts)} passages with {model_name} ...")
    embs = embed_texts(
        texts, model_name=model_name, batch_size=batch_size,
        normalize=True, show_progress_bar=True
    )
    dim = embs.shape[1]
    logger.info(f"Embeddings shape: {embs.shape}")

    store = MilvusIndex(
        db_path=db_path,
        collection_name=collection_name,
        dim=dim,
        drop_if_exists=drop_if_exists,
        index_type=index_type,
        metric_type=metric_type,
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


def retrieve(
    index: MilvusIndex,
    query_texts: List[str],
    model_name: str,
    k: int = 3
) -> List[List[Tuple[int, str]]]:
    """Embed queries and search Milvus Lite. Returns [(id, passage), ...] per query."""
    qvecs = embed_texts(query_texts, model_name=model_name, batch_size=64, normalize=True)
    return index.search(qvecs.astype("float32"), k=k)


def run(args):
    set_seed(args.seed)

    # Load passages
    df = load_passages(args.passages_uri)
    logger.info(f"Loaded passages: {len(df)}")

    # Build Milvus Lite index and insert embeddings
    index, _ = build_index(
        df,
        args.embed_model,
        batch_size=args.embed_batch_size,
        db_path=args.milvus_db_path,
        collection_name=args.milvus_collection,
        drop_if_exists=args.milvus_drop,
        index_type=args.milvus_index_type,
        metric_type=args.milvus_metric,
    )

    # Load queries
    qdf = load_queries(args.queries_uri).head(args.max_queries if args.max_queries > 0 else None)
    logger.info(f"Loaded queries: {len(qdf)}")

    # Retrieve contexts
    retrieved = retrieve(index, qdf["question"].tolist(), args.embed_model, k=args.top_k)

    # Build prompts
    prompts = [
        build_prompt(q, ctxs, system= getattr(args, "prompt", None) or DEFAULT_SYSTEM_PROMPT)
        for q, ctxs in zip(qdf["question"].tolist(), retrieved)
    ]

    # Generate answers
    preds = generate_answers_transformers(
        prompts,
        model_name=args.gen_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # Persist results
    out = []
    for row, ctxs, pred in tqdm(zip(qdf.itertuples(index=False), retrieved, preds), total=len(preds)):
        out.append({
            "qid": int(row.id),
            "question": row.question,
            "prediction": pred,
            "gold": row.answer,
            "context_ids": [pid for pid, _ in ctxs],
        })
    out_path = os.path.join("../results", args.output_file)
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info(f"Wrote results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--passages_uri", type=str,default="../assignment2-rag/data/processed/passages.csv", required=False)
    parser.add_argument("--queries_uri", type=str, default="../assignment2-rag/data/evaluation/test_dataset.csv", required=False)
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", required=False)
    parser.add_argument("--gen_model", type=str, default="google/flan-t5-base", required=False)
    parser.add_argument("--embed_batch_size", type=int, default=128, required=False)
    parser.add_argument("--top_k", type=int, default=3, required=False)
    parser.add_argument("--max_new_tokens", type=int, default=128, required=False)
    parser.add_argument("--temperature", type=float, default=0.0, required=False)
    parser.add_argument("--max_queries", type=int, default=50, required=False)
    parser.add_argument("--output_file", type=str, default="naive_results.json", required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)
    parser.add_argument("--milvus_db_path", type=str, default="rag_wikipedia_mini.db", required=False)
    parser.add_argument("--milvus_collection", type=str, default="rag_mini", required=False)
    parser.add_argument("--milvus_drop", action="store_true", required=False)
    parser.add_argument("--milvus_index_type", type=str, default="HNSW", choices=["HNSW", "IVF_FLAT"], required=False)
    parser.add_argument("--milvus_metric", type=str, default="IP", choices=["IP", "L2"], required=False)
    parser.add_argument("--prompt", type=str, default=DEFAULT_SYSTEM_PROMPT, required=False)
    

    args = parser.parse_args()
    run(args)