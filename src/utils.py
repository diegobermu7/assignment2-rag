import os
import math
import json
import random
import logging
import tiktoken
import numpy as np
import pandas as pd
from pymilvus import MilvusClient
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Iterable, Optional, Tuple
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection



# ----------------------------------
# Reproducibility & Helper Functions
# ----------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

LOGGER_NAME = "rag_utils"

def get_logger(name: str = LOGGER_NAME, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

logger = get_logger()



def approx_token_count(text: str) -> int:
    """Approximate tokens; use tiktoken if available else chars/4 heuristic."""
    if tiktoken is not None:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


# --------------------------
# Dataset loading & cleaning
# --------------------------
HF_PASSAGES_PATH = "../assignment2-rag/data/processed/passages.csv"
HF_QUERIES_PATH  = "../assignment2-rag/data/evaluation/test_dataset.csv"

def normalize_text(s: str) -> str:
    return (
        s.lower()
         .replace("\n", " ")
         .replace("\t", " ")
    )

def load_passages(file_path: str = HF_PASSAGES_PATH) -> pd.DataFrame:
    """Load passages parquet from Hugging Face URI."""
    df = pd.read_csv(file_path)
    df["passage_norm"] = df["passage"].apply(normalize_text)
    df["char_len"] = df["passage"].str.len()
    df["tok_est"] = df["passage"].apply(approx_token_count)
    df["id"] = df.index.astype(int)
    return df[["id","passage","passage_norm","char_len","tok_est"]]


def load_queries(file_path: str = HF_QUERIES_PATH) -> pd.DataFrame:
    """Expect columns like: 'question', 'answer' (string or list)."""
    qdf = pd.read_csv(file_path)
    return qdf[["id","question","answer"]]

# ------------------------------------
# Embeddings (Sentence Transformers)
# ------------------------------------
def embed_texts(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 128,
    normalize: bool = True,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """Encode texts with sentence-transformers; returns np.ndarray [N, D]."""
    model = SentenceTransformer(model_name)
    embs = model.encode(
        texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=normalize, show_progress_bar=show_progress_bar
    )
    return embs

# --------------------------
# Milvus index (primary)
# --------------------------
class MilvusIndex:
 # ---- Simple Milvus (Lite) index using MilvusClient ----
# Drop this into src/utils.py (replacing the previous MilvusIndex if present)
    """
    Vector store wrapper backed by Milvus Lite.
    Stores (id, passage, embedding) with a single HNSW index (IP / cosine).
    """


    def __init__(
        self,
        db_path: str = "rag_wikipedia_mini.db",
        collection_name: str = "rag_mini",
        dim: int = 384,
        drop_if_exists: bool = False,
        varchar_max_length: int = 4096,
        index_type: str = "HNSW",
        metric_type: str = "IP",
    ):
        
        self.client = MilvusClient(db_path)     # Milvus Lite (2.6.x)
        self.dim = dim
        self.collection_name = collection_name
        self.metric_type = metric_type
        self.index_type = index_type

        if drop_if_exists and self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)

        if not self.client.has_collection(collection_name):
            f_id = FieldSchema(
                name="id", dtype=DataType.INT64, is_primary=True, auto_id=False
            )
            f_text = FieldSchema(
                name="passage", dtype=DataType.VARCHAR, max_length=varchar_max_length
            )
            f_vec = FieldSchema(
                name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim
            )

            schema = CollectionSchema(
                fields=[f_id, f_text, f_vec],
                description="RAG passages (id, passage, embedding)",
                enable_dynamic_field=True,  # safe default; okay to omit if not used
            )

            # 2) ANN index params
            index_params = self.client.prepare_index_params()
            if index_type.upper() == "HNSW":
                index_params.add_index(
                    field_name="embedding",
                    index_type="HNSW",
                    metric_type=metric_type,            # "IP" or "L2"
                    params={"M": 16, "efConstruction": 200},
                )
            else:
                index_params.add_index(
                    field_name="embedding",
                    index_type="IVF_FLAT",
                    metric_type=metric_type,
                    params={"nlist": 1024},
                )

            # 3) Create collection (pass ORM schema to avoid fast-path)
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
            )

    def add(self, ids: Iterable[int], passages: Iterable[str], embeddings: np.ndarray) -> None:
        # Convert to lists first
        id_list = list(map(int, ids))
        passage_list = list(map(str, passages))
        embedding_list = embeddings.astype("float32").tolist()
        
        # Create list of dictionaries (row-based format)
        data = []
        for i in range(len(id_list)):
            data.append({
                "id": id_list[i],
                "passage": passage_list[i], 
                "embedding": embedding_list[i]
            })
        
        self.client.insert(
            collection_name=self.collection_name,
            data=data
        )

    def search(self, qvecs: np.ndarray, k: int = 3) -> List[List[Tuple[int, str]]]:
        results = self.client.search(
            collection_name=self.collection_name,
            data=np.asarray(qvecs, dtype="float32"),
            limit=k,
            search_params={
                "metric_type": self.metric_type,
                "params": {"ef": 128} if self.index_type.upper() == "HNSW" else {"nprobe": 16},
            },
            output_fields=["id", "passage"],
        )
        out: List[List[Tuple[int, str]]] = []
        for hits in results:
            ctxs: List[Tuple[int, str]] = []
            for h in hits:
                ent = h.get("entity") or {}
                pid = int(ent.get("id", h.get("id")))
                text = ent.get("passage", "")
                ctxs.append((pid, text))
            out.append(ctxs)
        return out


# ----------------------------------------------------
# Simple prompt templating -- RAG's first layer
# ----------------------------------------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer using only the provided context. "
    "If the answer is not in the context, say you don't know. Be concise. "
    "Cite the passage IDs you used."
)

def build_prompt(question: str, contexts: List[Tuple[int, str]], system: str = DEFAULT_SYSTEM_PROMPT) -> str:
    ctx_block = "\n\n".join([f"[{pid}] {text}" for pid, text in contexts])
    return f"{system}\n\nContext:\n{ctx_block}\n\nQuestion: {question}\nAnswer:"

# ----------------------------------------------------
# Generation with Transformers -- RAG's second layer
# ----------------------------------------------------
def generate_answers_transformers(
    prompts: List[str],
    model_name: str = "google/flan-t5-base",
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> List[str]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    outputs = []
    for p in prompts:
        inpt = tok(p, return_tensors="pt", truncation=True, padding=True).to(device)
        gen = model.generate(**inpt, max_new_tokens=max_new_tokens, do_sample=temperature>0.0, temperature=temperature)
        txt = tok.decode(gen[0], skip_special_tokens=True).strip()
        outputs.append(txt)
    return outputs



