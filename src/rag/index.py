import json
import os
from dataclasses import dataclass

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.utils.io import load_jsonl, write_jsonl


def _tokenize(text):
    return text.lower().split()


@dataclass
class RAGIndex:
    docs: list
    bm25: BM25Okapi
    faiss_index: faiss.Index
    normalize: bool

    def save(self, index_dir):
        os.makedirs(index_dir, exist_ok=True)
        write_jsonl(os.path.join(index_dir, "docs.jsonl"), self.docs)
        faiss.write_index(self.faiss_index, os.path.join(index_dir, "dense.index"))
        with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as handle:
            json.dump({"normalize": self.normalize}, handle, indent=2)

    @classmethod
    def load(cls, index_dir):
        docs = load_jsonl(os.path.join(index_dir, "docs.jsonl"))
        faiss_index = faiss.read_index(os.path.join(index_dir, "dense.index"))
        meta_path = os.path.join(index_dir, "meta.json")
        normalize = False
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as handle:
                normalize = json.load(handle).get("normalize", False)
        bm25 = BM25Okapi([_tokenize(doc["text"]) for doc in docs])
        return cls(docs=docs, bm25=bm25, faiss_index=faiss_index, normalize=normalize)


def build_index(corpus_path, embedding_model_path, index_dir, normalize=True):
    docs = load_jsonl(corpus_path)
    model = SentenceTransformer(embedding_model_path)
    texts = [doc["text"] for doc in docs]
    embeddings = model.encode(texts, batch_size=8, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    if normalize:
        faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    rag_index = RAGIndex(docs=docs, bm25=BM25Okapi([_tokenize(t) for t in texts]), faiss_index=index, normalize=normalize)
    rag_index.save(index_dir)
    return rag_index
