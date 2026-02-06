from sentence_transformers import SentenceTransformer

from src.rag.index import RAGIndex, _tokenize


def _min_max_norm(values):
    if not values:
        return values
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        return [0.0 for _ in values]
    return [(v - min_v) / (max_v - min_v) for v in values]


class HybridRetriever:
    def __init__(self, index_dir, embedding_model_path, alpha=0.55, top_k=5, reranker_model_path=None, rerank_top_k=10):
        self.index = RAGIndex.load(index_dir)
        self.alpha = alpha
        self.top_k = top_k
        self.embedder = SentenceTransformer(embedding_model_path)
        self.reranker = None
        self.rerank_top_k = rerank_top_k
        if reranker_model_path:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(reranker_model_path)

    def retrieve(self, query, top_k=None):
        if top_k is None:
            top_k = self.top_k
        bm25_scores = self.index.bm25.get_scores(_tokenize(query))
        bm25_norm = _min_max_norm(list(bm25_scores))

        query_emb = self.embedder.encode([query], show_progress_bar=False).astype("float32")
        if self.index.normalize:
            import faiss
            faiss.normalize_L2(query_emb)
        dense_scores, dense_indices = self.index.faiss_index.search(query_emb, min(top_k * 3, len(self.index.docs)))
        dense_scores = dense_scores[0].tolist()
        dense_indices = dense_indices[0].tolist()

        combined = {}
        for idx, score in enumerate(bm25_norm):
            combined[idx] = self.alpha * score
        dense_norm = _min_max_norm(dense_scores)
        for rank, doc_idx in enumerate(dense_indices):
            if doc_idx < 0:
                continue
            combined[doc_idx] = combined.get(doc_idx, 0.0) + (1.0 - self.alpha) * dense_norm[rank]

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_idx, score in ranked[:top_k]:
            doc = self.index.docs[doc_idx]
            results.append({"score": float(score), "doc": doc})

        if self.reranker:
            rerank_candidates = results[: min(self.rerank_top_k, len(results))]
            pairs = [[query, item["doc"]["text"]] for item in rerank_candidates]
            scores = self.reranker.predict(pairs).tolist()
            rescored = [
                {"score": float(score), "doc": item["doc"]}
                for item, score in zip(rerank_candidates, scores)
            ]
            rescored.sort(key=lambda x: x["score"], reverse=True)
            results = rescored[:top_k]
        return results


def build_citations(results, max_chars=260):
    citations = []
    for item in results:
        doc = item["doc"]
        snippet = doc["text"][:max_chars]
        citations.append({"doc_id": doc["doc_id"], "snippet": snippet})
    return citations
