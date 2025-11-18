import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime

class RAGSystem:
    def __init__(self, db):
        self.db = db
        self.model = None
        self.index = None
        self.documents = []
        self._initialize_model()
        self._build_index()

    def _initialize_model(self):
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            self.model = None

    def _build_index(self):
        try:
            labeled = self.db.get_recent_labeled_data(1000)
            docs = []
            for it in labeled:
                text = self._document_to_text(it)
                if text:
                    docs.append(text)
            if not docs:
                self._create_empty_index()
                return
            self.documents = docs
            if self.model:
                embeddings = self.model.encode(docs, show_progress_bar=False)
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dimension)
                self.index.add(np.array(embeddings).astype('float32'))
        except Exception:
            self._create_empty_index()

    def _document_to_text(self, document):
        try:
            if isinstance(document, dict):
                parts = []
                fields = ['description', 'category', 'date', 'debit', 'credit', 'balance']
                for f in fields:
                    if f in document and document[f] not in (None, ""):
                        parts.append(f"{f} {document[f]}")
                return " | ".join(parts) if parts else None
            return str(document)
        except Exception:
            return None

    def _create_empty_index(self):
        if self.model:
            dim = self.model.get_sentence_embedding_dimension()
        else:
            dim = 384
        self.index = faiss.IndexFlatL2(dim)
        self.documents = []

    def retrieve(self, query, k=5):
        try:
            if not query or self.index is None or self.index.ntotal == 0 or not self.model:
                return []
            emb = self.model.encode([query])
            D, I = self.index.search(np.array(emb).astype('float32'), min(k, self.index.ntotal))
            results = []
            for dist, idx in zip(D[0], I[0]):
                if idx < len(self.documents):
                    score = float(1 / (1 + float(dist)))
                    results.append({"document": self.documents[idx], "score": score, "index": int(idx)})
            results.sort(key=lambda x: x["score"], reverse=True)
            return results
        except Exception:
            return []

    def add_documents(self, docs):
        # docs is list of dicts
        new_texts = []
        for d in docs:
            t = self._document_to_text(d)
            if t:
                new_texts.append(t)
        if not new_texts:
            return False
        self.documents.extend(new_texts)
        if self.model:
            embeddings = self.model.encode(self.documents, show_progress_bar=False)
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(np.array(embeddings).astype('float32'))
            return True
        return False

    def get_index_stats(self):
        return {
            "documents_count": len(self.documents),
            "index_ready": bool(self.index and getattr(self.index, "ntotal", 0) > 0),
            "model_loaded": self.model is not None,
            "timestamp": datetime.utcnow().isoformat()
        }
