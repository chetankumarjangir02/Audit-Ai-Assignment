import pandas as pd
import logging
from datetime import datetime


class DocumentQASystem:
    def __init__(self, db, rag_system, qa_generator=None):
        self.db = db
        self.rag = rag_system
        self.qa_generator = qa_generator
        self.logger = logging.getLogger(__name__)

    def answer_question(self, question, document_id=None):
        try:
            
            if document_id:
                self.rag.build_index_for_document(document_id)

            labeled_data = self._get_relevant_transactions(document_id)

            # Financial logic based ONLY on this doc
            if labeled_data and self._is_financial_question(question):
                return self._answer_from_actual_data(question, labeled_data)

            # Non-financial → use RAG (per-document)
            rag_results = self.rag.retrieve(question)

            answer = self._generate_fallback_answer(question, rag_results)

            return {
                "question": question,
                "answer": answer["answer"],
                "confidence": answer["confidence"],
                "sources": answer["sources"],
                "context_used": True,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "question": question,
                "answer": f"Error: {e}",
                "confidence": 0.0,
                "sources": [],
                "timestamp": datetime.now().isoformat()
            }

   
    def _get_relevant_transactions(self, document_id):
        """Return ONLY data belong to this document. Never fallback."""
        if not document_id:
            return []

        labeled = list(self.db.labeled_data.find({
            "source_document_id": document_id
        }))

        return [self._clean(d) for d in labeled]

    def _clean(self, doc):
        safe = {}
        for k, v in doc.items():
            if k == "_id":
                continue
            safe[k] = v
        return safe

    #  Financial detection
    def _is_financial_question(self, q):
        q = q.lower()
        terms = ["total", "sum", "debit", "credit", "balance", "largest", "smallest"]
        return any(t in q for t in terms)

    #  Financial answer logic
    def _answer_from_actual_data(self, question, rows):
        df = pd.DataFrame(rows)
        q = question.lower()

        if "total debit" in q:
            total = df.get("debit", 0).astype(float).sum()
            return {
                "answer": f" Total Debit Amount: ${total:,.2f}",
                "confidence": 0.95,
                "sources": ["document_data"]
            }

        if "total credit" in q:
            total = df.get("credit", 0).astype(float).sum()
            return {
                "answer": f" Total Credit Amount: ${total:,.2f}",
                "confidence": 0.95,
                "sources": ["document_data"]
            }

        # Default summary
        return {
            "answer": " I analyzed your document, but could not match a financial query.",
            "confidence": 0.4,
            "sources": []
        }

    #  Non-financial fallback
    def _generate_fallback_answer(self, question, docs):
        if not docs:
            return {
                "answer": "⚠ No relevant information found for this document.",
                "confidence": 0.3,
                "sources": []
            }

        return {
            "answer": f"Based on this document: {docs[0]['document']}",
            "confidence": 0.6,
            "sources": ["rag"]
        }

