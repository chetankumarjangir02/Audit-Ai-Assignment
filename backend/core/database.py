from pymongo import MongoClient
from datetime import datetime
import os

class MongoDBManager:
    def __init__(self):
        self.client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017/"))
        self.db = self.client[os.getenv("DATABASE_NAME", "audit_ai")]

        # Collections
        self.documents = self.db.documents
        self.labeled_data = self.db.labeled_data
        self.qa_pairs = self.db.qa_pairs
        self.fine_tuning_logs = self.db.fine_tuning_logs
        self.agent_logs = self.db.agent_logs
        self.qa_interactions = self.db.qa_interactions

    def _keep_latest(self, collection):
        last = collection.find_one(sort=[("_id", -1)])
        if last:
            collection.delete_many({"_id": {"$ne": last["_id"]}})

    def log_agent_action(self, agent_name, action, input_data, output_data, confidence=1.0):
        try:
            entry = {
                "agent": agent_name,
                "action": action,
                "input": input_data,
                "output": output_data,
                "confidence": confidence,
                "timestamp": datetime.utcnow(),
                "version": "1.0"
            }
            return self.agent_logs.insert_one(entry)
        except Exception:
            return None

    def store_document(self, file_path, content, doc_type):
        try:
            # keep only latest document (option B1)
            self.documents.delete_many({})
            doc = {"file_path": file_path, "content": content, "type": doc_type, "uploaded_at": datetime.utcnow(), "processed": False}
            return self.documents.insert_one(doc)
        except Exception:
            return None

    def store_document_with_id(self, file_path, content, doc_type, document_id):
        try:
            self.documents.delete_many({})
            doc = {"document_id": document_id, "file_path": file_path, "content": content, "type": doc_type, "uploaded_at": datetime.utcnow(), "processed": False}
            return self.documents.insert_one(doc)
        except Exception:
            return None

    def store_labeled_data(self, labeled_data, source_doc_id):
        try:
            # keep latest only
            self.labeled_data.delete_many({})
            for item in labeled_data:
                item["source_document_id"] = source_doc_id
                item["created_at"] = datetime.utcnow()
            if labeled_data:
                return self.labeled_data.insert_many(labeled_data)
            return None
        except Exception:
            return None

    def store_qa_pairs(self, qa_pairs, source_doc_id):
        try:
            if not qa_pairs:
                return None
            self.qa_pairs.delete_many({})
            qa_data = []
            for qa in qa_pairs:
                q = qa.copy()
                q["source_document_id"] = source_doc_id
                q["created_at"] = datetime.utcnow()
                qa_data.append(q)
            if qa_data:
                return self.qa_pairs.insert_many(qa_data)
            return None
        except Exception:
            return None

    def get_transactions_by_document_id(self, document_id):
        try:
            return list(self.labeled_data.find({"source_document_id": document_id}))
        except Exception:
            return []

    def get_recent_labeled_data(self, limit=1000):
        try:
            return list(self.labeled_data.find().sort("created_at", -1).limit(limit))
        except Exception:
            return []

    def store_qa_interaction(self, interaction_data):
        try:
            return self.qa_interactions.insert_one(interaction_data)
        except Exception:
            return None

    def get_qa_history(self, limit=10):
        try:
            history = list(self.qa_interactions.find().sort("timestamp", -1).limit(limit))
            for h in history:
                if '_id' in h:
                    h['_id'] = str(h['_id'])
            return history
        except Exception:
            return []
