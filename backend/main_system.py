import os
from dotenv import load_dotenv
from backend.agents.planner_agent import PlannerAgent
from backend.agents.executor_agent import ExecutorAgent
from backend.agents.reviewer_agent import ReviewerAgent
from backend.agents.labeling_agent import LabelingAgent
from backend.core.database import MongoDBManager
from backend.core.rag import RAGSystem
from backend.core.fine_tuning import FineTuningManager
from backend.core.qa_generator import QAGenerator
from backend.core.qa_system import DocumentQASystem
from backend.orchestrator.graph import build_workflow

load_dotenv()


class AuditIntelligenceSystem:
    def __init__(self):
        print("Initializing Audit Intelligence System...")

        # Database
        self.db = MongoDBManager()
        print("Database initialized")

        # RAG
        self.rag = RAGSystem(self.db)
        print("RAG initialized")

        # Fine-tuning manager
        self.fine_tuning_manager = FineTuningManager(self.db)
        print("Fine-tuning manager ready")

        # QA generator
        self.qa_generator = QAGenerator(self.db)
        print("QA generator ready")

        # QA System
        self.qa_system = DocumentQASystem(self.db, self.rag, self.qa_generator)
        print(" QA system ready")

        # Agents
        self.planner = PlannerAgent(self.db)
        self.executor = ExecutorAgent(self.db, self.rag)
        self.reviewer = ReviewerAgent(self.db)
        self.labeler = LabelingAgent(self.db)
        print("Agents initialized")

        # Orchestrator / workflow graph
        self.workflow = build_workflow(
            planner=self.planner,
            executor=self.executor,
            reviewer=self.reviewer,
            labeler=self.labeler,
            rag=self.rag,
            qa_generator=self.qa_generator,
            fine_tuner=self.fine_tuning_manager
        )
        print(" Workflow graph built")

    def process_request(self, user_input, file_path=None, document_id=None):
        # Build initial state dict
        state = {
            "user_input": user_input,
            "file_path": file_path,
            "document_id": document_id,
            "confidence": 1.0
        }

        # Run graph or fallback synchronous pipeline
        try:
            if hasattr(self.workflow, "invoke"):
                result = self.workflow.invoke(state)
            else:
                # fallback manual pipeline
                result = self.planner.process(state)
                result = self.executor.process(result)
                result = self.reviewer.process(result)
                if result.get("needs_labeling") or ("label" in user_input.lower() or "process" in user_input.lower()):
                    result = self.labeler.process(result)

            # After processing, if we have a document id, attempt to add transactions to RAG
            doc_id = result.get("document_id") or document_id
            try:
                if doc_id:
                    # get_transactions_by_document_id should return list of transactions
                    if hasattr(self.db, "get_transactions_by_document_id"):
                        txns = self.db.get_transactions_by_document_id(doc_id) or []
                        if txns:
                            self.rag.add_documents(txns)
            except Exception:
                pass

            return result
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def answer_question(self, question, document_id=None):
        return self.qa_system.answer_question(question, document_id)

    def get_system_status(self):
        try:
            return {
                "database_connected": self.db is not None,
                "rag_ready": self.rag is not None,
                "documents_count": self.db.documents.count_documents({}) if hasattr(self.db, "documents") else 0,
                "labeled_data_count": self.db.labeled_data.count_documents({}) if hasattr(self.db, "labeled_data") else 0,
                "qa_pairs_count": self.db.qa_pairs.count_documents({}) if hasattr(self.db, "qa_pairs") else 0
            }
        except Exception as e:
            return {"error": str(e)}

audit_system = AuditIntelligenceSystem()

