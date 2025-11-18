
from datetime import datetime
import json
from typing import Dict, Any, List
class PlannerAgent:
    def __init__(self, db):
        self.db = db
        self.name = "PlannerAgent"

    def process(self, context):
        """Break down user queries into sequential subtasks"""
        user_input = context["user_input"].lower()
        file_path = context.get("file_path")

        # Analyze user intent and create subtasks
        if "label" in user_input or "process" in user_input or file_path:
            subtasks = [
                {"task": "parse_document", "description": "Parse uploaded document"},
                {"task": "label_transactions", "description": "Label all transactions"},
                {"task": "generate_summary", "description": "Generate financial summary"},
                {"task": "store_results", "description": "Store results in database"}
            ]
        elif "analyze" in user_input or "pattern" in user_input:
            subtasks = [
                {"task": "retrieve_data", "description": "Retrieve relevant data"},
                {"task": "analyze_patterns", "description": "Analyze financial patterns"},
                {"task": "generate_insights", "description": "Generate insights and charts"}
            ]
        elif "?" in user_input or any(word in user_input for word in ["what", "how", "show", "tell"]):
            subtasks = [
                {"task": "rag_retrieval", "description": "Retrieve relevant information"},
                {"task": "generate_answer", "description": "Generate answer using RAG"}
            ]
        else:
            subtasks = [
                {"task": "general_processing", "description": "Process general request"}
            ]

        plan = {
            "subtasks": subtasks,
            "timestamp": datetime.utcnow().isoformat(),
            "user_input": context["user_input"]
        }

        # Log the planning action
        self.db.log_agent_action(
            self.name,
            "plan_creation",
            {"user_input": context["user_input"], "file_path": file_path},
            plan
        )

        context["plan"] = plan
        context["needs_labeling"] = file_path and any("label" in task["task"] for task in subtasks)

        return context
