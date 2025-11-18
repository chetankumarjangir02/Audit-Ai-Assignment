import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import json
import re
import os
from typing import Dict, Any, List



class ExecutorAgent:
    def __init__(self, db, rag):
        self.db = db
        self.rag = rag
        self.name = "ExecutorAgent"

    def process(self, context):
        """Execute the planned subtasks"""
        plan = context["plan"]
        user_input = context["user_input"]
        file_path = context.get("file_path")

        results = {}

        for subtask in plan["subtasks"]:
            task_name = subtask["task"]

            if task_name == "parse_document" and file_path:
                from utils.file_parser import FileParser
                parser = FileParser()
                results["parsed_data"] = parser.parse_file(file_path)
            elif task_name == "generate_summary":
                results["summary"] = self.generate_summary(results.get("parsed_data", {}))
            elif task_name == "analyze_patterns":
                results["analysis"] = self.analyze_patterns(user_input)
            elif task_name == "rag_retrieval":
                results["rag_results"] = self.rag.retrieve(user_input)
            elif task_name == "generate_answer":
                results["answer"] = self.generate_answer(user_input, results.get("rag_results"))

        # Log execution
        self.db.log_agent_action(
            self.name,
            "task_execution",
            {"plan": plan, "file_path": file_path, "user_input": user_input},
            {"tasks_completed": list(results.keys())}
        )

        context.update(results)
        return context

    def generate_summary(self, parsed_data):
        """Generate financial summary from parsed data"""
        if not parsed_data or "transactions" not in parsed_data:
            return {"error": "No data to summarize"}

        df = pd.DataFrame(parsed_data["transactions"])

        summary = {
            "total_transactions": len(df),
            "columns_available": list(df.columns)
        }

        # Calculate totals safely
        if 'debit' in df.columns:
            summary["total_debit"] = df['debit'].sum()
        else:
            summary["total_debit"] = 0

        if 'credit' in df.columns:
            summary["total_credit"] = df['credit'].sum()
        else:
            summary["total_credit"] = 0

        # Calculate average transaction
        total_amount = summary["total_debit"] + summary["total_credit"]
        summary["average_transaction"] = total_amount / len(df) if len(df) > 0 else 0

        # Date range if available
        if 'date' in df.columns:
            summary["period_start"] = df['date'].min()
            summary["period_end"] = df['date'].max()

        return summary

    def analyze_patterns(self, query):
        """Analyze financial patterns"""
        relevant_data = self.rag.retrieve(query)

        analysis = {
            "patterns": self._identify_patterns(relevant_data),
            "anomalies": self._detect_anomalies(relevant_data),
            "charts": self._generate_charts(relevant_data)
        }
        return analysis

    def _identify_patterns(self, data):
        return {"spending_trends": "stable", "unusual_activity": "none"}

    def _detect_anomalies(self, data):
        return {"anomalies": []}

    def _generate_charts(self, data):
        chart_paths = []
        try:
            plt.figure(figsize=(10, 6))
            chart_path = "./outputs/charts/analysis_chart.png"
            plt.savefig(chart_path)
            chart_paths.append(chart_path)
        except Exception as e:
            print(f"Chart generation error: {e}")
        return chart_paths

    def generate_answer(self, query, rag_results=None):
        """Generate answer using RAG results"""
        if not rag_results:
            return "I couldn't find specific information to answer your question. Please try uploading documents first."

        response = f"##  Query Results\n\n**Your question:** \"{query}\"\n\n**I found {len(rag_results)} relevant transactions:**\n\n"

        for i, result in enumerate(rag_results[:5], 1):
            doc = result.get('document', 'No details')
            score = result.get('score', 0)
            response += f"**{i}. {doc}**  \n"
            response += f"*Relevance score: {score:.3f}*  \n\n"

        return response