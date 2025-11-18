
from datetime import datetime
import json
from typing import Dict, Any, List
class ReviewerAgent:
    def __init__(self, db):
        self.db = db
        self.name = "ReviewerAgent"
        self.confidence_threshold = 0.7

    def process(self, context):
        """Validate executor outputs and check consistency"""
        # Validate outputs
        validation_results = {
            "factual_consistency": self.check_factual_consistency(context),
            "completeness": self.check_completeness(context),
            "confidence": self.calculate_confidence(context)
        }

        # Determine if re-execution is needed
        needs_retry = (
            validation_results["confidence"] < self.confidence_threshold or
            not validation_results["factual_consistency"]
        )

        review_result = {
            "validation_results": validation_results,
            "needs_retry": needs_retry,
            "recommendations": self.generate_recommendations(validation_results)
        }

        # Log review action
        self.db.log_agent_action(
            self.name,
            "output_review",
            context,
            review_result,
            validation_results["confidence"]
        )

        context["review"] = review_result
        context["confidence"] = validation_results["confidence"]

        return context

    def check_factual_consistency(self, data):
        """Check if outputs are factually consistent"""
        # Basic consistency check
        if "summary" in data and "parsed_data" in data:
            summary = data["summary"]
            parsed_data = data["parsed_data"]
            if "total_transactions" in summary:
                expected_count = parsed_data.get("transaction_count", 0)
                return summary["total_transactions"] == expected_count
        return True

    def check_completeness(self, data):
        """Check if all required outputs are present"""
        required_fields = ["user_input"]
        if data.get("file_path"):
            required_fields.extend(["parsed_data", "summary"])
        return all(field in data for field in required_fields)

    def calculate_confidence(self, data):
        """Calculate confidence score for the outputs"""
        confidence = 1.0

        # Reduce confidence for errors
        if "summary" in data and data["summary"].get("error"):
            confidence *= 0.5

        if "parsed_data" in data and data["parsed_data"].get("error"):
            confidence *= 0.3

        # Increase confidence for successful processing
        if "labeled_data" in data and len(data["labeled_data"]) > 0:
            confidence *= 1.2

        return min(1.0, confidence)

    def generate_recommendations(self, validation_results):
        """Generate recommendations for improvement"""
        recommendations = []

        if validation_results["confidence"] < self.confidence_threshold:
            recommendations.append("Consider re-executing with more context")

        if not validation_results["completeness"]:
            recommendations.append("Missing required output fields")

        if not validation_results["factual_consistency"]:
            recommendations.append("Data inconsistencies detected")

        return recommendations