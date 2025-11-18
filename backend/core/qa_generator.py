import random
from datetime import datetime
import pandas as pd
import logging

class QAGenerator:
    def __init__(self, db):
        self.db = db
        self.logger = logging.getLogger(__name__)

    def generate_qa_pairs(self, labeled_data, num_pairs=20):
        """Generate Q&A pairs from labeled transaction data"""
        if not labeled_data:
            self.logger.warning("No labeled data available for Q&A generation")
            return self._generate_fallback_qa_pairs(labeled_data)

        try:
            df = self._clean_dataframe(labeled_data)

            if df.empty:
                return self._generate_fallback_qa_pairs(labeled_data)

            qa_pairs = []

            # Generate different types of questions
            question_generators = [
                (self._total_questions, 0.3),
                (self._category_questions, 0.25),
                (self._transaction_questions, 0.25),
                (self._date_questions, 0.1),
                (self._pattern_questions, 0.1),
            ]

            remaining_pairs = num_pairs
            for generator, weight in question_generators:
                pairs_to_generate = max(1, int(num_pairs * weight))
                try:
                    pairs = generator(df, pairs_to_generate)
                    if pairs:
                        qa_pairs.extend(pairs)
                        remaining_pairs -= len(pairs)
                except Exception as e:
                    self.logger.error(f"Error in {generator.__name__}: {e}")

            # Ensure we have enough pairs
            if len(qa_pairs) < num_pairs:
                additional = self._additional_questions(df, num_pairs - len(qa_pairs))
                qa_pairs.extend(additional)

            self.logger.info(f"Generated {len(qa_pairs)} Q&A pairs")
            return qa_pairs[:num_pairs]

        except Exception as e:
            self.logger.error(f"Error in Q&A generation: {e}")
            return self._generate_fallback_qa_pairs(labeled_data, num_pairs)

    def _clean_dataframe(self, labeled_data):
        """Clean and prepare dataframe for Q&A generation"""
        try:
            df = pd.DataFrame(labeled_data)

            # Remove MongoDB _id field if present
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)

            # Convert numeric columns
            numeric_columns = ['debit', 'credit', 'balance']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

            # Clean text columns
            text_columns = ['description', 'category', 'date']
            for col in text_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).fillna('Unknown')

            return df
        except Exception as e:
            self.logger.error(f"Error cleaning dataframe: {e}")
            return pd.DataFrame()

    def _generate_fallback_qa_pairs(self, labeled_data, num_pairs=3):
        """Generate fallback Q&A pairs"""
        fallback_pairs = [
            {
                "question": "What transactions are available in this document?",
                "answer": "This document contains financial transactions including debits, credits, and balance information.",
                "type": "general",
                "confidence": 0.8
            },
            {
                "question": "How many transactions are in this statement?",
                "answer": f"There are {len(labeled_data)} transactions in this statement.",
                "type": "count",
                "confidence": 0.9
            },
            {
                "question": "What type of financial data is available?",
                "answer": "The data includes transaction dates, descriptions, debit/credit amounts, balances, and categories.",
                "type": "general",
                "confidence": 0.8
            }
        ]
        return fallback_pairs[:num_pairs]

    def _total_questions(self, df, count):
        """Generate questions about totals"""
        questions = []

        if 'debit' in df.columns:
            total_debit = df['debit'].sum()
            if total_debit > 0:
                questions.append({
                    "question": "What is the total amount of debits?",
                    "answer": f"The total debit amount is ${total_debit:.2f}",
                    "type": "total_debit",
                    "confidence": 0.9
                })

        if 'credit' in df.columns:
            total_credit = df['credit'].sum()
            if total_credit > 0:
                questions.append({
                    "question": "What is the total amount of credits?",
                    "answer": f"The total credit amount is ${total_credit:.2f}",
                    "type": "total_credit",
                    "confidence": 0.9
                })

        if 'debit' in df.columns and 'credit' in df.columns:
            net_flow = total_credit - total_debit
            flow_type = "positive" if net_flow >= 0 else "negative"
            questions.append({
                "question": "What is the net cash flow?",
                "answer": f"The net cash flow is ${net_flow:.2f} ({flow_type})",
                "type": "net_flow",
                "confidence": 0.9
            })

        return questions[:count]

    def _category_questions(self, df, count):
        """Generate questions about categories"""
        questions = []

        if 'category' in df.columns:
            category_counts = df['category'].value_counts()
            if not category_counts.empty:
                # Most common category
                top_category = category_counts.index[0]
                questions.append({
                    "question": "Which category has the most transactions?",
                    "answer": f"'{top_category}' has the most transactions with {category_counts.iloc[0]} occurrences",
                    "type": "category_count",
                    "confidence": 0.9
                })

                # Category list
                if len(category_counts) > 1:
                    categories_list = ", ".join([f"{cat} ({count})" for cat, count in category_counts.head(5).items()])
                    questions.append({
                        "question": "What categories are in the transactions?",
                        "answer": f"Categories include: {categories_list}",
                        "type": "category_list",
                        "confidence": 0.8
                    })

        return questions[:count]

    def _transaction_questions(self, df, count):
        """Generate questions about individual transactions"""
        questions = []

        # Largest debit
        if 'debit' in df.columns:
            non_zero_debits = df[df['debit'] > 0]
            if not non_zero_debits.empty:
                max_debit = non_zero_debits['debit'].max()
                max_debit_row = non_zero_debits[non_zero_debits['debit'] == max_debit].iloc[0]

                questions.append({
                    "question": "What was the largest debit transaction?",
                    "answer": f"The largest debit was ${max_debit:.2f} for '{max_debit_row.get('description', 'unknown')}'",
                    "type": "largest_debit",
                    "confidence": 0.9
                })

        # Largest credit
        if 'credit' in df.columns:
            non_zero_credits = df[df['credit'] > 0]
            if not non_zero_credits.empty:
                max_credit = non_zero_credits['credit'].max()
                max_credit_row = non_zero_credits[non_zero_credits['credit'] == max_credit].iloc[0]

                questions.append({
                    "question": "What was the largest credit transaction?",
                    "answer": f"The largest credit was ${max_credit:.2f} for '{max_credit_row.get('description', 'unknown')}'",
                    "type": "largest_credit",
                    "confidence": 0.9
                })

        return questions[:count]

    def _date_questions(self, df, count):
        """Generate questions about dates"""
        questions = []

        if 'date' in df.columns:
            try:
                dates = pd.to_datetime(df['date'], errors='coerce').dropna()
                if not dates.empty:
                    start_date = dates.min().strftime('%Y-%m-%d')
                    end_date = dates.max().strftime('%Y-%m-%d')
                    days_span = (dates.max() - dates.min()).days

                    questions.append({
                        "question": "What is the date range of these transactions?",
                        "answer": f"The transactions cover from {start_date} to {end_date} ({days_span} days)",
                        "type": "date_range",
                        "confidence": 0.9
                    })
            except:
                pass

        return questions[:count]

    def _pattern_questions(self, df, count):
        """Generate questions about patterns"""
        questions = []

        # Simple pattern questions
        if 'debit' in df.columns:
            debit_count = (df['debit'] > 0).sum()
            questions.append({
                "question": "How many debit transactions are there?",
                "answer": f"There are {debit_count} debit transactions",
                "type": "debit_count",
                "confidence": 0.9
            })

        if 'credit' in df.columns:
            credit_count = (df['credit'] > 0).sum()
            questions.append({
                "question": "How many credit transactions are there?",
                "answer": f"There are {credit_count} credit transactions",
                "type": "credit_count",
                "confidence": 0.9
            })

        return questions[:count]

    def _additional_questions(self, df, count):
        """Generate additional question types"""
        questions = []

        # Balance questions
        if 'balance' in df.columns and not df.empty:
            final_balance = df['balance'].iloc[-1]
            questions.append({
                "question": "What is the final balance?",
                "answer": f"The final balance is ${final_balance:.2f}",
                "type": "final_balance",
                "confidence": 0.8
            })

        # Transaction frequency
        total_transactions = len(df)
        questions.append({
            "question": "How many transactions are there in total?",
            "answer": f"There are {total_transactions} transactions in total",
            "type": "total_count",
            "confidence": 0.9
        })

        return questions[:count]
