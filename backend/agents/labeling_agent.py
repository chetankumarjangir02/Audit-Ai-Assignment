
import pandas as pd
import re
import json
import os
from datetime import datetime
from backend.core.qa_generator import QAGenerator

class LabelingAgent:
    def __init__(self, db):
        self.db = db
        self.qa_generator = QAGenerator(db)
        self.name = "LabelingAgent"

        self.patterns = {
            'date': [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b\d{4}-\d{2}-\d{2}\b',
            ],
            'amount': [
                r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?',
                r'\d+(?:\.\d{2})?'
            ]
        }

        self.field_mappings = {
            'date': ['date', 'transaction date', 'posting date'],
            'description': ['description', 'transaction description', 'details', 'narration'],
            'debit': ['debit', 'withdrawal', 'debit amount'],
            'credit': ['credit', 'deposit', 'credit amount'],
            'balance': ['balance', 'running balance', 'available balance']
        }

    def process(self, context):
        parsed = context.get("parsed_data") or {}
        file_path = context.get("file_path", "")
        if not parsed or "transactions" not in parsed:
            context["labeling_error"] = "No transactions to label"
            # ensure key exists as empty list
            context["labeled_data"] = []
            return context

        document_id = context.get("document_id") or (os.path.basename(file_path).replace('.', '_') if file_path else "unknown")
        transactions = parsed.get("transactions", [])
        labeled = []

        for t in transactions:
            item = self.label_transaction(t)
            if item:
                item["source_document_id"] = document_id
                item["created_at"] = datetime.utcnow().isoformat()
                labeled.append(item)

        # Ensure labeled is a list (maybe empty)
        labeled = labeled or []

        if labeled:
            try:
                if hasattr(self.db, "store_labeled_data"):
                    self.db.store_labeled_data(labeled, document_id)
            except Exception:
                pass
            # persist outputs locally for inspection
            self.generate_output_files(labeled, document_id)
            try:
                qa_pairs = self.qa_generator.generate_qa_pairs(labeled)
                if qa_pairs and hasattr(self.db, "store_qa_pairs"):
                    self.db.store_qa_pairs(qa_pairs, document_id)
                    context["qa_pairs"] = qa_pairs
            except Exception:
                pass

        context["labeled_data"] = labeled
        context["document_id"] = document_id
        context["new_labeled_data"] = bool(labeled)

        try:
            if hasattr(self.db, "log_agent_action"):
                self.db.log_agent_action(self.name, "transaction_labeling", {"count": len(transactions)}, {"labeled_count": len(labeled)})
        except Exception:
            pass

        return context

    def label_transaction(self, transaction):
        if not transaction:
            return None
        labeled = {"original_data": str(transaction), "labeled_at": datetime.utcnow().isoformat(), "confidence": 1.0}
        # If dict: map fields
        if isinstance(transaction, dict):
            for std_field, candidates in self.field_mappings.items():
                for c in candidates:
                    if c in transaction and transaction.get(c) not in (None, ""):
                        labeled[std_field] = transaction.get(c)
                        break
        # string-based heuristics
        s = str(transaction).lower()
        for p in self.patterns['date']:
            m = re.search(p, s)
            if m and 'date' not in labeled:
                labeled['date'] = m.group()
        # numeric extraction for amounts
        for p in self.patterns['amount']:
            m = re.search(p, s)
            if m:
                val = m.group().replace(',', '').replace('$','')
                try:
                    num = float(val)
                    if 'debit' in s or 'dr' in s:
                        labeled['debit'] = num
                    elif 'credit' in s or 'cr' in s or 'deposit' in s:
                        labeled['credit'] = num
                    else:
                        # fallback
                        labeled['debit'] = num
                except Exception:
                    pass
                break

        labeled = self._clean_labeled_data(labeled)
        labeled["category"] = self.categorize_transaction(labeled)
        return labeled

    def _safe_float(self, v):
        try:
            if v is None or v == "":
                return 0.0
            if isinstance(v, (int, float)):
                return float(v)
            s = str(v).replace('$','').replace(',','').strip()
            return float(s) if s else 0.0
        except Exception:
            return 0.0

    def _clean_labeled_data(self, labeled):
        debit = self._safe_float(labeled.get('debit', 0))
        credit = self._safe_float(labeled.get('credit', 0))
        if debit > 0 and credit > 0:
            if debit >= credit:
                credit = 0.0
            else:
                debit = 0.0
        labeled['debit'] = debit
        labeled['credit'] = credit
        if 'description' not in labeled or not labeled.get('description'):
            labeled['description'] = labeled.get('original_data', '')[:120]
        return labeled

    def categorize_transaction(self, t):
        desc = str(t.get('description',"")).lower()
        debit = t.get('debit',0) or 0
        credit = t.get('credit',0) or 0
        categories = {
            'food_dining': ['restaurant','grocery','supermarket','food','coffee','cafe'],
            'transportation': ['uber','lyft','taxi','gas','fuel','transport'],
            'entertainment': ['movie','netflix','spotify','entertainment'],
            'shopping': ['amazon','walmart','target','store','shopping'],
            'utilities': ['electric','water','internet','phone','gas bill'],
            'salary_income': ['salary','payroll','deposit','income'],
            'transfer': ['transfer','zelle','venmo','paypal']
        }
        for k, keys in categories.items():
            if any(x in desc for x in keys):
                return k
        if credit > 1000:
            return 'salary_income'
        if debit > 0 and debit < 50:
            return 'food_dining'
        return 'other'

    def generate_output_files(self, labeled_data, document_id):
        try:
            os.makedirs('./datasets/labeled_data', exist_ok=True)
            os.makedirs('./outputs/labeled_docs', exist_ok=True)
            rows = []
            for it in labeled_data:
                rows.append({
                    'date': it.get('date',''),
                    'description': it.get('description',''),
                    'debit': it.get('debit',0),
                    'credit': it.get('credit',0),
                    'balance': it.get('balance',0),
                    'category': it.get('category','other')
                })
            df = pd.DataFrame(rows)
            csv_path = f'./datasets/labeled_data/{document_id}_labeled.csv'
            human_csv = f'./outputs/labeled_docs/{document_id}_human_readable.csv'
            df.to_csv(csv_path, index=False)
            df.to_csv(human_csv, index=False)
            json_path = f'./datasets/labeled_data/{document_id}_labeled.json'
            with open(json_path, 'w') as f:
                json.dump(labeled_data, f, default=str, indent=2)
        except Exception:
            pass


