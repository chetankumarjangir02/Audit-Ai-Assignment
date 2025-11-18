import pandas as pd
import PyPDF2
import re
import os
from datetime import datetime

class FileParser:
    def __init__(self):
        self.supported = ['.csv', '.pdf', '.txt']

    def parse_file(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == '.csv':
                parsed = self._parse_csv(path)
            elif ext == '.pdf':
                parsed = self._parse_pdf(path)
            elif ext == '.txt':
                parsed = self._parse_txt(path)
            else:
                return {"error": f"Unsupported: {ext}"}
        except Exception as e:
            return {"error": str(e)}

        # Final normalization: always return a dict with transactions list
        if isinstance(parsed, dict):
            if "transactions" not in parsed:
                # Maybe returned metadata-only; wrap accordingly
                return {"transactions": parsed}
            return parsed
        elif isinstance(parsed, list):
            return {"transactions": parsed}
        else:
            return {"transactions": []}

    def _parse_csv(self, path: str):
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        transactions = []
        for _, row in df.iterrows():
            t = {}
            mapping = {
                'date': ['date','transaction date','post date','posting date'],
                'description': ['description','transaction description','details','narration'],
                'debit': ['debit','withdrawal','debit amount','amount debited'],
                'credit': ['credit','deposit','credit amount','amount credited'],
                'balance': ['balance','running balance','available balance']
            }
            for k, candidates in mapping.items():
                for cand in candidates:
                    if cand in row and pd.notna(row[cand]) and row[cand] != '':
                        val = row[cand]
                        if k in ['debit','credit','balance']:
                            try:
                                t[k] = float(str(val).replace(',','').replace('$','').strip())
                            except Exception:
                                t[k] = 0.0
                        else:
                            t[k] = val
                        break
            transactions.append(t)
        return {"file_name": os.path.basename(path), "file_type":"csv","transaction_count":len(transactions),"transactions":transactions,"parsed_at":datetime.utcnow().isoformat()}

    def _parse_pdf(self, path: str):
        transactions = []
        with open(path,'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for p in reader.pages:
                text = p.extract_text()
                transactions.extend(self._extract_transactions_from_text(text))
        return {"file_name": os.path.basename(path),"file_type":"pdf","transaction_count":len(transactions),"transactions":transactions,"parsed_at":datetime.utcnow().isoformat()}

    def _parse_txt(self, path: str):
        with open(path,'r',encoding='utf-8') as f:
            text = f.read()
        transactions = self._extract_transactions_from_text(text)
        return {"file_name": os.path.basename(path),"file_type":"txt","transaction_count":len(transactions),"transactions":transactions,"parsed_at":datetime.utcnow().isoformat()}

    def _extract_transactions_from_text(self, text: str):
        if not text:
            return []
        lines = text.splitlines()
        transactions = []
        for line in lines:
            l = line.strip()
            if not l:
                continue
            date_match = re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', l)
            amount_match = re.search(r'\$?(\d+[,\d]*\.\d{2})', l)
            if date_match or amount_match:
                transactions.append({
                    "raw_text": l,
                    "date": date_match.group() if date_match else "",
                    "amount": amount_match.group() if amount_match else "",
                    "description": l
                })
        return transactions
