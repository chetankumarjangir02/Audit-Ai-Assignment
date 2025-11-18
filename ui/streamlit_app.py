
import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from backend.main_system import audit_system
from utils.file_parser import FileParser
from backend.core.qa_system import DocumentQASystem


class AuditIntelligenceUI:

    def __init__(self):
        self.system = audit_system
        self.file_parser = FileParser()
        self.qa_system = self.system.qa_system
        self.setup_page()

    def setup_page(self):
        st.set_page_config(page_title="Audit Intelligence System Assignment", page_icon="🔍", layout="wide")
        st.title("🔍 Audit Intelligence System ")
        st.markdown("Local, Autonomous Financial Audit Assistant")


    def run(self):
        if 'current_document_id' not in st.session_state:
            st.session_state.current_document_id = None
        if 'qa_history' not in st.session_state:
            st.session_state.qa_history = []
        if 'uploaded_documents' not in st.session_state:
            st.session_state.uploaded_documents = []

        st.sidebar.title("Navigation")
        mode = st.sidebar.selectbox(
            "Choose Mode",
            ["Document Upload", "Query System", "Fine-Tuning", "Reports", "System Status"]
        )

        if mode == "Document Upload":
            self.document_upload_page()
        elif mode == "Query System":
            self.query_system_page()
        elif mode == "Fine-Tuning":
            self.fine_tuning_page()
        elif mode == "Reports":
            self.reports_page()
        elif mode == "System Status":
            self.system_status_page()

    def document_upload_page(self):

        st.header("📄 Document Upload & Processing")

        uploaded_file = st.file_uploader("Upload Bank Statement", type=['csv','pdf','txt'])

        if uploaded_file is not None:

            file_path = f"./datasets/raw_data/{uploaded_file.name}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"Uploaded {uploaded_file.name}")

            document_id = uploaded_file.name.replace('.', '_')

            if document_id not in st.session_state.uploaded_documents:
                st.session_state.uploaded_documents.append(document_id)

            st.session_state.current_document_id = document_id

            col1, col2 = st.columns(2)

            with col1:
                if st.button("📊 Preview File"):
                    preview = self.file_parser.parse_file(file_path)
                    if "error" in preview:
                        st.error(preview["error"])
                    else:
                        st.json(preview)

            with col2:
                if st.button("🚀 Process Document"):
                    with st.spinner("Processing..."):
                        result = self.system.process_request(
                            user_input="Please parse and label this bank statement",
                            file_path=file_path,
                            document_id=document_id
                        )

                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success(" Document processed successfully!")

                        # TABLE DISPLAY
                        if "labeled_data" in result and result["labeled_data"]:
                            st.subheader("📊 Labeled Transactions")
                            df = pd.DataFrame(result["labeled_data"])
                            st.dataframe(df.head(20))

                        elif "parsed_data" in result and "transactions" in result["parsed_data"]:
                            st.subheader("📄 Parsed Transactions")
                            df = pd.DataFrame(result["parsed_data"]["transactions"])
                            st.dataframe(df.head(20))

                        # SUMMARY
                        if "summary" in result:
                            st.subheader(" Financial Summary")
                            summary = result["summary"]

                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Total Transactions", summary.get("total_transactions", 0))
                            c2.metric("Total Debit", f"${summary.get('total_debit', 0):.2f}")
                            c3.metric("Total Credit", f"${summary.get('total_credit', 0):.2f}")
                            c4.metric("Average Transaction", f"${summary.get('average_transaction', 0):.2f}")

                        # CHARTS
                        st.subheader(" Transaction Analysis")

                        if "labeled_data" in result and result["labeled_data"]:
                            self.display_transaction_charts(result["labeled_data"])

                        elif "parsed_data" in result and "transactions" in result["parsed_data"]:
                            self.display_transaction_charts(result["parsed_data"]["transactions"])

        if st.session_state.uploaded_documents:
            st.sidebar.subheader("Uploaded Documents")
            selected = st.sidebar.selectbox(
                "Select Document",
                st.session_state.uploaded_documents,
                index=(
                    st.session_state.uploaded_documents.index(st.session_state.current_document_id)
                    if st.session_state.current_document_id in st.session_state.uploaded_documents
                    else 0
                )
            )
            st.session_state.current_document_id = selected

    #
    def query_system_page(self):

        st.header(" Query System")

        doc_id = st.session_state.get("current_document_id")
        if not doc_id:
            st.warning("Upload a document first.")
            return

        st.info(f"Querying document: {doc_id}")

        query = st.text_input("Ask a question:")

        if st.button("🔍 Search") and query:

            with st.spinner("Thinking..."):
                result = self.qa_system.answer_question(query, doc_id)

            confidence = result.get("confidence", 0)

            if confidence >= 0.7:
                st.success(f"Confidence: {confidence:.2f}")
            elif confidence >= 0.4:
                st.warning(f"Confidence: {confidence:.2f}")
            else:
                st.error(f"Confidence: {confidence:.2f}")

            st.write(result.get("answer"))

            st.session_state.qa_history.append({
                "question": query,
                "answer": result.get("answer"),
                "confidence": confidence,
                "timestamp": datetime.now()
            })

        self._show_sample_questions()
        self._show_qa_history()

    def _show_sample_questions(self):
        st.subheader("Sample Questions")
        sample = [
            "What are the total debits?",
            "what are the total credit?",
            "What categories are in my transactions?",
            "What is the net cash flow?",
            "List all salary-related transactions",
            "Show me spending by category",
            "What is the date range of transactions?",
            "Find transactions above $1000"
        ]
        c = st.columns(2)
        for i, q in enumerate(sample):
            if c[i % 2].button(q):
                st.info(f"Try typing: {q}")

    def _show_qa_history(self):
        if not st.session_state.qa_history:
            return
        st.subheader("Recent Q&A")
        for qa in reversed(st.session_state.qa_history[-6:]):
            with st.expander(f"Q: {qa['question']}"):
                st.write(qa['answer'])
                st.caption(f"Confidence: {qa['confidence']:.2f} at {qa['timestamp'].strftime('%H:%M:%S')}")

    def fine_tuning_page(self):
        st.header(" Fine-Tuning")
        if st.button("Trigger Fine-Tuning Now"):
            with st.spinner("Working..."):
                res = self.system.fine_tuning_manager.trigger_fine_tuning()
            st.json(res)

    def reports_page(self):
        st.header("📋 Reports")

        doc_id = st.session_state.get("current_document_id")
        if not doc_id:
            st.warning("Upload a document first.")
            return

        if st.button("Generate Comprehensive Report"):
            from utils.reporting import ReportGenerator
            with st.spinner("Generating..."):
                rg = ReportGenerator(self.system.db)
                path = rg.generate_audit_report(doc_id)

            if path and os.path.exists(path):
                st.success("Report ready")
                with open(path, "rb") as f:
                    st.download_button("Download Report", f, file_name=os.path.basename(path))
            else:
                st.error("Failed")

    def system_status_page(self):
        st.header("🖥️ System Status")
        st.json(self.system.get_system_status())

    def display_transaction_charts(self, data):

        if not data:
            st.info("No data for charts.")
            return

        df = pd.DataFrame(data)

        # Clean numbers
        for col in ['debit', 'credit', 'balance']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Pie chart
        if "category" in df.columns:
            cat = df["category"].value_counts()
            fig = px.pie(values=cat.values, names=cat.index, title="Category Breakdown")
            st.plotly_chart(fig)

        # Debit vs Credit
        if "debit" in df.columns or "credit" in df.columns:
            totals = pd.DataFrame({
                "Type": ["Debit", "Credit"],
                "Amount": [df.get("debit", pd.Series()).sum(), df.get("credit", pd.Series()).sum()]
            })
            fig2 = px.bar(totals, x="Type", y="Amount", title="Debit vs Credit")
            st.plotly_chart(fig2)


if __name__ == "__main__":
    app = AuditIntelligenceUI()
    app.run()
