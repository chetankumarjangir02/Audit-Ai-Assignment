# Audit Intelligence System — Summary

The Audit Intelligence System is a fully local, agent-driven financial audit assistant capable of parsing bank statements, labeling transactions, generating summaries and charts, answering natural-language questions using RAG, and producing complete audit reports — all without using any external LLM APIs.

**The system integrates:**

**Streamlit UI for interaction**

**LangGraph (open-source SDK) for agent workflow orchestration if not work then fallback simple graph**

**MongoDB for storage**

**FAISS-based RAG for semantic search**

**Custom agents (Planner, Executor, Reviewer, Labeler)**

**CSV/PDF/TXT parsing, transaction classification, and category detection**

**Report generator and fine-tuning pipeline**

# LangGraph Orchestrator
Nodes:
planner → executor → reviewer → labeler
All tasks execute synchronously 

# 🚀 How to Run Locally
# Clone Repository

**git clone https://github.com/chetankumarjangir02/Audit-Ai-Assignment**

# Install Dependencies

**Python -m venv .venv**

**.venv\Scripts\activate**

**pip install -r requirements.txt**

# Start App

**streamlit run ui/streamlit_app.py**

# Folder Structure
```
project/
├── 📁 backend/          # Core backend systems
│   ├── 📁 orchestrator/ # LangGraph workflow
│   ├── 📁 agents/       # AI agents (planner, executor, reviewer, labeling)
│   ├── 📁 core/         # Database, RAG, fine-tuning, QA systems
│   └── main_system.py   # Main entry point
├── 📁 ui/               # Streamlit frontend
├── 📁 utils/            # File parsing & reporting utilities
├── 📁 datasets/         # Raw & labeled data storage
├── 📁 outputs/          # Generated outputs (reports, charts)
├── requirements.txt
└── README.md
```

# Demo File


https://github.com/user-attachments/assets/8144cd3a-366c-4bca-92a3-058b9a16e07d

# Generated Report file

[audit_report_20251117_211534.docx](https://github.com/user-attachments/files/23587202/audit_report_20251117_211534.docx)

# sample test data file
[bank_statement_feb_2024.csv](https://github.com/user-attachments/files/23587254/bank_statement_feb_2024.csv)
