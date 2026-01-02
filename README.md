# Intelligent Complaint Analysis - RAG Chatbot

## Overview
CrediTrust Financial is building an AI-powered internal tool to analyze customer complaints. This project implements a **Retrieval-Augmented Generation (RAG)** chatbot that allows stakeholders to ask natural language questions about customer feedback across Credit Cards, Loans, Savings, and Money Transfers.

## Key Features
-   **Multi-Product Support**: Handles complaints from 4 major financial categories.
-   **Semantic Search**: Uses vector embeddings to find relevant narratives.
-   **Generative Answers**: Synthesizes insights using an LLM.
-   **Interactive UI**: Simple chat interface for non-technical users.

## Project Structure
```
rag-complaint-chatbot/
├── data/               # Raw and processed datasets (gitignored)
├── vector_store/       # ChromaDB/FAISS index (gitignored)
├── notebooks/          # Analysis and experiments
├── reports/            # Project reports and summaries
├── src/                # Source code for pipeline
├── app.py              # Main application entry point
└── requirements.txt    # Project dependencies
```

## Progress
- [x] **Task 1: EDA & Preprocessing**: Data filtered and cleaned. See [Interim Report](reports/interim_report.md).
- [x] **Task 2: Vector Store**: Chunking and Indexing.
- [x] **Task 3: RAG Core**: Retrieval and Generation logic. See [Evaluation Report](reports/rag_evaluation.md).
- [x] **Task 4: UI**: Gradio application.

## Setup & Usage
1.  **Install dependencies**: 
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the UI**:
    ```bash
    python app.py
    ```
3.  **Access**: Open the URL shown in the terminal (usually `http://127.0.0.1:7860`).

## Reports
-   [Interim Report (Tasks 1-2)](reports/interim_report.md)
-   [Final Report (Tasks 1-4)](reports/final_report.md)
