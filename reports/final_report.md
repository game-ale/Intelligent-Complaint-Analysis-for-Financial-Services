# Intelligent Complaint Analysis - Final Report

## Introduction
The **Intelligent Complaint Analysis** project aims to transform raw customer feedback into actionable insights for CrediTrust Financial. By leveraging Retrieval-Augmented Generation (RAG), we have built a chatbot that allows Product Managers and Compliance teams to query thousands of unstructured complaint narratives instantly.

## Technical Architecture
The system follows a standard RAG pipeline:

1.  **Data Processing**: 
    -   Filtered 9.6M+ records down to ~330k relevant complaints across 4 categories: Credit Cards, Personal Loans, Savings Accounts, and Money Transfers.
    -   Cleaned and standardized text.
2.  **Vector Store**:
    -   **Sampling**: Stratified sample of 10,000 documents to ensure representative coverage.
    -   **Chunking**: RecursiveCharacterTextSplitter (500 chars, 50 overlap).
    -   **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384d).
    -   **Database**: ChromaDB (Percented locally).
3.  **RAG Core**:
    -   **Retriever**: ChromaDB similarity search (k=5).
    -   **Generator**: `google/flan-t5-base` (Seq2Seq LLM).
    -   **Prompting**: Context-aware instruction prompts to ground answers in facts.
4.  **Interface**:
    -   **Gradio Web App**: A user-friendly chat interface displaying both the AI answer and the cited source documents.

## System Evaluation
We performed a robust evaluation on 8 representative questions using the enhanced evaluation module (`src/evaluate_rag.py`).

| Question | Latency (s) | 
| :--- | :--- |
| Common complaints about credit card fees? | ~3.33s |
| Issues with money transfers? | ~7.23s |
| Issues with personal loans? | ~3.88s |
| ... | ... |

*Full detailed evaluation logs, including retrieved context IDs and manual relevance ratings, are available in [`reports/rag_evaluation_enhanced.md`](reports/rag_evaluation_enhanced.md).*

## Code Quality Improvements
Based on feedback, we implemented:
-   **Configuration Management**: Centralized `src/config.py`.
-   **Robustness**: Lazy loading of heavy models and try-except blocks in `src/rag_pipeline.py`.
-   **Type Safety**: Added type hints and docstrings.
-   **Extensibility**: Modular evaluation script.

## UI Showcase
The application provides a clean chat interface.

*(Please insert your screenshot of the running Gradio app here)*

**Features**:
-   **Chat Window**: Natural language interaction.
-   **Source Citations**: Collapsible/Listed sources with Product, Company, and Complaint ID for verification.

## Conclusion and Future Work
The prototype successfully demonstrates the value of RAG for complaint analysis. 
**Key Learnings**:
-   **Data Quality**: Filtering for detailed narratives is crucial.
-   **Model Choice**: `Flan-T5` is efficient for local CPU usage but larger models (like Llama 3) would provide more conversational fluency.
-   **Context**: Increasing the sample size from 10k to the full dataset would significantly improve recall for niche topics.

**Next Steps**:
-   Deploy to a cloud instance (AWS/GCP).
-   Scale the Vector Store to the full 330k dataset.
-   Implement feedback mechanisms for users to rate answers.
