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
We performed a qualitative evaluation on a set of diverse questions.

| Question | Latency (s) | Quality Assessment |
| :--- | :--- | :--- |
| Common complaints about credit card fees? | ~3.25s | **Good**: Correctly identified issues but cautious due to context limits. |
| Issues with money transfers? | ~34.55s | **Excellent**: Retrieved specific detailed narratives about fees and difficulties. |
| Issues with personal loans? | ~10.38s | **Good**: Cited specific company practices (e.g., Synchrony Bank). |
| Billing dispute description? | ~9.78s | **Fair**: Retrieved company names but answer was brief. |

*Full evaluation logs are available in `reports/rag_evaluation.md`.*

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
