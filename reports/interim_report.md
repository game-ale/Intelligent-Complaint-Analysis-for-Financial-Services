# Interim Report - Intelligent Complaint Analysis

## Task 1: Exploratory Data Analysis (EDA) and Preprocessing

### Objective
The goal was to filter the massive CFPB complaint dataset to focus on 4 key financial products and prepare the text for the RAG pipeline.

### Methodology
1.  **Filtering**: selected complaints belonging to 'Credit Card', 'Personal Loan', 'Savings Account', and 'Money Transfer'.
2.  **Cleaning**: Removed records with missing narratives. Normalized text (lowercasing).
3.  **Processing**: Saved the clean dataset to `data/filtered_complaints.csv`.

### Key Findings
-   **Volume**: Reduced the dataset from ~9.6 million raw records to **330,008** high-quality records with narratives.
-   **Distribution**:
    -   **Credit Cards** dominate the dataset (189k records), suggesting this will be the most query-rich category.
    -   **Money Transfers** follow with 98k.
    -   **Savings** (24k) and **Personal Loans** (17k) have smaller but sufficient volumes for RAG.
-   **Text Length**:
    -   Average complaint length is ~196 words.
    -   The distribution is right-skewed, with some complaints exceeding 6000 words. This confirms the need for **chunking** in Task 2 to fit within the embedding model's context window.

### Next Steps
-   Proceed to Task 2: Stratified sampling (10k) and Vector Store creation.
