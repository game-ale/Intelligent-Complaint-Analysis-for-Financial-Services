
import pandas as pd
import os
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import shutil

# Configurations
DATA_PATH = 'data/filtered_complaints.csv'
SAMPLE_SIZE = 10000
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
VECTOR_STORE_PATH = 'vector_store'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

def main():
    # 1. Load Data
    print(f"Loading data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Please run Task 1 first.")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Total records: {len(df)}")

    # 2. Stratified Sampling
    print(f"performing stratified sampling (n={SAMPLE_SIZE})...")
    # Calculate proportion of each category
    # usage of groupby and apply to sample valid amount
    try:
        # We want to sample SAMPLE_SIZE total
        # We can use train_test_split with stratify, or just pandas sample
        # Pandas sample doesn't strictly support "stratified n" easily in one shot without weights or loops
        # But we can do:
        df_sample = df.groupby('Category', group_keys=False).apply(
            lambda x: x.sample(int(np.rint(SAMPLE_SIZE * len(x) / len(df))))
        )
        
        # If rounding caused issues, just take head or random sample
        if len(df_sample) > SAMPLE_SIZE:
             df_sample = df_sample.sample(SAMPLE_SIZE)
        
        # Fallback if manual calculation strictly required, but this is usually fine.
        # Let's simple use sklearn for robustness
        from sklearn.model_selection import train_test_split
        # This gives us a test set of size X, but we want a specific size. 
        # Easier to specific frac. 
        frac = SAMPLE_SIZE / len(df)
        _, df_sample = train_test_split(df, test_size=frac, stratify=df['Category'], random_state=42)
        
    except ImportError:
        import numpy as np
        # Fallback manual
        print("sklearn not found, using pandas manual stratification")
        df_sample = df.groupby('Category', group_keys=False).apply(lambda x: x.sample(frac=SAMPLE_SIZE/len(df)))

    print(f"Sampled records: {len(df_sample)}")
    print("Sample distribution:")
    print(df_sample['Category'].value_counts())
    
    # Save sample (optional but good for debugging)
    df_sample.to_csv('data/sampled_complaints.csv', index=False)

    # 3. Document Preparation
    # We need to convert DataFrame to LangChain Documents
    # We want to keep metadata
    print("Converting to Documents...")
    # Clean NaN content just in case
    df_sample['cleaned_narrative'] = df_sample['cleaned_narrative'].fillna('')
    
    # Loader
    loader = DataFrameLoader(df_sample, page_content_column="cleaned_narrative")
    documents = loader.load()
    
    # 4. Chunking
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(texts)}")
    
    # 5. Embedding & Indexing
    print(f"Initializing embedding model ({EMBEDDING_MODEL})...")
    # Using local HF embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    print(f"Creating/Updating Vector Store at {VECTOR_STORE_PATH}...")
    # Check if vector store exists and clear it if we want a fresh start?
    # Task says "Create", usually implies fresh.
    if os.path.exists(VECTOR_STORE_PATH):
        shutil.rmtree(VECTOR_STORE_PATH) # Clear old test runs
        
    # Create Chroma
    # We pass 'ids' potentially? Chroma generates them if not provided.
    vector_store = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )
    
    # Persist (auto-persisted in newer chroma versions, but persist() is safe)
    # vector_store.persist() # Deprecated in newer langchain/chroma, often auto-saves.
    
    print("Vector Store successfully created.")

if __name__ == "__main__":
    main()
