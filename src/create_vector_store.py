
import pandas as pd
import numpy as np
import os
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
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
    try:
        from sklearn.model_selection import train_test_split
        # Calculate fraction for sampling
        frac = SAMPLE_SIZE / len(df)
        if frac > 1.0:
            frac = 1.0
        
        # We use train_test_split to get a stratified subset
        # train_test_split returns (train, test). We can treat the 'test' as our sample if size is small,
        # or 'train' if large. Here we want `frac` size. 
        # Actually StratifiedShuffleSplit or just train_test_split(test_size=frac)
        # If we want exactly X items, we can use test_size=int(SAMPLE_SIZE) (if <= len) or train_size...
        
        _, df_sample = train_test_split(
            df, 
            test_size=SAMPLE_SIZE, 
            stratify=df['Category'], 
            random_state=42
        )
    except Exception as e:
        print(f"Sampling failed or sklearn missing: {e}. Fallback to random sample.")
        df_sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)

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
