
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_STORE_PATH = 'vector_store'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

def main():
    print(f"Loading Vector Store from {VECTOR_STORE_PATH}...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vector_store = Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=embeddings
        )
        
        # Count items
        # There is no direct .count() in standard interface easily maybe, but we can try getting all IDs or just search
        # or vector_store._collection.count()
        count = vector_store._collection.count()
        print(f"Total documents in store: {count}")
        
        if count == 0:
            print("Store is empty!")
            return

        # Test Search
        query = "credit card late fee"
        print(f"\nTest Query: '{query}'")
        results = vector_store.similarity_search(query, k=3)
        
        for i, res in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Content: {res.page_content[:200]}...")
            print(f"Metadata: {res.metadata}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
