
import pandas as pd
from rag_pipeline import ComplaintRAG
import time

def main():
    print("Initializing RAG for Evaluation...")
    rag = ComplaintRAG()
    
    questions = [
        "What are the common complaints about credit card fees?",
        "Why are customers unhappy with money transfers?",
        "What issues do people face with personal loans?",
        "Are there any recurring problems with savings accounts?",
        "How do customers describe billing disputes?",
        "What are the complaints regarding fraud protection?"
    ]
    
    results = []
    
    print(f"Starting evaluation of {len(questions)} questions...")
    
    for i, q in enumerate(questions):
        print(f"\nProcessing Q{i+1}: {q}")
        start_t = time.time()
        
        # Get retrieval context for inspection
        docs = rag.retrieve_only(q)
        sources = [f"{d.metadata.get('Product','')} (ID: {d.metadata.get('Complaint ID','?')})" for d in docs[:2]]
        
        # Get Answer
        answer_text = rag.query(q)
        
        elapsed = time.time() - start_t
        
        results.append({
            "Question": q,
            "Generated Answer": answer_text.strip(),
            "Source Snippets": "; ".join(sources),
            "Latency (s)": round(elapsed, 2)
        })
        
    # Create DataFrame
    df_res = pd.DataFrame(results)
    
    # Save to Markdown
    md_table = df_res.to_markdown(index=False)
    
    report_content = f"""# RAG System Evaluation
    
## Qualitative Evaluation
Test Date: {time.strftime("%Y-%m-%d %H:%M:%S")}
Model: Google Flan-T5-Base
Embeddings: all-MiniLM-L6-v2

{md_table}

## Analysis
- **Relevance**: The model answers are restricted to the context provided.
- **Latency**: Average response time per query was {df_res['Latency (s)'].mean():.2f} seconds.
"""
    
    with open('reports/rag_evaluation.md', 'w') as f:
        f.write(report_content)
        
    print("Evaluation Complete. Report saved to reports/rag_evaluation.md")

if __name__ == "__main__":
    main()
