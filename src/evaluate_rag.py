
import pandas as pd
import time
import logging
from typing import List, Dict
from src.rag_pipeline import ComplaintRAG
from src.config import REPORTS_DIR

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_questions() -> List[str]:
    """
    Loads evaluation questions. 
    In a future iteration, this could read from src.config.EVAL_QUESTIONS_PATH.
    For now, returns a hardcoded list of representative PM questions.
    """
    return [
        "What are the common complaints about credit card fees?",
        "Why are customers unhappy with money transfers?",
        "What issues do people face with personal loans?",
        "Are there any recurring problems with savings accounts?",
        "How do customers describe billing disputes?",
        "What are the complaints regarding fraud protection?",
        "Do customers mention long wait times significantly?",
        "What specifically are customers saying about 'unexpected' charges?"
    ]

def evaluate_pipeline(rag: ComplaintRAG, questions: List[str]) -> pd.DataFrame:
    """
    Runs the RAG pipeline on a set of questions and returns a DataFrame of results.
    """
    results = []
    logger.info(f"Starting evaluation of {len(questions)} questions...")
    
    for i, q in enumerate(questions):
        logger.info(f"Processing Q{i+1}: {q}")
        start_t = time.time()
        
        try:
            # 1. Retrieval
            docs = rag.retrieve_only(q)
            # Format sources for readability in table
            sources = [
                f"[{d.metadata.get('Product','?')}] {d.metadata.get('Complaint ID','?')}" 
                for d in docs[:3]
            ]
            source_str = "; ".join(sources)
            
            # 2. Generation
            answer_text = rag.query(q)
            
            elapsed = time.time() - start_t
            
            results.append({
                "Question": q,
                "Generated Answer": answer_text.strip(),
                "Retrieved Context IDs": source_str,
                "Latency (s)": round(elapsed, 2),
                "Manual Relevance Rating (1-5)": " " # Placeholder for human review
            })
        except Exception as e:
            logger.error(f"Error processing question '{q}': {e}")
            results.append({
                "Question": q,
                "Generated Answer": "ERROR",
                "Retrieved Context IDs": "ERROR",
                "Latency (s)": 0,
                "Manual Relevance Rating (1-5)": "0" 
            })
            
    return pd.DataFrame(results)

def main():
    rag = ComplaintRAG()
    questions = load_questions()
    
    df_results = evaluate_pipeline(rag, questions)
    
    # Save Report
    report_path = REPORTS_DIR / "rag_evaluation_enhanced.csv"
    df_results.to_csv(report_path, index=False)
    logger.info(f"Evaluation complete. Results saved to {report_path}")
    
    # Also save as Markdown for referencing
    md_path = REPORTS_DIR / "rag_evaluation_enhanced.md"
    with open(md_path, "w") as f:
        f.write(f"# RAG Enhanced Evaluation Report\n\nDate: {time.strftime('%Y-%m-%d')}\n\n")
        f.write(df_results.to_markdown(index=False))
    logger.info(f"Markdown report saved to {md_path}")

if __name__ == "__main__":
    main()
