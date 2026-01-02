
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_PATH = BASE_DIR / "vector_store"
REPORTS_DIR = BASE_DIR / "reports"

# Model Configurations
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base"

# RAG Parameters
RETRIEVER_K = 5
GENERATION_MAX_LENGTH = 512
GENERATION_TEMP = 0.3

# Evaluation
EVAL_QUESTIONS_PATH = DATA_DIR / "eval_questions.txt" # Future proofing
