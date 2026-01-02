
import logging
from typing import Optional, List, Dict, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import src.config as cfg

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComplaintRAG:
    """
    Retrieval-Augmented Generation (RAG) pipeline for analyzing customer complaints.
    
    Attributes:
        vector_store_path (str): Path to the ChromaDB vector store.
        embedding_model (str): Name of the HuggingFace embedding model.
        llm_model (str): Name of the HuggingFace LLM model.
    """
    
    def __init__(self, 
                 vector_store_path: str = str(cfg.VECTOR_STORE_PATH),
                 embedding_model: str = cfg.EMBEDDING_MODEL_NAME,
                 llm_model: str = cfg.LLM_MODEL_NAME):
        """
        Initializes the RAG pipeline components.
        
        Args:
            vector_store_path (str): Path to persistent vector store.
            embedding_model (str): HF model identifier for embeddings.
            llm_model (str): HF model identifier for generation.
        """
        self.vector_store_path = vector_store_path
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        
        # Lazy loading components
        self._vector_store = None
        self._retriever = None
        self._llm = None
        self._chain = None

    def _load_retriever(self):
        """Loads and configures the ChromaDB retriever."""
        if not self._retriever:
            try:
                logger.info(f"Loading embedding model: {self.embedding_model_name}")
                embedding_fn = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
                
                logger.info(f"Loading vector store from: {self.vector_store_path}")
                self._vector_store = Chroma(
                    persist_directory=self.vector_store_path,
                    embedding_function=embedding_fn
                )
                self._retriever = self._vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": cfg.RETRIEVER_K}
                )
            except Exception as e:
                logger.error(f"Failed to load retriever: {e}")
                raise RuntimeError("Critical Error: Could not load Vector Store.") from e
        return self._retriever

    def _load_llm(self):
        """Loads the LLM pipeline."""
        if not self._llm:
            try:
                logger.info(f"Loading LLM model: {self.llm_model_name}")
                tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(self.llm_model_name)
                
                pipe = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=cfg.GENERATION_MAX_LENGTH,
                    truncation=True,
                    temperature=cfg.GENERATION_TEMP
                )
                self._llm = HuggingFacePipeline(pipeline=pipe)
            except Exception as e:
                logger.error(f"Failed to load LLM: {e}")
                raise RuntimeError("Critical Error: Could not load LLM.") from e
        return self._llm

    def get_chain(self):
        """Constructs and returns the RAG execution chain."""
        if not self._chain:
            retriever = self._load_retriever()
            llm = self._load_llm()
            
            template = """
            You are a helpful financial analyst assistant for CrediTrust. 
            Answer the question based ONLY on the following context. 
            If the answer is not in the context, say "I don't have enough information."
            
            Context:
            {context}
            
            Question: 
            {question}
            
            Answer:
            """
            prompt = PromptTemplate.from_template(template)
            
            self._chain = (
                {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            logger.info("RAG Chain initialized successfully.")
        return self._chain

    def _format_docs(self, docs: List[Document]) -> str:
        """Formats retrieved documents into a single context string."""
        return "\n\n".join([d.page_content for d in docs])

    def query(self, question: str) -> str:
        """
        Executes a query against the RAG pipeline.
        
        Args:
            question (str): User's question.
            
        Returns:
            str: Generated answer.
        """
        chain = self.get_chain()
        logger.info(f"Processing query: {question}")
        try:
            return chain.invoke(question)
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return "Error: Unable to generate response."

    def retrieve_only(self, question: str) -> List[Document]:
        """
        Retrieves relevant documents without generation.
        
        Args:
            question (str): Query string.
            
        Returns:
            List[Document]: List of retrieved documents.
        """
        retriever = self._load_retriever()
        return retriever.invoke(question)

if __name__ == "__main__":
    # Test block
    rag = ComplaintRAG()
    res = rag.query("What are the credit card fees?")
    print(res)
