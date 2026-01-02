
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configuration
VECTOR_STORE_PATH = 'vector_store'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL = 'google/flan-t5-base' # CPU friendly, good instruction following

class ComplaintRAG:
    def __init__(self):
        print("Initializing RAG Pipeline...")
        
        # 1. Load Vector Store (Retriever)
        print("Loading Vector Store...")
        self.embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_store = Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=self.embedding_fn
        )
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # 2. Load LLM (Generator)
        print(f"Loading LLM ({LLM_MODEL})...")
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
        
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            truncation=True,
            temperature=0.3 # Low temp for factual answers
        )
        
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        # 3. Define Prompt
        # Flan-T5 needs clear instructions.
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
        
        self.prompt = PromptTemplate.from_template(template)
        
        # 4. Build Chain
        self.chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        print("RAG Pipeline Initialized.")

    def _format_docs(self, docs):
        return "\n\n".join([d.page_content for d in docs])

    def query(self, question):
        print(f"Querying: {question}")
        response = self.chain.invoke(question)
        return response
    
    def retrieve_only(self, question):
        """Helper to inspect retrieved docs"""
        return self.retriever.invoke(question)

if __name__ == "__main__":
    # Simple test
    rag = ComplaintRAG()
    q = "What are the common complaints about credit card fees?"
    result = rag.query(q)
    print("\nFINAL ANSWER:")
    print(result)
