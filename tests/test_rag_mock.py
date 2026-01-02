
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Put src in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_pipeline import ComplaintRAG

class TestRAGPipeline(unittest.TestCase):
    @patch('src.rag_pipeline.Chroma')
    @patch('src.rag_pipeline.HuggingFaceEmbeddings')
    @patch('src.rag_pipeline.AutoTokenizer')
    @patch('src.rag_pipeline.AutoModelForSeq2SeqLM')
    @patch('src.rag_pipeline.pipeline')
    @patch('src.rag_pipeline.HuggingFacePipeline')
    def test_initialization_and_query(self, mock_hf_pipe, mock_pipeline, mock_model, mock_tokenizer, mock_embeddings, mock_chroma):
        # Setup mocks
        mock_retriever = MagicMock()
        mock_chroma.return_value.as_retriever.return_value = mock_retriever
        
        # Mock the chain invocation
        rag = ComplaintRAG()
        
        # Mock the chain to return a string directly
        # Since self.chain is constructed using pipe syntax, it's a RunnableBinding. 
        # We can mock the invoke method on the chain object if we can access it, 
        # or implies that rag.chain.invoke returns "Mock Answer"
        rag.chain = MagicMock()
        rag.chain.invoke.return_value = "Mocked RAG Answer"
        
        # Test Query
        response = rag.query("Test Question")
        self.assertEqual(response, "Mocked RAG Answer")
        rag.chain.invoke.assert_called_with("Test Question")

if __name__ == '__main__':
    unittest.main()
