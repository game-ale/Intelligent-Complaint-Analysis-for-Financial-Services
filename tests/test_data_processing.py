
import unittest
import pandas as pd
from src.process_data import clean_text

class TestDataProcessing(unittest.TestCase):
    def test_clean_text(self):
        # Test basic lowercasing
        self.assertEqual(clean_text("HELLO"), "hello")
        
        # Test special char removal (xxx)
        self.assertEqual(clean_text("Account xxx123"), "account 123")
        
        # Test non-string input
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(123), "")

if __name__ == '__main__':
    unittest.main()
