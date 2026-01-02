
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    print("Found in langchain.text_splitter")
except ImportError as e:
    print(f"Failed langchain.text_splitter: {e}")

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("Found in langchain_text_splitters")
except ImportError as e:
    print(f"Failed langchain_text_splitters: {e}")
