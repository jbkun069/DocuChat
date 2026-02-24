import os
import sys

# Fix for Windows: Create a fake 'pwd' module before importing langchain
if sys.platform == 'win32':
    import types
    pwd = types.ModuleType('pwd')
    sys.modules['pwd'] = pwd

from langchain_community.document_loaders import TextLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "data", "AI.txt")

loader = TextLoader(file_path, encoding='utf-8')

try:
    my_documents = loader.load()
    
    content = my_documents[0].page_content
    print("--- FILE CONTENT ---")
    print(content)
    print("\n--- METADATA ---")
    print(my_documents[0].metadata)

except Exception as e:
    print(f"Oops! Something went wrong: {e}")