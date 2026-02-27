import os
import sys
from pathlib import Path

if sys.platform == "win32":
    import types
    pwd = types.ModuleType("pwd")
    sys.modules["pwd"] = pwd

from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_document(file_path : Path) -> list:
    """Load a document based on it's extension along with it's metadata"""
    try:
        if file_path.suffix.lower() == '.pdf':
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix.lower() == '.txt':
            loader = TextLoader(str(file_path), encoding="utf-8")
        else:
            print(f"Unsupported File extension: {file_path.suffix}")
            return []
        docs = loader.load()
        print(f"\n Loading :{file_path.name}")
        print(f"Pages/chunks loaded: {len(docs)}")
        return docs
    except Exception as e:
        print(f"Error in loading {file_path.name}: {e}")
        return []



def main() -> None:
    """Main function to load and display document contents."""
    current_dir = Path(__file__).parent
    datadir = current_dir / ".." / "data" 
    
    supported_extensions = {'.pdf', '.txt'}
    
    try:
        my_documents = []
        for file_path in datadir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                docs = load_document(file_path)
                my_documents.extend(docs)
        
        if my_documents:
            print("\n--- FILE CONTENT ---")
            content = my_documents[1].page_content
            print(content)
            print("\n--- METADATA ---")
            print(my_documents[1].metadata)
            print(f"\n--- TOTAL DOCUMENTS LOADED: {len(my_documents)} ---")
        else:
            print("No documents found in the data directory.")

    except Exception as e:
        print(f"Oops! Something went wrong: {e}")


if __name__ == "__main__":
    main()