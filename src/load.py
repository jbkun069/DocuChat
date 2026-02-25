import os
import sys
from pathlib import Path

if sys.platform == "win32":
    import types
    pwd = types.ModuleType("pwd")
    sys.modules["pwd"] = pwd

from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_text_document(file_path: Path) -> None:
    """Load and display text document content and metadata."""
    loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()
    print(docs[0])


def load_pdf_document(file_path: Path) -> None:
    """Load and display PDF document content and metadata."""
    Pdfloader = PyPDFLoader(str(file_path))
    pdf_docs = Pdfloader.load()
    print("\n--- PDF CONTENT ---")
    print(pdf_docs[0])
    print(Pdfloader.lazy_load())


def main() -> None:
    """Main function to load and display document contents."""
    current_dir = Path(__file__).parent
    file_path = current_dir / ".." / "data" / "AI.txt"
    pdf_path = current_dir / ".." / "data" / "DA_2026_Syllabus.pdf"

    try:
        load_text_document(file_path)
        load_pdf_document(pdf_path)

        my_documents = TextLoader(str(file_path), encoding="utf-8").load()
        content = my_documents[0].page_content
        print("--- FILE CONTENT ---")
        print(content)
        print("\n--- METADATA ---")
        print(my_documents[0].metadata)

    except Exception as e:
        print(f"Oops! Something went wrong: {e}")


if __name__ == "__main__":
    main()