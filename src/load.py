import os
import sys
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from  langchain_chroma import Chroma #type:ignore

if sys.platform == "win32":
    import types
    pwd = types.ModuleType("pwd")
    sys.modules["pwd"] = pwd

from langchain_community.document_loaders import PyPDFLoader, TextLoader

model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name = model_name)

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


def chunk_and_embed(documents: list, chunk_size: int = 500, chunk_overlap: int = 50) -> tuple:
    """Split documents into chunks and generate embeddings for each chunk."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from your documents.")
    
    if not chunks:
        return [], []
    
    chunk_texts = [chunk.page_content for chunk in chunks]
    vectors = embeddings.embed_documents(chunk_texts)
    
    print(f"\n--- First Chunk Preview ---")
    print(chunks[0].page_content[:200])
    print(f"\n--- Embedding Vector Preview (First 5 numbers) ---")
    print(vectors[0][:5])
    
    return chunks, vectors


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
            
            print(f"\n--- TOTAL DOCUMENTS LOADED: {len(my_documents)} ---")
            chunks, vectors = chunk_and_embed(my_documents)
            
            vectorstore = Chroma.from_documents(
                documents = chunks,
                embedding = embeddings
            )
            
            print("In-memory Chroma store created successfully!")
            
            test_str = "Machine Learning Algorithms"
            result = vectorstore.similarity_search_with_score(test_str, k=3)
            
            if result:
                for doc, score in result:
                 print("\n--- RETRIEVED DOCUMENT ---")
                 print(doc.page_content)

                 print("\n--- SIMILARITY SCORE ---")
                 print(score)
            
            # print(f"\nTotal chunks: {len(chunks)}")
            # print(f"Total vectors: {len(vectors)}")
        else:
            print("No documents found in the data directory.")

    except Exception as e:
        print(f"Oops! Something went wrong: {e}")


if __name__ == "__main__":
    main()