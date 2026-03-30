import sys
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from config import Config
from langchain_google_genai import ChatGoogleGenerativeAI # type: ignore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

if sys.platform == "win32":
    import types
    pwd = types.ModuleType("pwd")
    sys.modules["pwd"] = pwd

from langchain_community.document_loaders import PyPDFLoader, TextLoader

model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name = model_name)

llm = ChatGoogleGenerativeAI(
    model=Config.LLM_MODEL,
    google_api_key=Config.GEMINI_API_KEY,
    temperature=0.3  
)

template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert academic assistant. Use the provided context to answer the question comprehensively and in detail. When asked to compare or find overlaps, analyze the text thoroughly. If the context does not contain the answer, explicitly state that you don't know based on the provided files."),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

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
        return docs
    except Exception as e:
        print(f"Error in loading {file_path.name}: {e}")
        return []


def chunk_and_embed(documents: list, chunk_size: int = 1500, chunk_overlap: int = 200) -> tuple:
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
    
    return chunks, vectors

def create_qa_chain(vectorstore, llm, prompt_template):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return chain


def answer_questions(question: str, qa_chain) -> str:
    """
    Takes a user question and returns an AI-generated answer using the QA chain.
    
    Args:
        question: The user's question
        qa_chain: The LCEL QA chain instance
        
    Returns:
        The AI-generated answer or an error message
    """
    print(f"\n[Searching your files for: '{question}']...")
    try:
        result = qa_chain.invoke(question)
        return result
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "I couldn't find any relevant information in the provided documents" 

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
            Config.DB_DIR.mkdir(parents=True, exist_ok=True)

            if Config.DB_DIR.exists() and any(Config.DB_DIR.iterdir()):
                print("Loading existing database from disk...")
                vectorstore = Chroma(
                    persist_directory=str(Config.DB_DIR),
                    embedding_function=embeddings
                )
            else:
                print("Creating new database...")
                chunks, _ = chunk_and_embed(my_documents)
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=str(Config.DB_DIR)
                )
                print(f"Chroma store persisted at: {Config.DB_DIR}")

            qa_chain = create_qa_chain(
                vectorstore=vectorstore,
                llm=llm,
                prompt_template=template
            )

            test_questions = [
                "List the subjects under DA syllabus",
                "List the subjects under CS syllabus",
                "Find any topics overlap(if any) between the subjects",
            ]

            for q in test_questions:
                answer = answer_questions(
                    question=q, 
                    qa_chain=qa_chain
                )
                print("\n🤖 AI ANSWER:")
                print(answer)
                print("-" * 40)

    except Exception as e:
        print(f"Oops! Something went wrong: {e}")


if __name__ == "__main__":
    main()