import sys
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from config import Config
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# Fix for Windows pwd issue
if sys.platform == "win32":
    import types
    pwd = types.ModuleType("pwd")
    sys.modules["pwd"] = pwd

# ------------------ SETUP ------------------

model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

llm = ChatGoogleGenerativeAI(
    model=Config.LLM_MODEL,
    google_api_key=Config.GEMINI_API_KEY,
    temperature=0.3,
)

template = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert academic assistant. Use the provided context to answer the question comprehensively and in detail. "
     "When asked to compare or find overlaps, analyze the text thoroughly. "
     "If the context does not contain the answer, explicitly state that you don't know based on the provided files."
    ),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

# ------------------ DOCUMENT LOADING ------------------

def load_document(file_path: Path) -> list:
    try:
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix.lower() == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
        else:
            print(f"Unsupported File extension: {file_path.suffix}")
            return []

        return loader.load()

    except Exception as e:
        print(f"Error loading {file_path.name}: {e}")
        return []

# ------------------ CHUNKING ------------------

def chunk_documents(documents, chunk_size=1500, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks

# ------------------ FORMATTER ------------------

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ------------------ QA CHAIN (LCEL) ------------------

def create_qa_chain(vectorstore, llm, prompt_template):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    answer_chain = (
        {
            "context": lambda x: format_docs(x["source_documents"]),
            "question": lambda x: x["question"],
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    qa_chain = (
        RunnableParallel(
            question=RunnablePassthrough(),
            source_documents=retriever,
        ).assign(answer=answer_chain)
    )

    return qa_chain

# ------------------ QUERY ------------------

def answer_questions(question: str, qa_chain):
    print(f"\n[Searching for: '{question}']...")

    try:
        result = qa_chain.invoke(question)

        print("\n🤖 ANSWER:\n")
        print(result["answer"])

        print("\n📄 SOURCES:\n")
        for i, doc in enumerate(result["source_documents"], 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "N/A")
            print(f"{i}. {source} (page {page})")

        return result["answer"]

    except Exception as e:
        print(f"Error: {e}")
        return "No answer found."

# ------------------ MAIN ------------------

def main():
    current_dir = Path(__file__).parent
    datadir = current_dir / ".." / "data"

    supported_extensions = {".pdf", ".txt"}

    try:
        documents = []

        for file_path in datadir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                documents.extend(load_document(file_path))

        if not documents:
            print("No documents found.")
            return

        print(f"\nLoaded {len(documents)} documents.")

        Config.DB_DIR.mkdir(parents=True, exist_ok=True)

        if any(Config.DB_DIR.iterdir()):
            print("Loading existing DB...")
            vectorstore = Chroma(
                persist_directory=str(Config.DB_DIR),
                embedding_function=embeddings,
            )
        else:
            print("Creating new DB...")
            chunks = chunk_documents(documents)

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=str(Config.DB_DIR),
            )

        qa_chain = create_qa_chain(vectorstore, llm, template)

        questions = [
            "List the subjects under DA syllabus",
            "List the subjects under CS syllabus",
            "Find any topics overlap between them",
        ]

        for q in questions:
            answer_questions(q, qa_chain)
            print("-" * 50)

    except Exception as e:
        print(f"Fatal error: {e}")

# ------------------

if __name__ == "__main__":
    main()