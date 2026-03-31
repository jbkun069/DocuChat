import sys
from pathlib import Path

from langchain_community.vectorstores import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from config import Config
from langchain_google_genai import ChatGoogleGenerativeAI #type:ignore  
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

llm = ChatGoogleGenerativeAI(
    model=Config.LLM_MODEL,
    google_api_key=Config.GEMINI_API_KEY,
    temperature=0.3,
)

template = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert academic assistant. Your primary goal is to answer using the provided context. "
     "If the provided context contains the answer, use it to respond comprehensively. "
     "If the context does NOT contain the answer, you may rely on your general knowledge, BUT you must start your answer by stating: 'I couldn't find this in the provided documents, but based on my general knowledge...'."
    ),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

def load_document(file_path: Path) -> list:
    if file_path.suffix.lower() == ".pdf":
        return PyPDFLoader(str(file_path)).load()
    elif file_path.suffix.lower() == ".txt":
        return TextLoader(str(file_path), encoding="utf-8").load()
    return []

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return splitter.split_documents(documents)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_qa_chain(vectorstore, llm, prompt_template):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    
    answer_chain = (
        {"context": lambda x: format_docs(x["source_documents"]), "question": lambda x: x["question"]}
        | prompt_template | llm | StrOutputParser()
    )
    
    return RunnableParallel(
        question=RunnablePassthrough(),
        source_documents=retriever,
    ).assign(answer=answer_chain)
    

def main():
    Config.DB_DIR.mkdir(parents=True, exist_ok=True)

    if any(Config.DB_DIR.iterdir()):
        print("✅ Database exists. To refresh it, delete the 'vector_db' folder and run again.")
        return

    print("🚀 Creating new database...")
    datadir = Config.DATA_DIR
    documents = []

    for file_path in datadir.iterdir():
        if file_path.suffix.lower() in {".pdf", ".txt"}:
            documents.extend(load_document(file_path))

    if documents:
        chunks = chunk_documents(documents)
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(Config.DB_DIR),
        )
        print("✨ Database successfully created!")
    else:
        print("⚠️ No documents found in the data folder.")

if __name__ == "__main__":
    main()