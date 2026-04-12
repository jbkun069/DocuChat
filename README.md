# 📚 Document Q&A Chatbot (RAG-based)

A Retrieval-Augmented Generation (RAG) chatbot that allows users to query their own documents using natural language. The system combines semantic search with a large language model to generate accurate, context-aware answers grounded in user-provided data.

---

## 🚀 Overview

This project implements a complete pipeline for document-based question answering:

- Documents are ingested and split into manageable chunks  
- Chunks are converted into embeddings using a local model  
- Stored in a vector database for semantic retrieval  
- A language model generates answers using retrieved context  

The chatbot provides precise answers along with references to the original sources.

---

## ✨ Features

- 📄 **Multi-format Support**  
  Supports PDF and TXT documents  

- 💬 **Natural Language Queries**  
  Ask questions conversationally  

- 🔍 **Source Attribution**  
  Answers include relevant document excerpts and metadata  

- 🧠 **RAG Architecture**  
  Combines vector search (Chroma) with LLM reasoning (Gemini)  

- 🌐 **Web Interface (Streamlit)**  
  Interactive UI with chat history and session management  

- 💻 **Command-Line Interface (CLI)**  
  Lightweight terminal-based interaction  

- ⚡ **Optimized Embeddings**  
  ONNX-accelerated HuggingFace embeddings for efficient inference  

---

## 🏗️ Architecture

User Query
↓
Retriever (Chroma Vector DB)
↓
Relevant Document Chunks
↓
Prompt Construction
↓
LLM (Google Gemini)
↓
Final Answer + Sources


---

## 🧩 Tech Stack

- **LLM:** Google Gemini  
- **Embeddings:** HuggingFace (ONNX optimized)  
- **Vector Store:** ChromaDB  
- **Framework:** LangChain  
- **Frontend:** Streamlit  
- **CLI:** Python  

---

## 📦 Installation

```bash
git clone <your-repo-url>
cd <repo-name>

pip install -r requirements.txt

## Create a .env file:
GEMINI_API_KEY=your_api_key_here

⚙️ Usage

1. Load Documents
   python load.py

2. Run Web App
   streamlit run app.py

3. Run CLI
   python cli.py

🧠 Example Workflow

Upload a document
System processes and indexes the content
Ask a question
Receive:
Generated answer
Supporting excerpts
Source metadata