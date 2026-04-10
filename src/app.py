import json
import streamlit as st  # type: ignore
from langchain_core.documents import Document
from config import Config
from langchain_community.vectorstores import Chroma
from load import embeddings, llm, template, create_qa_chain, load_document, chunk_documents

st.set_page_config(
    page_title="DocuChat",
    page_icon="🧠",
    layout="centered"
)
st.title("📚 Academic Chatbot Assistant")

@st.cache_resource
def init_qa_engine():
    if not Config.DB_DIR.exists():
        st.error("Database not found! Run load.py first.")
        st.stop()
    
    vectorstore = Chroma(
        persist_directory=str(Config.DB_DIR),
        embedding_function=embeddings
    )
    return create_qa_chain(vectorstore, llm, template)

qa_chain = init_qa_engine()

chat_history_path = Config.DATA_DIR / "chat_history.json"

def load_chat_history():
    if chat_history_path.exists():
        with open(chat_history_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                for msg in data:
                    if "sources" in msg:
                        msg["sources"] = [
                            Document(page_content=s.get("page_content", ""), metadata=s.get("metadata", {})) 
                            for s in msg["sources"]
                        ]
                return data
            except Exception:
                return []
    return []

def save_chat_history(messages):
    Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = []
    for msg in messages:
        msg_copy = {"role": msg["role"], "content": msg["content"]}
        if "sources" in msg:
            msg_copy["sources"] = [
                {"page_content": s.page_content, "metadata": s.metadata} 
                for s in msg["sources"]
            ]
        data.append(msg_copy)
    with open(chat_history_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

with st.sidebar:
    st.title("🧠 DocuChat")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        if chat_history_path.exists():
            chat_history_path.unlink()
        st.rerun()

uploaded_file = st.file_uploader("Upload a PDF, TXT or docx file", type=["pdf", "txt", "docx"])
if uploaded_file is not None:
    Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_path = Config.DATA_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Processing document..."):
        docs = load_document(file_path)
        if docs:
            chunks = chunk_documents(docs)
            vectorstore = Chroma(
                persist_directory=str(Config.DB_DIR),
                embedding_function=embeddings
            )
            vectorstore.add_documents(chunks)
            st.success(f"Successfully processed {uploaded_file.name}!")
            init_qa_engine.clear()

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

messages = st.session_state.messages

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if "sources" in message:
            with st.expander("🔍 View Sources & Metadata"):
                for i, doc in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}: {doc.metadata.get('source', 'Unknown File')}**")
                    st.write(doc.page_content)
                    st.json(doc.metadata)
                    st.markdown("---") 


if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_chat_history(st.session_state.messages)
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents..."):
            try:
                result = qa_chain.invoke(prompt)
                answer = result["answer"]
                sources = result["source_documents"]
                
                st.markdown(answer)
                
                with st.expander("🔍 View Sources & Metadata"):
                    for i, doc in enumerate(sources, 1):
                        st.markdown(f"**Source {i}: {doc.metadata.get('source', 'Unknown File')}**")
                        st.write(doc.page_content) 
                        st.json(doc.metadata)      
                        st.markdown("---")
                        
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources 
                })
                save_chat_history(st.session_state.messages)
                
            except Exception as e:
                st.error(f"Error: {e}")