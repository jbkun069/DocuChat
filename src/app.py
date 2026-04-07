import streamlit as st  # type: ignore
from config import Config
from langchain_community.vectorstores import Chroma
from load import embeddings, llm, template, create_qa_chain

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

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
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
                
            except Exception as e:
                st.error(f"Error: {e}")