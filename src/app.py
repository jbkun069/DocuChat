import chainlit as cl # type: ignore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from config import Config
from load import embeddings, llm, template, create_qa_chain, load_document, chunk_documents
import tempfile
from pathlib import Path

@cl.on_chat_start
async def on_chat_start():
    """Initialize the QA chain and store it in the user's session."""

    if not Config.DB_DIR.exists():
       
        await cl.Message(
            content="❌ Database not found! Please run `load.py` first."
        ).send()
        return

    vectorstore = Chroma(
        persist_directory=str(Config.DB_DIR),
        embedding_function=embeddings
    )
    qa_chain = create_qa_chain(vectorstore, llm, template)

    cl.user_session.set("qa_chain", qa_chain)
    cl.user_session.set("vectorstore", vectorstore)

    await cl.Message(
        content="👋 Hello! I'm your Academic Assistant. Upload a document or ask me anything."
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages."""

    qa_chain = cl.user_session.get("qa_chain")
    vectorstore = cl.user_session.get("vectorstore")

    if message.elements:
        processed_files = []

        for file_element in message.elements:
            try:
                file_path = Path(file_element.path)
                
                if not file_path.exists():
                    await cl.Message(content=f"❌ File not found: {file_element.name}").send()
                    continue

                docs = load_document(file_path)
                if docs:
                    chunks = chunk_documents(docs)
                    vectorstore.add_documents(chunks)
                    # Chroma 0.4.x+ auto-persists, no manual persist needed
                    processed_files.append(file_element.name)
                    print(f"✅ Successfully loaded: {file_element.name}")
                    await cl.Message(content=f"✅ Processed: {file_element.name}").send()
                else:
                    await cl.Message(content=f"⚠️ Could not process: {file_element.name}").send()
            except Exception as e:
                await cl.Message(content=f"❌ Error processing {file_element.name}: {str(e)}").send()
                print(f"Error loading file {file_element.name}: {e}")

        if processed_files:
            # Recreate qa_chain with updated vectorstore
            qa_chain = create_qa_chain(vectorstore, llm, template)
            cl.user_session.set("qa_chain", qa_chain)

        # If the message was ONLY a file with no text question, stop here
        if not message.content.strip():
            return

    response_message = cl.Message(content="")
    await response_message.send()  
    
    if not qa_chain:
        response_message.content = "❌ Error: QA chain not initialized. Database may be corrupted."
        await response_message.update()
        return
    
    try:
        result = await cl.make_async(qa_chain.invoke)(message.content)
        answer = result["answer"]
        sources = result["source_documents"]

        sources_content = ""
        for i, doc in enumerate(sources, 1):
            source_name = doc.metadata.get("source", f"Source {i}")
            sources_content += f"**Chunk {i}** from *{Path(source_name).name}*:\n\n{doc.page_content}\n\n---\n\n"

        if sources:
            response_message.content = f"{answer}\n\n<details><summary>📄 Sources Used</summary>\n\n{sources_content}</details>"
        else:
            response_message.content = answer

        await response_message.update()

    except Exception as e:
        response_message.content = f"❌ Error: {str(e)}"
        await response_message.update()


@cl.on_chat_end
async def on_chat_end():
    print("Session ended.")