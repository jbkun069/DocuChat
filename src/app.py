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
        # cl.Message sends a message to the chat UI
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


    if message.elements:
        vectorstore = cl.user_session.get("vectorstore")
        processed_files = []

        for file_element in message.elements:
            file_path = Path(file_element.path)

            docs = load_document(file_path)
            if docs:
                chunks = chunk_documents(docs)
                vectorstore.add_documents(chunks)
                processed_files.append(file_element.name)

        if processed_files:
            qa_chain = create_qa_chain(vectorstore, llm, template)
            cl.user_session.set("qa_chain", qa_chain)
            cl.user_session.set("vectorstore", vectorstore)

            await cl.Message(
                content=f"✅ Processed: {', '.join(processed_files)}. You can now ask questions about them!"
            ).send()

        # If the message was ONLY a file with no text question, stop here
        if not message.content.strip():
            return

    response_message = cl.Message(content="")
    await response_message.send()  # sends the empty "bubble" first

    try:
        result = await cl.make_async(qa_chain.invoke)(message.content)
        answer = result["answer"]
        sources = result["source_documents"]

        # Update the message bubble with the actual answer
        response_message.content = answer
        await response_message.update()
        
        source_elements = []
        for i, doc in enumerate(sources, 1):
            source_name = doc.metadata.get("source", f"Source {i}")
            
            source_elements.append(
                cl.Text(
                    name=f"📄 {Path(source_name).name} — chunk {i}",
                    content=f"**File:** {source_name}\n\n{doc.page_content}",
                    display="side"  
                )
            )

        if source_elements:
           
            response_message.elements = source_elements
            await response_message.update()

    except Exception as e:
        response_message.content = f"❌ Error: {str(e)}"
        await response_message.update()


@cl.on_chat_end
async def on_chat_end():
    print("Session ended.")