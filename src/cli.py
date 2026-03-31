import sys
from config import Config
from langchain_community.vectorstores import Chroma

from load import embeddings, llm, template, create_qa_chain

def main():
    print("="*30)
    print("🧠 Starting RAG Assistant...")
    print("="*30)
    
    if not Config.DB_DIR.exists() or not any(Config.DB_DIR.iterdir()):
        print("❌ Error: No database found. Please run load.py first.")
        sys.exit(1)
        
    print("[Loading database into memory.....]")
    vectorstore = Chroma(
        persist_directory = str(Config.DB_DIR),
        embedding_function = embeddings,
    )
    
    print("[Initializing AI engine]...")
    qa_chain = create_qa_chain(vectorstore, llm, template)
    
    print("\n✅ Assistant is ready! (Type 'quit' or 'exit' to stop)")
    print("-" * 30)
    
    while True:
        
        user_input = input("\n 🧑 You: ")
        
        if user_input.strip().lower() in ['quit', 'exit']:
            print("👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        print("🤖 Thinking...")
        
        try:
            result = qa_chain.invoke(user_input)
            print(f"\n🤖 Assistant: {result['answer']}")
            
            print("\n📄 Sources:")
            
            for i, doc in enumerate(result["source_documents"],1):
                source = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "N/A")
                print(f"   {i}. {source} (page {page})")
            
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            
if __name__ == "__main__":
    main()