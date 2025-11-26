"""
LangChain RAG Pipeline for D&D Documents

A 2-step RAG pipeline that answers questions about D&D rules
using markdown documents from the PDFs folder.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Constants
PDFS_DIR = Path(__file__).parent / "PDFs"
CHROMA_PERSIST_DIR = Path(__file__).parent / ".chroma_db"


def load_documents() -> list:
    """Load all markdown documents from the PDFs directory."""
    loader = DirectoryLoader(
        str(PDFS_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents


def split_documents(documents: list) -> list:
    """Split documents into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def create_vectorstore(chunks: list) -> Chroma:
    """Create or load a Chroma vector store from document chunks."""
    embeddings = OpenAIEmbeddings()

    # Check if vector store already exists
    if CHROMA_PERSIST_DIR.exists():
        print("Loading existing vector store...")
        vectorstore = Chroma(
            persist_directory=str(CHROMA_PERSIST_DIR),
            embedding_function=embeddings,
        )
    else:
        print("Creating new vector store...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(CHROMA_PERSIST_DIR),
        )

    return vectorstore


def create_rag_chain(vectorstore: Chroma) -> RetrievalQA:
    """Create a RAG chain for question answering."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=False,
    )

    return rag_chain


def format_response(result: dict) -> str:
    """Format the RAG response with sources."""
    answer = result["result"]
    sources = result.get("source_documents", [])

    output = f"\n{'='*60}\n"
    output += f"ANSWER:\n{answer}\n"
    output += f"\n{'='*60}\n"
    output += "SOURCES:\n"

    seen_sources = set()
    for doc in sources:
        source = doc.metadata.get("source", "Unknown")
        if source not in seen_sources:
            seen_sources.add(source)
            output += f"  - {Path(source).name}\n"

    return output


def main():
    """Main function to run the RAG pipeline."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please create a .env file with your OpenAI API key:")
        print("  OPENAI_API_KEY=sk-...")
        return

    print("🐉 D&D Rules RAG Pipeline\n")

    # Build the knowledge base
    print("📚 Loading documents...")
    documents = load_documents()

    print("✂️  Splitting documents...")
    chunks = split_documents(documents)

    print("🔢 Creating vector store...")
    vectorstore = create_vectorstore(chunks)

    print("⛓️  Building RAG chain...")
    rag_chain = create_rag_chain(vectorstore)

    print("\n✅ RAG pipeline ready!")
    print("Type 'quit' or 'exit' to stop.\n")

    # Interactive query loop
    while True:
        try:
            question = input("\n🎲 Ask a question about D&D: ").strip()

            if not question:
                continue

            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye! May your rolls be ever in your favor! 🎲")
                break

            print("\n🔍 Searching...")
            result = rag_chain.invoke({"query": question})
            print(format_response(result))

        except KeyboardInterrupt:
            print("\n\nGoodbye! May your rolls be ever in your favor! 🎲")
            break


if __name__ == "__main__":
    main()
