"""
LangChain RAG Pipeline with Tool-Use (Agent)
Uses 3 tools: date extraction, text summarization, and RAG retrieval
Compatible with both local and Google Colab environments
"""
import os
import sys
from pathlib import Path

# Handle Colab vs local paths
if 'google.colab' in sys.modules:
    # In Colab, detect project directory
    current_dir = Path.cwd()
    if (current_dir / 'generate_cuad.py').exists():
        BASE_DIR = current_dir
    else:
        BASE_DIR = Path('/content')
        print("Warning: Using /content as BASE_DIR. Make sure you're in the project directory.")
    print("Running in Google Colab")
else:
    BASE_DIR = Path(__file__).parent
    print("Running locally")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType


# -----------------------------
# Tool 1: Extract Dates
# -----------------------------

def extract_dates(text: str) -> str:
    """
    Extract all dates from contract text.
    
    Args:
        text: Contract text or clause text
        
    Returns:
        Comma-separated list of dates found
    """
    import re
    
    # Date patterns
    patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
        r'\b\d{1,2}\s+(day|days|month|months|year|years)\b',  # Relative dates
    ]
    
    dates = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        dates.extend(matches)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_dates = []
    for date in dates:
        date_str = date if isinstance(date, str) else ' '.join(date) if isinstance(date, tuple) else str(date)
        if date_str not in seen:
            seen.add(date_str)
            unique_dates.append(date_str)
    
    return ", ".join(unique_dates) if unique_dates else "No dates found"


# -----------------------------
# Tool 2: Summarize Text
# -----------------------------

def summarize_text(text: str, max_sentences: int = 2) -> str:
    """
    Summarize contract text or clauses.
    
    Args:
        text: Text to summarize
        max_sentences: Maximum number of sentences in summary (default: 2)
        
    Returns:
        Summarized text
    """
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    
    prompt = f"""Summarize the following contract text in {max_sentences} sentences. 
Focus on key legal terms, obligations, and important details.

Text:
{text[:2000]}  # Limit input length

Summary:"""
    
    try:
        summary = llm.invoke(prompt).content
        return summary.strip()
    except Exception as e:
        return f"Error summarizing: {str(e)}"


# -----------------------------
# Tool 3: RAG Retrieval
# -----------------------------

# Global vectorstore for RAG retrieval
_vectorstore = None

def rag_retrieve(query: str) -> str:
    """
    Retrieve relevant contract information using RAG (Retrieval-Augmented Generation).
    
    Args:
        query: Question or query about the contract
        
    Returns:
        Relevant contract text chunks retrieved via RAG
    """
    global _vectorstore
    
    if _vectorstore is None:
        raise ValueError("Vector store not initialized. Run setup_pipeline() first.")
    
    # Search for relevant chunks
    docs = _vectorstore.similarity_search(query, k=5)
    
    if not docs:
        return f"No relevant information found for: {query}"
    
    # Combine top chunks
    retrieved_text = "\n\n".join([doc.page_content for doc in docs[:3]])
    return retrieved_text[:2000]  # Limit length


# -----------------------------
# Setup Pipeline
# -----------------------------

def setup_pipeline():
    """Setup the RAG pipeline and tools."""
    global _vectorstore
    
    # Check if vectorstore already exists
    vectorstore_dir = BASE_DIR / "data" / "vectorstores"
    vectorstore_path = vectorstore_dir / "langchain_tool_use_faiss"
    
    embeddings = OpenAIEmbeddings()
    
    if vectorstore_path.exists() and (vectorstore_path / "index.faiss").exists():
        print(f"Loading existing vector store from {vectorstore_path}")
        _vectorstore = FAISS.load_local(
            str(vectorstore_path),
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded")
    else:
        print("Vector store not found. Creating new one...")
        
        # Load corpus (handle both local and Colab paths)
        corpus_dir = BASE_DIR / "data" / "corpus" / "cuad"
        documents = []
        for filename in os.listdir(corpus_dir):
            if filename.endswith(".txt"):
                filepath = corpus_dir / filename
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": filename}
                    ))
        
        print(f"Loaded {len(documents)} documents")
        
        # Chunk documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        chunks = splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        # Create vector store
        _vectorstore = FAISS.from_documents(chunks, embeddings)
        print("Vector store created")
        
        # Save vectorstore
        vectorstore_dir.mkdir(parents=True, exist_ok=True)
        _vectorstore.save_local(str(vectorstore_path))
        print(f"Vector store saved to {vectorstore_path}")
    
    # Create tools
    date_tool = Tool(
        name="extract_dates",
        func=extract_dates,
        description="Extracts dates from contract text. Use when asked about dates, expiration, renewal, deadlines, or time periods."
    )
    
    summarize_tool = Tool(
        name="summarize_text",
        func=lambda x: summarize_text(x, max_sentences=2),
        description="Summarizes contract text or clauses. Use when asked to summarize, condense, or provide a brief overview of contract sections."
    )
    
    rag_tool = Tool(
        name="rag_retrieve",
        func=rag_retrieve,
        description="Retrieves relevant contract information using RAG. Use when asked general questions about contracts, clauses, terms, or need to find relevant information across the contract corpus."
    )
    
    # Create agent
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    agent = initialize_agent(
        tools=[date_tool, summarize_tool, rag_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,  # Set to True for debugging
        return_intermediate_steps=True,
    )
    
    print("Agent created with 3 tools: extract_dates, summarize_text, rag_retrieve")
    
    return agent


if __name__ == "__main__":
    agent = setup_pipeline()
    print("\nPipeline setup complete!")
    print("Run evaluation with: python evaluate_langchain_tool_use.py")

