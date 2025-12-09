"""
Evaluate LangChain Tool-Use (Agent) on CUAD benchmark
Tests agent's ability to use 2 tools: extract_dates, rag_qa
"""
import os
import json
import time
import sys
from typing import List, Dict
from pathlib import Path
# extract_dates will be created locally with retrieval capability
# -------------------------------
# Colab vs local paths
# -------------------------------
if 'google.colab' in sys.modules:
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
from langchain.agents import create_agent
from benchmark_types import Benchmark
from langchain.tools import tool
from langchain_classic.chains import RetrievalQA

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Metrics will not be logged.")

# -------------------------------
# Metrics
# -------------------------------
def f1_score(predicted: str, gold: str) -> float:
    pred_tokens = set(predicted.lower().split())
    gold_tokens = set(gold.lower().split())
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    intersection = pred_tokens & gold_tokens
    if not intersection:
        return 0.0
    precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(intersection) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

# -------------------------------
# Tool-Use Test Cases
# -------------------------------
def create_tool_use_test_cases(benchmark: Benchmark) -> List[Dict]:
    tool_use_cases = []
    
    # Keywords that suggest tool use is needed
    date_keywords = ['date', 'expiration', 'expires', 'renewal', 'deadline', 'period', 'term']
    rag_keywords = ['what', 'how', 'which', 'clause', 'provision', 'term', 'agreement']
    
    corpus_dir = BASE_DIR / "data" / "corpus"
    
    for test in benchmark.tests:
        query_lower = test.query.lower()
        
        needs_dates = any(kw in query_lower for kw in date_keywords)
        needs_rag = any(kw in query_lower for kw in rag_keywords) or not needs_dates
        
        if needs_dates or needs_rag:
            gold_texts = []
            for snippet in test.snippets:
                filepath = corpus_dir / snippet.file_path
                if filepath.exists():
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read()
                        gold_texts.append(text[snippet.span[0]:snippet.span[1]])
            
            expected_tools = []
            if needs_dates:
                expected_tools.append('extract_dates')
            if needs_rag:
                expected_tools.append('rag_qa')  # Updated tool name
            
            tool_use_cases.append({
                'query': test.query,
                'gold_answers': gold_texts,
                'expected_tools': expected_tools,
                'snippets': test.snippets,
            })
    
    return tool_use_cases

# -------------------------------
# Modern RAG Tool
# -------------------------------
def create_rag_tool(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )

    @tool
    def rag_qa(query: str) -> str:  # Wrapped tool
        """
        Retrieves and answers questions using contract documents.
        
        Use this tool for general questions about contracts, clauses, terms, or provisions.
        Do NOT use this for date-specific questions - use extract_dates instead.
        
        Input: Your question about the contract.
        Output: An answer based on relevant contract information.
        """
        try:
            # RetrievalQA.invoke expects {"query": query} format
            result = qa_chain.invoke({"query": query})
            # Extract the answer from the result
            if isinstance(result, dict):
                return result.get("result", str(result))
            return str(result)
        except Exception as e:
            return f"Error retrieving information: {str(e)}"
    
    return rag_qa

# -------------------------------
# Enhanced Extract Dates Tool (with retrieval)
# -------------------------------
def create_extract_dates_tool(vectorstore):
    """Create an extract_dates tool that can work with queries or text."""
    
    @tool
    def extract_dates(query_or_text: str) -> str:
        """
        Extracts dates from contract text. 
        
        IMPORTANT: Use this tool when the query asks about:
        - Expiration dates
        - Renewal terms
        - Deadlines
        - Time periods
        - Dates in general
        
        This tool will:
        1. Retrieve relevant contract text if you provide a query
        2. Extract all dates from that text
        
        Input: Either a query about dates OR contract text containing dates.
        Output: A comma-separated list of all dates found.
        """
        from langchain_tool_use import extract_dates as extract_dates_func
        
        # If it looks like a query (contains question words), retrieve first
        if any(word in query_or_text.lower() for word in ['what', 'when', 'which', 'expiration', 'renewal', 'deadline']):
            # Retrieve relevant text first
            docs = vectorstore.similarity_search(query_or_text, k=3)
            retrieved_text = "\n\n".join([doc.page_content for doc in docs])
            # Then extract dates from retrieved text
            return extract_dates_func(retrieved_text)
        else:
            # Assume it's already text, just extract dates
            return extract_dates_func(query_or_text)
    
    return extract_dates

# -------------------------------
# Load Agent
# -------------------------------
def load_tool_use_agent():
    embeddings = OpenAIEmbeddings()
    vectorstore_path = BASE_DIR / "data" / "vectorstores" / "langchain_faiss"
    vectorstore = FAISS.load_local(
        str(vectorstore_path),
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create tools
    rag_tool = create_rag_tool(vectorstore, llm)
    extract_dates_tool = create_extract_dates_tool(vectorstore)
    
    tools = [extract_dates_tool, rag_tool]
    
    # Create agent with explicit instructions to avoid loops
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""You are a legal analysis assistant. 

When answering questions:
1. For questions about DATES (expiration, renewal, deadlines, terms, periods):
   - First use rag_qa to retrieve relevant contract text
   - Then use extract_dates on the retrieved text to find specific dates
   - Combine the results to provide your answer

2. For other questions:
   - Use rag_qa to retrieve and answer the question

3. Always provide a final answer after using tools. Do not call tools repeatedly."""
    )
    
    return agent

# -------------------------------
# Evaluation Function
# -------------------------------
def evaluate_tool_use(agent, test_cases: List[Dict], max_cases: int = None, recall_k: int = 3) -> Dict:
    if max_cases:
        test_cases = test_cases[:max_cases]
    
    results = {
        "f1_scores": [],
        "latencies": [],
        "tool_efficiency": [],
        "tool_usage_breakdown": {"extract_dates": 0, "rag_qa": 0},  # updated
        "detailed_results": [],
        "recall_at_k": [],
    }
    
    print(f"Evaluating on {len(test_cases)} tool-use test cases...")
    
    for i, test_case in enumerate(test_cases):
        query = test_case['query']
        expected_tools = test_case.get('expected_tools', [])
        print(query)
        print(expected_tools)
        start_time = time.time()
        
        try:
            # Use threading timeout to prevent infinite loops (works on all platforms)
            from threading import Thread
            import queue
            
            result_queue = queue.Queue()
            error_queue = queue.Queue()
            
            def run_agent():
                try:
                    # Modern LangChain agent - use correct invoke format
                    result = agent.invoke({"input": query})
                    result_queue.put(result)
                except Exception as e:
                    error_queue.put(e)
            
            # Run agent in separate thread with timeout
            thread = Thread(target=run_agent)
            thread.daemon = True
            thread.start()
            thread.join(timeout=120)  # 30 second timeout
            
            if thread.is_alive():
                print(f"WARNING: Query timed out after 120 seconds: {query[:100]}")
                latency = time.time() - start_time
                raise TimeoutError("Query timed out after 120 seconds")
            
            if not error_queue.empty():
                raise error_queue.get()
            
            if result_queue.empty():
                raise RuntimeError("Agent returned no result")
            
            result = result_queue.get()
            latency = time.time() - start_time
            
            # Extract answer and tool calls from result
            if isinstance(result, dict):
                answer = result.get("output", str(result))
                # Try to extract tool calls from messages
                messages = result.get("messages", [])
                tools_used = []
                for msg in messages:
                    # Check for tool_calls attribute
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tool_name = tc.get('name', 'unknown') if isinstance(tc, dict) else getattr(tc, 'name', 'unknown')
                            tools_used.append(tool_name)
                    # Also check if message content mentions tool usage
                    if hasattr(msg, 'content') and isinstance(msg.content, str):
                        if 'extract_dates' in msg.content.lower():
                            tools_used.append('extract_dates')
                        if 'rag_qa' in msg.content.lower():
                            tools_used.append('rag_qa')
            else:
                answer = str(result)
                tools_used = []
            
            # Remove duplicates while preserving order
            tools_used = list(dict.fromkeys(tools_used))
            print(f"Tools used: {tools_used}")
            print(f"Answer preview: {answer[:200]}...")
            # Update tool usage breakdown
            for t in tools_used:
                if t in results["tool_usage_breakdown"]:
                    results["tool_usage_breakdown"][t] += 1
            
            gold_texts = test_case['gold_answers']
            
            best_f1 = max(f1_score(answer, gold) for gold in gold_texts)
            
            # Tool efficiency
            expected_set = set(expected_tools)
            tools_used_set = set(tools_used)
            tool_efficiency = len(expected_set & tools_used_set) / len(expected_set) if expected_set else 0.0
            
            # recall@k
            recall_hits = sum(1 for gold in gold_texts if gold.lower() in answer.lower())
            recall = min(1.0, recall_hits / recall_k)
            
            results["f1_scores"].append(best_f1)
            results["latencies"].append(latency)
            results["tool_efficiency"].append(tool_efficiency)
            results["recall_at_k"].append(recall)
            
            results["detailed_results"].append({
                "query": query,
                "answer": answer,
                "gold_answers": gold_texts,
                "f1": best_f1,
                "total_latency": latency,
                "tools_used": tools_used,
                "expected_tools": expected_tools,
                "tool_efficiency": tool_efficiency,
                "recall_at_k": recall,
            })
        except TimeoutError as e:
            latency = time.time() - start_time
            print(f"Timeout on query {i+1}: {e}")
            results["f1_scores"].append(0.0)
            results["latencies"].append(latency)
            results["tool_efficiency"].append(0.0)
            results["recall_at_k"].append(0.0)
            results["detailed_results"].append({
                "query": query,
                "answer": "TIMEOUT",
                "error": str(e),
                "f1": 0.0,
                "total_latency": latency,
            })
        except Exception as e:
            latency = time.time() - start_time
            print(f"Error on query {i+1}: {e}")
            import traceback
            traceback.print_exc()
            results["f1_scores"].append(0.0)
            results["latencies"].append(latency)
            results["tool_efficiency"].append(0.0)
            results["recall_at_k"].append(0.0)
            results["detailed_results"].append({
                "query": query,
                "answer": "ERROR",
                "error": str(e),
                "f1": 0.0,
                "total_latency": latency,
            })
    
    return results

# -------------------------------
# Main Evaluation
# -------------------------------
def main():
    # Load benchmark
    benchmark_path = BASE_DIR / "data" / "benchmarks" / "cuad.json"
    with open(benchmark_path, "r") as f:
        benchmark_data = json.load(f)
    benchmark = Benchmark(**benchmark_data)
    
    # Create tool-use test cases
    tool_use_cases = create_tool_use_test_cases(benchmark)
    
    # Load agent
    agent = load_tool_use_agent()
    
    # Evaluate
    results = evaluate_tool_use(agent, tool_use_cases, max_cases=50)
    
    # Summarize
    avg_f1 = sum(results["f1_scores"]) / len(results["f1_scores"])
    avg_latency = sum(results["latencies"]) / len(results["latencies"])
    avg_tool_efficiency = sum(results["tool_efficiency"]) / len(results["tool_efficiency"])
    avg_recall = sum(results["recall_at_k"]) / len(results["recall_at_k"])
    
    print(f"F1: {avg_f1:.2%}, Latency: {avg_latency:.2f}s, Tool Efficiency: {avg_tool_efficiency:.2%}, Recall@k: {avg_recall:.2%}")

if __name__ == "__main__":
    main()
