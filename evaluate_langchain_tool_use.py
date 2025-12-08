"""
Evaluate LangChain Tool-Use (Agent) on CUAD benchmark
Tests agent's ability to use 3 tools: extract_dates, summarize_text, rag_retrieve
"""
import os
import json
import time
import sys
from typing import List, Dict
from pathlib import Path

# Handle Colab vs local paths
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
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from benchmark_types import Benchmark

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Metrics will not be logged.")


# -----------------------------
# Tool Definitions (same as langchain_tool_use.py)
# -----------------------------

def extract_dates(text: str) -> str:
    """Extract dates from contract text."""
    import re
    patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{4}-\d{2}-\d{2}\b',
    ]
    dates = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        dates.extend(matches)
    seen = set()
    unique_dates = [d if isinstance(d, str) else ' '.join(d) if isinstance(d, tuple) else str(d) 
                    for d in dates if (d_str := (d if isinstance(d, str) else ' '.join(d) if isinstance(d, tuple) else str(d))) not in seen and not seen.add(d_str)]
    return ", ".join(unique_dates) if unique_dates else "No dates found"


def summarize_text(text: str, max_sentences: int = 2) -> str:
    """Summarize contract text."""
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    prompt = f"""Summarize the following contract text in {max_sentences} sentences. 
Focus on key legal terms, obligations, and important details.

Text:
{text[:2000]}

Summary:"""
    
    try:
        summary = llm.invoke(prompt).content
        return summary.strip()
    except Exception as e:
        return f"Error summarizing: {str(e)}"


_vectorstore = None

def rag_retrieve(query: str) -> str:
    """Retrieve relevant contract information using RAG."""
    global _vectorstore
    if _vectorstore is None:
        raise ValueError("Vector store not initialized")
    docs = _vectorstore.similarity_search(query, k=5)
    if not docs:
        return f"No relevant information found for: {query}"
    retrieved_text = "\n\n".join([doc.page_content for doc in docs[:3]])
    return retrieved_text[:2000]


# -----------------------------
# Evaluation Metrics
# -----------------------------

def f1_score(predicted: str, gold: str) -> float:
    """Calculate token-level F1 score."""
    pred_tokens = set(predicted.lower().split())
    gold_tokens = set(gold.lower().split())
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    intersection = pred_tokens & gold_tokens
    if not intersection:
        return 0.0
    precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(intersection) / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def count_tool_calls(intermediate_steps) -> Dict:
    """Count tool calls from agent's intermediate steps."""
    tool_calls = []
    for step in intermediate_steps:
        if len(step) >= 2:
            tool_name = step[0].tool if hasattr(step[0], 'tool') else "unknown"
            tool_input = step[0].tool_input if hasattr(step[0], 'tool_input') else ""
            tool_calls.append({
                'tool': tool_name,
                'input': str(tool_input)[:200],
            })
    
    tool_counts = {}
    for call in tool_calls:
        tool_name = call['tool']
        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
    
    return {
        'total_calls': len(tool_calls),
        'tool_counts': tool_counts,
        'tool_calls': tool_calls
    }


# -----------------------------
# Tool-Use Test Cases
# -----------------------------

def create_tool_use_test_cases(benchmark: Benchmark) -> List[Dict]:
    """Create test cases that require tool use."""
    tool_use_cases = []
    
    # Keywords that suggest tool use is needed
    date_keywords = ['date', 'expiration', 'expires', 'renewal', 'deadline', 'period', 'term']
    summarize_keywords = ['summarize', 'summary', 'brief', 'overview', 'condense']
    rag_keywords = ['what', 'how', 'which', 'clause', 'provision', 'term', 'agreement']
    
    corpus_dir = BASE_DIR / "data" / "corpus"
    
    for test in benchmark.tests:
        query_lower = test.query.lower()
        
        # Check which tools might be needed
        needs_dates = any(kw in query_lower for kw in date_keywords)
        needs_summarize = any(kw in query_lower for kw in summarize_keywords)
        needs_rag = any(kw in query_lower for kw in rag_keywords) or not (needs_dates or needs_summarize)
        
        if needs_dates or needs_summarize or needs_rag:
            # Get gold answer
            gold_texts = []
            for snippet in test.snippets:
                filepath = corpus_dir / snippet.file_path
                if filepath.exists():
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read()
                        gold_text = text[snippet.span[0]:snippet.span[1]]
                        gold_texts.append(gold_text)
            
            expected_tools = []
            if needs_dates:
                expected_tools.append('extract_dates')
            if needs_summarize:
                expected_tools.append('summarize_text')
            if needs_rag:
                expected_tools.append('rag_retrieve')
            
            tool_use_cases.append({
                'query': test.query,
                'gold_answers': gold_texts,
                'expected_tools': expected_tools,
                'snippets': test.snippets,
            })
    
    return tool_use_cases


# -----------------------------
# Load Pipeline
# -----------------------------

def load_tool_use_agent():
    """Load the tool-use agent."""
    global _vectorstore
    
    # Load vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore_path = BASE_DIR / "data" / "vectorstores" / "langchain_tool_use_faiss"
    _vectorstore = FAISS.load_local(
        str(vectorstore_path),
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Create tools
    date_tool = Tool(
        name="extract_dates",
        func=extract_dates,
        description="Extracts dates from contract text. Use when asked about dates, expiration, renewal, deadlines."
    )
    
    summarize_tool = Tool(
        name="summarize_text",
        func=lambda x: summarize_text(x, max_sentences=2),
        description="Summarizes contract text or clauses. Use when asked to summarize, condense, or provide a brief overview."
    )
    
    rag_tool = Tool(
        name="rag_retrieve",
        func=rag_retrieve,
        description="Retrieves relevant contract information using RAG. Use when asked general questions about contracts, clauses, terms, or need to find relevant information."
    )
    
    # Create agent
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    agent = initialize_agent(
        tools=[date_tool, summarize_tool, rag_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        return_intermediate_steps=True,
    )
    
    return agent


# -----------------------------
# Evaluation Function
# -----------------------------

def evaluate_tool_use(agent, test_cases: List[Dict], max_cases: int = None) -> Dict:
    """Evaluate tool-use agent on test cases."""
    if max_cases:
        test_cases = test_cases[:max_cases]
    
    results = {
        "f1_scores": [],
        "latencies": [],
        "tool_call_counts": [],
        "tool_efficiency": [],
        "tool_usage_breakdown": {"extract_dates": 0, "summarize_text": 0, "rag_retrieve": 0},
        "detailed_results": []
    }
    
    print(f"Evaluating on {len(test_cases)} tool-use test cases...")
    
    for i, test_case in enumerate(test_cases):
        query = test_case['query']
        expected_tools = test_case.get('expected_tools', [])
        
        # Time the query
        start_time = time.time()
        try:
            result = agent.run(query)
            latency = time.time() - start_time
            
            # Extract tool calls
            tool_info = count_tool_calls(agent.intermediate_steps if hasattr(agent, 'intermediate_steps') else [])
            
            # Track tool usage
            for tool_name in tool_info['tool_counts'].keys():
                if tool_name in results["tool_usage_breakdown"]:
                    results["tool_usage_breakdown"][tool_name] += tool_info['tool_counts'][tool_name]
            
            answer = str(result)
            gold_texts = test_case['gold_answers']
            
            # Calculate correctness (F1 only)
            best_f1 = 0.0
            for gold_text in gold_texts:
                f1 = f1_score(answer, gold_text)
                best_f1 = max(best_f1, f1)
            
            # Tool efficiency: did agent use expected tools?
            tools_used = set(tool_info['tool_counts'].keys())
            expected_set = set(expected_tools)
            tool_efficiency = len(expected_set & tools_used) / len(expected_set) if expected_set else 0.0
            
            results["f1_scores"].append(best_f1)
            results["latencies"].append(latency)
            results["tool_call_counts"].append(tool_info['total_calls'])
            results["tool_efficiency"].append(tool_efficiency)
            
            results["detailed_results"].append({
                "query": query,
                "answer": answer,
                "gold_answers": gold_texts,
                "f1": best_f1,
                "total_latency": latency,
                "tool_calls": tool_info['total_calls'],
                "tools_used": list(tools_used),
                "expected_tools": expected_tools,
                "tool_efficiency": tool_efficiency,
            })
            
        except Exception as e:
            error_latency = time.time() - start_time
            print(f"Error on test case {i+1}: {e}")
            results["f1_scores"].append(0.0)
            results["latencies"].append(error_latency)
            results["tool_call_counts"].append(0)
            results["tool_efficiency"].append(0.0)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(test_cases)} cases...")
    
    return results


# -----------------------------
# Main Evaluation
# -----------------------------

def main():
    # Initialize wandb (if available)
    if WANDB_AVAILABLE:
        wandb.init(
            project="legal-rag-evaluation",
            name="langchain-tool-use-cuad",
            config={
                "framework": "LangChain",
                "mode": "tool-use",
                "dataset": "CUAD",
                "model": "gpt-3.5-turbo",
                "num_tools": 3,
                "tools": ["extract_dates", "summarize_text", "rag_retrieve"],
            }
        )
    
    # Load benchmark
    print("Loading benchmark...")
    benchmark_path = BASE_DIR / "data" / "benchmarks" / "cuad.json"
    with open(benchmark_path, "r") as f:
        benchmark_data = json.load(f)
    
    benchmark = Benchmark(**benchmark_data)
    print(f"Loaded benchmark with {len(benchmark.tests)} test cases")
    
    # Create tool-use test cases
    print("\nCreating tool-use test cases...")
    tool_use_cases = create_tool_use_test_cases(benchmark)
    print(f"Found {len(tool_use_cases)} test cases that benefit from tool use")
    
    # Load agent
    print("\nLoading tool-use agent...")
    agent = load_tool_use_agent()
    print("Agent loaded with 3 tools")
    
    # Evaluate
    print("\nStarting evaluation...")
    results = evaluate_tool_use(agent, tool_use_cases, max_cases=30)
    
    # Calculate summary statistics
    avg_f1 = sum(results["f1_scores"]) / len(results["f1_scores"])
    avg_latency = sum(results["latencies"]) / len(results["latencies"])
    avg_tool_calls = sum(results["tool_call_counts"]) / len(results["tool_call_counts"])
    avg_tool_efficiency = sum(results["tool_efficiency"]) / len(results["tool_efficiency"])
    
    # Log to wandb (if available)
    if WANDB_AVAILABLE:
        wandb.log({
            "f1_score": avg_f1,
            "avg_latency": avg_latency,
            "avg_tool_calls": avg_tool_calls,
            "tool_efficiency": avg_tool_efficiency,
            "num_test_cases": len(results["f1_scores"]),
            "tool_usage_extract_dates": results["tool_usage_breakdown"]["extract_dates"],
            "tool_usage_summarize": results["tool_usage_breakdown"]["summarize_text"],
            "tool_usage_rag": results["tool_usage_breakdown"]["rag_retrieve"],
        })
        
        # Log example table
        table_data = []
        for i, detail in enumerate(results["detailed_results"][:10]):
            table_data.append([
                i + 1,
                detail["query"][:80] + "...",
                detail["answer"][:80] + "...",
                detail["f1"],
                detail["tool_calls"],
                ", ".join(detail["tools_used"]),
                detail["tool_efficiency"],
                detail["total_latency"],
            ])
        
        wandb.log({
            "tool_use_examples": wandb.Table(
                columns=["Case", "Query", "Answer", "F1", "Tool Calls", "Tools Used", "Tool Efficiency", "Latency"],
                data=table_data
            )
        })
    
    # Print results
    print("\n" + "="*50)
    print("TOOL-USE EVALUATION RESULTS")
    print("="*50)
    print(f"Test Cases: {len(results['f1_scores'])}")
    print(f"F1 Score: {avg_f1:.2%}")
    print(f"Avg Latency: {avg_latency:.2f}s")
    print(f"Avg Tool Calls: {avg_tool_calls:.1f}")
    print(f"Tool Efficiency: {avg_tool_efficiency:.2%}")
    print(f"\nTool Usage Breakdown:")
    print(f"  extract_dates: {results['tool_usage_breakdown']['extract_dates']} calls")
    print(f"  summarize_text: {results['tool_usage_breakdown']['summarize_text']} calls")
    print(f"  rag_retrieve: {results['tool_usage_breakdown']['rag_retrieve']} calls")
    print("="*50)
    
    if WANDB_AVAILABLE:
        print(f"\nResults logged to wandb: {wandb.run.url}")
        wandb_url = wandb.run.url
    else:
        wandb_url = None
    
    # Save results
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "langchain_tool_use_evaluation.json"
    
    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "num_cases": len(results["f1_scores"]),
                "f1_score": avg_f1,
                "avg_latency": avg_latency,
                "avg_tool_calls": avg_tool_calls,
                "tool_efficiency": avg_tool_efficiency,
                "tool_usage_breakdown": results["tool_usage_breakdown"],
                "wandb_run_url": wandb_url,
            },
            "detailed_results": results["detailed_results"]
        }, f, indent=2)
    
    print(f"Detailed results saved to {output_path}")
    
    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()

