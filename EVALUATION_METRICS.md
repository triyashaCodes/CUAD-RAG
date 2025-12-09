# Evaluation Metrics Summary

## Overview
We're evaluating **LangChain** orchestration framework on the **CUAD legal dataset** in two modes:
1. **Pure RAG Pipeline** (`evaluate_langchain.py`)
2. **Tool-Use (Agent) Pipeline** (`evaluate_langchain_tool_use.py`)

---

## 1. Pure RAG Evaluation (`evaluate_langchain.py`)

### Metrics Evaluated:

#### ✅ **Answer Correctness**
- **F1 Score** (token-level)
  - Measures token overlap between predicted and gold answers
  - Range: 0.0 to 1.0 (higher is better)
  - Formula: `2 * (precision * recall) / (precision + recall)`

- **Exact Match** (EM)
  - Exact string match (removed from tool-use, kept in RAG for comparison)
  - Binary: 1.0 if exact match, 0.0 otherwise

#### ✅ **Retrieval Quality**
- **Retrieval Recall@5**
  - Measures if gold answer snippets appear in top-5 retrieved chunks
  - Range: 0.0 to 1.0 (higher is better)
  - Formula: `(gold snippets found in top-k) / (total gold snippets)`

#### ✅ **Performance**
- **Latency** (seconds)
  - End-to-end query time
  - Includes: retrieval + LLM generation
  - Lower is better

#### ✅ **Cost/Resource Usage**
- **LLM Calls per Query**
  - Number of API calls made
  - Lower = cheaper

### What Gets Tested:
- All 4,042 CUAD test cases (or subset with `max_cases`)
- Each test has:
  - Query: "Consider the {contract_title}; {clause_question}"
  - Gold answer: Exact text spans from contracts
  - Multiple snippets possible per query

---

## 2. Tool-Use (Agent) Evaluation (`evaluate_langchain_tool_use.py`)

### Metrics Evaluated:

#### ✅ **Answer Correctness**
- **F1 Score** (token-level)
  - Same as RAG evaluation
  - Measures how well agent's final answer matches gold

#### ✅ **Tool Usage Efficiency**
- **Tool Call Count**
  - Number of tool invocations per query
  - Measures: How many times agent calls tools
  - Lower = more efficient (fewer unnecessary calls)

- **Tool Efficiency**
  - Percentage of expected tools actually used
  - Range: 0.0 to 1.0 (higher is better)
  - Formula: `(expected tools used) / (total expected tools)`
  - Example: If query needs `extract_dates` and agent uses it → 100%

#### ✅ **Tool Selection Quality**
- **Which Tools Were Used**
  - Tracks: `extract_dates` vs `retrieve_clause`
  - Analyzes: Did agent choose the right tool for the task?

#### ✅ **Performance**
- **Latency** (seconds)
  - End-to-end time including tool calls
  - Typically higher than pure RAG (more steps)
  - Includes: tool calls + LLM reasoning + final answer generation

### What Gets Tested:
- **Filtered test cases** that benefit from tool use:
  - **Date-related queries**: Should use `extract_dates` tool
    - Keywords: date, expiration, expires, renewal, deadline, period, term
  - **Clause-related queries**: Should use `retrieve_clause` tool
    - Keywords: clause, provision, section, non-compete, termination, confidentiality, governing law, liability, warranty

- **Two Tools Available**:
  1. `extract_dates`: Extracts dates from text using regex
  2. `retrieve_clause`: Retrieves clauses using RAG/vector search

---

## Comparison: RAG vs Tool-Use

| Metric | Pure RAG | Tool-Use |
|--------|----------|----------|
| **F1 Score** | ✅ | ✅ |
| **Exact Match** | ✅ | ❌ (removed) |
| **Retrieval Recall** | ✅ | ❌ (not applicable) |
| **Latency** | ✅ | ✅ |
| **Tool Calls** | ❌ | ✅ |
| **Tool Efficiency** | ❌ | ✅ |
| **LLM Calls** | ✅ | ✅ (implicit) |

---

## What We're Comparing

### Framework Capabilities:
1. **Pure RAG Performance**
   - How well does LangChain retrieve and answer?
   - Baseline for comparison

2. **Tool-Use Performance**
   - Does using tools improve accuracy?
   - Are tools used efficiently?
   - Is tool-use worth the added latency?

### Use Cases:
- **Simple Q&A**: Pure RAG is faster, cheaper
- **Complex Tasks**: Tool-use may be more accurate
  - Date extraction tasks
  - Clause-specific queries

---

## Outputs

### For Each Evaluation:
1. **Summary Statistics** (printed to console)
   - Average scores across all metrics
   - Number of test cases evaluated

2. **Detailed Results** (JSON file)
   - Per-query results
   - All metrics for each test case
   - Query, answer, gold answer, scores

3. **Wandb Logging** (if available)
   - Metrics tracked over time
   - Example tables
   - Histograms of score distributions
   - Comparison across runs

### Files Generated:
- `results/langchain_evaluation.json` - Pure RAG results
- `results/langchain_tool_use_evaluation.json` - Tool-use results

---

## Key Questions Answered

1. ✅ **How accurate is the RAG pipeline?** → F1 Score
2. ✅ **How good is retrieval?** → Recall@5
3. ✅ **How fast is it?** → Latency
4. ✅ **Do tools help?** → Compare F1: RAG vs Tool-Use
5. ✅ **Are tools used efficiently?** → Tool Efficiency
6. ✅ **How many tool calls?** → Tool Call Count
7. ✅ **Which tools are used when?** → Tool Usage Patterns

---

## Next Steps (Future)

When you add **LlamaIndex** evaluation, you'll compare:
- LangChain RAG vs LlamaIndex RAG
- LangChain Tool-Use vs LlamaIndex Tool-Use
- Which framework is better for legal domain?
- Which framework uses tools more efficiently?

