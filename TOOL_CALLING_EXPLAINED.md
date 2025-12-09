# Tool Calling for Legal RAG Evaluation

## What is Tool Calling?

**Tool calling** (also called "function calling" or "tool use") is when an LLM acts as an **agent** that can:
1. Decide which tools/functions to use
2. Call those tools with appropriate parameters
3. Use the tool results to answer questions
4. Chain multiple tool calls together

**Key difference from pure RAG:**
- **RAG:** Retrieve → Generate answer directly
- **Tool Calling:** Retrieve → Use tools → Process → Generate answer

---

## Tool Calling vs. RAG Pipeline

### Pure RAG Pipeline
```
Question → Retrieve chunks → LLM generates answer
```

### Tool-Use Pipeline (Agent)
```
Question → Agent decides → Call Tool 1 → Process result → 
  → Call Tool 2 → Combine results → Generate answer
```

**Example:**
- **RAG:** "What is the expiration date?" → Retrieve → Answer: "December 31, 2024"
- **Tool-Use:** "What is the expiration date?" → Agent calls `extract_dates()` → Agent calls `validate_date()` → Answer: "December 31, 2024"

---

## Legal Domain Tool-Use Tasks

Based on your requirements, here are relevant tool-use tasks for legal contracts:

### Task 1: Extract Clause Dates
**Question:** "Extract all dates mentioned in the termination clause"

**Tools needed:**
- `retrieve_clause(clause_type)` - Retrieves specific clause
- `extract_dates(text)` - Extracts dates from text
- `format_dates(dates)` - Formats dates consistently

**Agent flow:**
1. Agent identifies: "termination clause" → calls `retrieve_clause("Termination For Convenience")`
2. Agent gets clause text → calls `extract_dates(clause_text)`
3. Agent formats result → returns answer

---

### Task 2: Summarize Clause
**Question:** "Summarize the confidentiality clause in 2 sentences"

**Tools needed:**
- `retrieve_clause(clause_type)` - Gets the clause
- `summarize_text(text, length)` - Summarizes text

**Agent flow:**
1. Agent calls `retrieve_clause("Non-Disparagement")`
2. Agent calls `summarize_text(clause, max_sentences=2)`
3. Agent returns summary

---

### Task 3: Compare Clauses Across Contracts
**Question:** "Compare the non-compete clauses in contract A and contract B"

**Tools needed:**
- `retrieve_clause_from_contract(contract_id, clause_type)` - Gets clause from specific contract
- `compare_texts(text1, text2)` - Compares two texts
- `highlight_differences(diff_result)` - Highlights differences

**Agent flow:**
1. Agent calls `retrieve_clause_from_contract("contract_A", "Non-Compete")`
2. Agent calls `retrieve_clause_from_contract("contract_B", "Non-Compete")`
3. Agent calls `compare_texts(clause_A, clause_B)`
4. Agent formats comparison → returns answer

---

### Task 4: Check Clause Existence
**Question:** "Does this contract have a non-compete clause?"

**Tools needed:**
- `search_clause(contract_id, clause_type)` - Searches for clause
- `check_existence(result)` - Validates if clause exists

**Agent flow:**
1. Agent calls `search_clause("contract_123", "Non-Compete")`
2. Agent evaluates result → returns Yes/No with evidence

---

### Task 5: Flag Risky Clauses
**Question:** "Flag any risky clauses in this contract"

**Tools needed:**
- `list_all_clauses(contract_id)` - Gets all clauses
- `check_risk_level(clause_text, clause_type)` - Evaluates risk
- `rank_risks(risk_results)` - Ranks by risk level

**Agent flow:**
1. Agent calls `list_all_clauses("contract_123")`
2. For each clause, calls `check_risk_level(clause, type)`
3. Agent calls `rank_risks(all_risks)`
4. Agent returns ranked list

---

## Implementation: LangChain Tool Calling

### Step 1: Define Tools

```python
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

# Tool 1: Extract dates
def extract_dates(text: str) -> str:
    """Extract all dates from contract text.
    
    Args:
        text: Contract text or clause text
        
    Returns:
        Comma-separated list of dates found
    """
    import re
    from datetime import datetime
    
    # Pattern for dates (simplified)
    date_patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',  # Month DD, YYYY
    ]
    
    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        dates.extend(matches)
    
    return ", ".join(dates) if dates else "No dates found"

date_tool = Tool(
    name="extract_dates",
    func=extract_dates,
    description="Extracts dates from contract text. Use this when asked about dates, expiration, renewal, or deadlines."
)

# Tool 2: Retrieve clause
def retrieve_clause(contract_text: str, clause_type: str) -> str:
    """Retrieve a specific clause from contract text.
    
    Args:
        contract_text: Full contract text
        clause_type: Type of clause (e.g., "Non-Compete", "Termination")
        
    Returns:
        The clause text if found, else "Clause not found"
    """
    # Simple keyword-based retrieval (you'd use RAG in practice)
    keywords = {
        "Non-Compete": ["non-compete", "noncompete", "competition"],
        "Termination": ["termination", "terminate", "end"],
        "Confidentiality": ["confidential", "proprietary", "non-disclosure"],
    }
    
    if clause_type not in keywords:
        return "Unknown clause type"
    
    # Search for clause (simplified - use RAG in practice)
    for keyword in keywords[clause_type]:
        if keyword.lower() in contract_text.lower():
            # Extract surrounding text (simplified)
            idx = contract_text.lower().find(keyword.lower())
            start = max(0, idx - 100)
            end = min(len(contract_text), idx + 500)
            return contract_text[start:end]
    
    return "Clause not found"

retrieve_tool = Tool(
    name="retrieve_clause",
    func=lambda x: retrieve_clause(contract_text, x),
    description="Retrieves a specific clause from the contract. Use when asked about specific clause types like 'non-compete', 'termination', 'confidentiality'."
)

# Tool 3: Summarize text
def summarize_clause(text: str, max_sentences: int = 2) -> str:
    """Summarize clause text.
    
    Args:
        text: Clause text to summarize
        max_sentences: Maximum number of sentences in summary
        
    Returns:
        Summarized text
    """
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate
    
    llm = OpenAI(temperature=0)
    prompt = PromptTemplate(
        input_variables=["text", "max_sentences"],
        template="Summarize the following contract clause in {max_sentences} sentences:\n\n{text}\n\nSummary:"
    )
    
    return llm(prompt.format(text=text, max_sentences=max_sentences))

summarize_tool = Tool(
    name="summarize_clause",
    func=summarize_clause,
    description="Summarizes contract clauses. Use when asked to summarize, condense, or provide a brief overview of a clause."
)
```

### Step 2: Create Agent

```python
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)

tools = [date_tool, retrieve_tool, summarize_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Uses ReAct pattern
    verbose=True,  # Shows tool calls
    return_intermediate_steps=True  # Returns tool call history
)

# Use agent
result = agent.run("Extract the expiration date from the termination clause")
```

### Step 3: Track Tool Calls

```python
def run_tool_use_task(agent, question, contract_text):
    """Run agent and track metrics."""
    import time
    
    start_time = time.time()
    
    # Set contract_text in retrieve_tool context
    # (In practice, you'd pass this differently)
    
    result = agent.run(question)
    
    latency = time.time() - start_time
    
    # Extract tool call info from agent's intermediate steps
    tool_calls = []
    if hasattr(agent, 'intermediate_steps'):
        for step in agent.intermediate_steps:
            tool_calls.append({
                'tool': step[0].tool,
                'input': step[0].tool_input,
                'output': step[1]
            })
    
    return {
        'answer': result,
        'tool_calls': tool_calls,
        'num_tool_calls': len(tool_calls),
        'latency': latency
    }
```

---

## Implementation: LlamaIndex Tool Calling

### Step 1: Define Tools

```python
from llama_index.tools import FunctionTool
from llama_index.agent import ReActAgent
from llama_index.llms import OpenAI

# Same functions as above, but wrapped differently
date_tool = FunctionTool.from_defaults(
    fn=extract_dates,
    name="extract_dates",
    description="Extracts dates from contract text"
)

retrieve_tool = FunctionTool.from_defaults(
    fn=lambda x: retrieve_clause(contract_text, x),
    name="retrieve_clause",
    description="Retrieves a specific clause from the contract"
)

summarize_tool = FunctionTool.from_defaults(
    fn=summarize_clause,
    name="summarize_clause",
    description="Summarizes contract clauses"
)
```

### Step 2: Create Agent

```python
from llama_index.agent import ReActAgent

llm = OpenAI(temperature=0)

tools = [date_tool, retrieve_tool, summarize_tool]

agent = ReActAgent.from_tools(
    tools=tools,
    llm=llm,
    verbose=True
)

# Use agent
response = agent.chat("Extract the expiration date from the termination clause")
```

---

## Evaluation Metrics for Tool Calling

### 1. Correctness Metrics
```python
def evaluate_tool_use_correctness(predicted_answer, gold_answer):
    """Evaluate if tool-use answer is correct."""
    # Exact Match
    em = 1 if predicted_answer.strip().lower() == gold_answer.strip().lower() else 0
    
    # F1 Score (token-level)
    pred_tokens = set(predicted_answer.lower().split())
    gold_tokens = set(gold_answer.lower().split())
    
    if len(gold_tokens) == 0:
        f1 = 1.0 if len(pred_tokens) == 0 else 0.0
    else:
        precision = len(pred_tokens & gold_tokens) / len(pred_tokens) if pred_tokens else 0
        recall = len(pred_tokens & gold_tokens) / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'exact_match': em, 'f1': f1}
```

### 2. Tool-Call Efficiency
```python
def evaluate_tool_efficiency(tool_calls, expected_tools):
    """Evaluate how efficiently tools were used."""
    metrics = {
        'num_tool_calls': len(tool_calls),
        'tools_used': [call['tool'] for call in tool_calls],
        'unnecessary_calls': 0,  # Tools called but not needed
        'missing_calls': 0,  # Expected tools not called
        'retry_rate': 0  # How often agent retried
    }
    
    # Check if expected tools were used
    tools_used_set = set(metrics['tools_used'])
    expected_set = set(expected_tools)
    
    metrics['missing_calls'] = len(expected_set - tools_used_set)
    metrics['unnecessary_calls'] = len(tools_used_set - expected_set)
    
    return metrics
```

### 3. Latency & Cost
```python
def evaluate_performance(latency, tool_calls, llm_calls):
    """Evaluate performance metrics."""
    return {
        'total_latency': latency,
        'avg_tool_latency': latency / len(tool_calls) if tool_calls else 0,
        'num_llm_calls': llm_calls,
        'num_tool_calls': len(tool_calls),
        'estimated_cost': estimate_cost(llm_calls)  # Based on token usage
    }
```

---

## Tool-Use Test Cases for CUAD

### Test Case 1: Date Extraction
```python
test_cases = [
    {
        'task': 'extract_dates',
        'question': 'Extract all dates mentioned in the expiration date clause',
        'expected_tools': ['retrieve_clause', 'extract_dates'],
        'gold_answer': 'December 31, 2024',  # From CUAD gold data
        'contract_id': 'contract_123'
    }
]
```

### Test Case 2: Clause Summarization
```python
{
    'task': 'summarize_clause',
    'question': 'Summarize the non-compete clause in 2 sentences',
    'expected_tools': ['retrieve_clause', 'summarize_clause'],
    'gold_answer': '...',  # Expected summary
    'contract_id': 'contract_456'
}
```

### Test Case 3: Multi-Step Reasoning
```python
{
    'task': 'compare_clauses',
    'question': 'Compare the termination clauses in contract A and contract B',
    'expected_tools': ['retrieve_clause', 'retrieve_clause', 'compare_texts'],
    'gold_answer': '...',
    'contract_ids': ['contract_A', 'contract_B']
}
```

---

## Comparison: Tool-Use vs. RAG

| Aspect | Pure RAG | Tool-Use (Agent) |
|--------|----------|------------------|
| **Complexity** | Simple | More complex |
| **Flexibility** | Limited | High (can chain tools) |
| **Latency** | Lower (1-2 LLM calls) | Higher (multiple tool + LLM calls) |
| **Cost** | Lower | Higher (more LLM calls) |
| **Use Case** | Direct Q&A | Multi-step reasoning, tool integration |
| **Best For** | Simple retrieval | Complex tasks requiring tools |

---

## When to Use Tool Calling?

**Use tool calling when:**
- ✅ Need to call external APIs (date parsing, validation)
- ✅ Multi-step reasoning required
- ✅ Need to combine multiple data sources
- ✅ Tasks require specific tools (calculators, validators)

**Use pure RAG when:**
- ✅ Simple Q&A from documents
- ✅ Need fast, low-cost answers
- ✅ Direct retrieval is sufficient

---

## For Your Evaluation

**Day 2 Tasks:**
1. Implement 3-5 legal tools (date extraction, clause retrieval, summarization, comparison)
2. Create agents in both LangChain and LlamaIndex
3. Run test cases and measure:
   - Correctness (EM, F1)
   - Tool-call efficiency
   - Latency
   - Cost

**Compare:**
- Which framework makes better tool-call decisions?
- Which has lower latency?
- Which is easier to debug?

This gives you a complete picture: **RAG performance** + **Tool-use capabilities**!

