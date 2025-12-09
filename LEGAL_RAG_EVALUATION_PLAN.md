# Legal RAG & Tool-Use Evaluation Plan
## 3-Day Step-by-Step Implementation Guide

**Objective:** Evaluate LangChain and LlamaIndex on CUAD dataset for RAG pipelines and tool-use tasks in legal domain.

---

## **DAY 1: Setup & RAG Pipeline Implementation** (8-10 hours)

### Step 1: Environment Setup (1 hour)

#### 1.1 Install Dependencies
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows

# Install core packages
pip install langchain langchain-openai langchain-community
pip install llama-index llama-index-vector-stores-faiss
pip install faiss-cpu  # or faiss-gpu if you have CUDA
pip install chromadb
pip install pandas numpy
pip install tiktoken  # for token counting
pip install python-dotenv  # for API keys
```

#### 1.2 Download CUAD Dataset
**Action Items:**
- [ ] Download CUAD from: https://zenodo.org/record/4595826
- [ ] Extract to `./data/cuad/`
- [ ] Verify structure:
  ```
  data/cuad/
  ├── CUAD_v1/
  │   ├── full_contract_txt/  # Contract text files
  │   └── master_clauses.csv   # 41 clause categories with Q/A
  ```

#### 1.3 Setup API Keys
- [ ] Create `.env` file:
  ```
  OPENAI_API_KEY=your_key_here
  ```
- [ ] Or use local LLM (Ollama/Llama.cpp) if preferred

#### 1.4 Create Project Structure
```
legal_rag_eval/
├── data/
│   └── cuad/
├── langchain_rag/
│   ├── pipeline.py
│   ├── retriever.py
│   └── evaluator.py
├── llamaindex_rag/
│   ├── pipeline.py
│   ├── retriever.py
│   └── evaluator.py
├── tools/
│   ├── date_extractor.py
│   ├── clause_summarizer.py
│   └── clause_comparer.py
├── evaluation/
│   ├── metrics.py
│   └── benchmark.py
├── results/
└── requirements.txt
```

**Deliverable:** Working environment with CUAD data loaded

---

### Step 2: Data Preprocessing (1-2 hours)

#### 2.1 Load CUAD Data
**Action Items:**
- [ ] Read `master_clauses.csv` to understand structure
- [ ] Load contract text files
- [ ] Create data loader script: `load_cuad.py`

**Key columns in master_clauses.csv:**
- `Filename`: Contract file name
- `Query`: Question about the contract
- `41 clause columns`: Each contains list of answer quotes

#### 2.2 Chunk Contracts
**Action Items:**
- [ ] Implement chunking strategy:
  - Chunk size: 512-1024 tokens
  - Overlap: 100-200 tokens
  - Preserve sentence boundaries
- [ ] Save chunks with metadata:
  - Contract filename
  - Chunk index
  - Start/end positions

**Code template:**
```python
def chunk_contract(text, chunk_size=512, overlap=100):
    # Implement text chunking
    # Return list of chunks with metadata
    pass
```

**Deliverable:** Chunked contract data ready for indexing

---

### Step 3: LangChain RAG Pipeline (3-4 hours)

#### 3.1 Create Vector Store
**Action Items:**
- [ ] Implement `langchain_rag/retriever.py`:
  ```python
  from langchain.vectorstores import FAISS
  from langchain.embeddings import OpenAIEmbeddings
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  
  # Load and chunk contracts
  # Create embeddings
  # Build FAISS index
  ```

#### 3.2 Build RAG Chain
**Action Items:**
- [ ] Implement `langchain_rag/pipeline.py`:
  ```python
  from langchain.chains import RetrievalQA
  from langchain.llms import OpenAI  # or ChatOpenAI
  from langchain.prompts import PromptTemplate
  
  # Create RetrievalQA chain
  # Customize prompt for legal domain
  # Add retrieval parameters (k=5, etc.)
  ```

**Key components:**
- [ ] Retriever: FAISS with similarity search
- [ ] LLM: GPT-3.5-turbo or GPT-4
- [ ] Prompt: Legal-domain specific template
- [ ] Chain: RetrievalQA with custom prompt

#### 3.3 Test Basic Pipeline
**Action Items:**
- [ ] Test with sample questions from CUAD
- [ ] Verify retrieval returns relevant chunks
- [ ] Check answer quality
- [ ] Log LLM calls and latency

**Deliverable:** Working LangChain RAG pipeline

---

### Step 4: LlamaIndex RAG Pipeline (3-4 hours)

#### 4.1 Create Vector Store Index
**Action Items:**
- [ ] Implement `llamaindex_rag/retriever.py`:
  ```python
  from llama_index import VectorStoreIndex, ServiceContext
  from llama_index.embeddings import OpenAIEmbedding
  from llama_index.node_parser import SimpleNodeParser
  
  # Load documents
  # Create nodes
  # Build VectorStoreIndex
  ```

#### 4.2 Build Query Engine
**Action Items:**
- [ ] Implement `llamaindex_rag/pipeline.py`:
  ```python
  from llama_index import VectorStoreIndex
  from llama_index.query_engine import RetrieverQueryEngine
  
  # Create query engine
  # Configure retriever (similarity_top_k=5)
  # Set response mode
  ```

**Key components:**
- [ ] Index: VectorStoreIndex with OpenAI embeddings
- [ ] Retriever: Top-k similarity search
- [ ] Query Engine: RetrieverQueryEngine
- [ ] Response Mode: "compact" or "tree_summarize"

#### 4.3 Test Basic Pipeline
**Action Items:**
- [ ] Test with same sample questions as LangChain
- [ ] Compare retrieval results
- [ ] Check answer quality
- [ ] Log LLM calls and latency

**Deliverable:** Working LlamaIndex RAG pipeline

---

## **DAY 2: Evaluation & Tool-Use Implementation** (8-10 hours)

### Step 5: Evaluation Framework (2-3 hours)

#### 5.1 Implement Metrics
**Action Items:**
- [ ] Create `evaluation/metrics.py`:
  ```python
  def exact_match(predicted, ground_truth):
      # Exact string match
      pass
  
  def f1_score(predicted, ground_truth):
      # Token-level F1
      pass
  
  def retrieval_recall(retrieved_chunks, gold_chunks, k=5):
      # How many gold chunks in top-k?
      pass
  
  def balanced_accuracy(predicted, ground_truth):
      # For binary classification tasks
      pass
  ```

#### 5.2 Create Evaluation Script
**Action Items:**
- [ ] Implement `evaluation/benchmark.py`:
  ```python
  def evaluate_rag_pipeline(pipeline, test_questions, gold_answers):
      results = {
          'exact_match': [],
          'f1_scores': [],
          'retrieval_recall': [],
          'latency': [],
          'llm_calls': []
      }
      # Run evaluation
      return results
  ```

#### 5.3 Run Baseline Evaluation
**Action Items:**
- [ ] Evaluate LangChain on CUAD test set
- [ ] Evaluate LlamaIndex on CUAD test set
- [ ] Record all metrics
- [ ] Save results to `results/day2_baseline.json`

**Metrics to collect:**
- [ ] Exact Match (EM) score
- [ ] F1 score (token-level)
- [ ] Retrieval Recall@k (k=1,3,5)
- [ ] Average latency per query
- [ ] Number of LLM calls per query
- [ ] Cost estimation (if using paid API)

**Deliverable:** Evaluation results for both frameworks

---

### Step 6: Tool-Use Implementation - LangChain (2-3 hours)

#### 6.1 Create Legal Tools
**Action Items:**
- [ ] Implement `tools/date_extractor.py`:
  ```python
  from langchain.tools import Tool
  import re
  from datetime import datetime
  
  def extract_dates(text: str) -> str:
      # Extract dates from contract text
      # Return formatted dates
      pass
  
  date_tool = Tool(
      name="extract_dates",
      func=extract_dates,
      description="Extracts dates from contract clauses"
  )
  ```

- [ ] Implement `tools/clause_summarizer.py`:
  ```python
  def summarize_clause(text: str) -> str:
      # Use LLM to summarize clause
      pass
  ```

- [ ] Implement `tools/clause_comparer.py`:
  ```python
  def compare_clauses(clause1: str, clause2: str) -> str:
      # Compare two clauses
      pass
  ```

#### 6.2 Build LangChain Agent
**Action Items:**
- [ ] Create agent with tools:
  ```python
  from langchain.agents import initialize_agent, AgentType
  from langchain.agents import Tool
  
  tools = [date_tool, summarize_tool, compare_tool, retrieval_tool]
  agent = initialize_agent(
      tools=tools,
      llm=llm,
      agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
      verbose=True
  )
  ```

#### 6.3 Create Tool-Use Tasks
**Action Items:**
- [ ] Design test tasks:
  - **Task 1:** "Extract the expiration date from this contract"
  - **Task 2:** "Summarize the confidentiality clause"
  - **Task 3:** "Compare termination clauses in contract A and B"
  - **Task 4:** "Check if this contract has a non-compete clause"

- [ ] Implement task runner:
  ```python
  def run_tool_use_task(agent, task, contract_text):
      # Run agent on task
      # Track tool calls, retries, correctness
      return {
          'answer': ...,
          'tool_calls': ...,
          'retries': ...,
          'latency': ...
      }
  ```

**Deliverable:** LangChain agent with legal tools

---

### Step 7: Tool-Use Implementation - LlamaIndex (2-3 hours)

#### 7.1 Create LlamaIndex Tools
**Action Items:**
- [ ] Implement tools using LlamaIndex's tool interface:
  ```python
  from llama_index.tools import FunctionTool
  
  date_tool = FunctionTool.from_defaults(
      fn=extract_dates,
      name="extract_dates"
  )
  ```

#### 7.2 Build LlamaIndex Agent
**Action Items:**
- [ ] Create query engine with tools:
  ```python
  from llama_index.agent import ReActAgent
  from llama_index.llms import OpenAI
  
  tools = [date_tool, summarize_tool, compare_tool]
  agent = ReActAgent.from_tools(
      tools=tools,
      llm=llm,
      verbose=True
  )
  ```

#### 7.3 Test Tool-Use Tasks
**Action Items:**
- [ ] Run same test tasks as LangChain
- [ ] Compare tool-call patterns
- [ ] Measure efficiency metrics

**Deliverable:** LlamaIndex agent with legal tools

---

### Step 8: Tool-Use Evaluation (1-2 hours)

#### 8.1 Evaluate Tool-Use Correctness
**Action Items:**
- [ ] Run both agents on tool-use tasks
- [ ] Measure:
  - Correctness (EM/F1 for extracted info)
  - Tool-call efficiency (number of calls)
  - Retry rate
  - End-to-end latency

#### 8.2 Compare Tool-Use Patterns
**Action Items:**
- [ ] Analyze tool-call sequences
- [ ] Compare reasoning patterns
- [ ] Identify strengths/weaknesses

**Deliverable:** Tool-use evaluation results

---

## **DAY 3: Analysis, Comparison & Documentation** (6-8 hours)

### Step 9: Comprehensive Comparison (2-3 hours)

#### 9.1 RAG Pipeline Comparison
**Action Items:**
- [ ] Create comparison table:

| Metric | LangChain | LlamaIndex | Notes |
|--------|-----------|------------|-------|
| **Retrieval Recall@5** | | | |
| **Exact Match Score** | | | |
| **F1 Score** | | | |
| **Avg Latency (ms)** | | | |
| **LLM Calls per Query** | | | |
| **Code Complexity** | | | |
| **Setup Time** | | | |

#### 9.2 Tool-Use Comparison
**Action Items:**
- [ ] Compare tool-use efficiency:

| Task | LangChain | LlamaIndex |
|------|-----------|------------|
| **Correctness (EM)** | | |
| **Tool Calls** | | |
| **Retries** | | |
| **Latency** | | |
| **Reasoning Quality** | | |

#### 9.3 Code Quality Analysis
**Action Items:**
- [ ] Compare:
  - Lines of code
  - API complexity
  - Debugging ease
  - Extensibility

**Deliverable:** Comprehensive comparison matrix

---

### Step 10: Robustness Testing (1-2 hours)

#### 10.1 Edge Cases
**Action Items:**
- [ ] Test with:
  - Missing context (clause not in contract)
  - Ambiguous queries
  - Multi-hop questions ("Compare termination in A with renewal in B")
  - Very long contracts
  - Contracts with no relevant clauses

#### 10.2 Failure Analysis
**Action Items:**
- [ ] Document failure modes:
  - When does retrieval fail?
  - When do tools fail?
  - How do frameworks handle errors?

**Deliverable:** Robustness test results

---

### Step 11: Write Evaluation Report (2-3 hours)

#### 11.1 Report Structure
**Sections:**
1. **Introduction**
   - Problem statement
   - Frameworks evaluated
   - Dataset (CUAD)

2. **Methodology**
   - RAG pipeline setup
   - Tool-use tasks
   - Evaluation metrics

3. **Results: RAG Pipeline**
   - Retrieval quality
   - Answer accuracy
   - Latency & cost
   - Code comparison

4. **Results: Tool-Use**
   - Correctness
   - Tool-call efficiency
   - Reasoning patterns

5. **Robustness Analysis**
   - Edge cases
   - Failure modes

6. **Discussion**
   - Strengths/weaknesses of each framework
   - When to use which framework
   - Limitations

7. **Conclusion**
   - Best framework for legal RAG
   - Recommendations

**Deliverable:** Complete evaluation report

---

### Step 12: Create Visualizations (1 hour)

#### 12.1 Charts to Create
**Action Items:**
- [ ] Performance comparison charts (latency, accuracy)
- [ ] Retrieval recall curves
- [ ] Tool-call efficiency graphs
- [ ] Architecture diagrams for each framework
- [ ] Code complexity visualization

**Deliverable:** Visual comparison materials

---

## **Evaluation Metrics Summary**

### RAG Pipeline Metrics:
- ✅ **Retrieval Recall@k** (k=1,3,5): How often gold chunks retrieved
- ✅ **Exact Match (EM)**: Exact string match with gold answers
- ✅ **F1 Score**: Token-level F1 between predicted and gold
- ✅ **Balanced Accuracy**: For binary classification tasks
- ✅ **Latency**: End-to-end query time
- ✅ **LLM Calls**: Number of API calls per query
- ✅ **Cost**: Estimated API costs

### Tool-Use Metrics:
- ✅ **Correctness (EM/F1)**: Accuracy of extracted/summarized info
- ✅ **Tool-Call Efficiency**: Number of tool calls per task
- ✅ **Retry Rate**: How often agent retries
- ✅ **Latency**: End-to-end task completion time
- ✅ **Reasoning Quality**: Qualitative assessment

---

## **Quick Reference: CUAD 41 Clause Categories**

Key clauses to test:
- Expiration Date
- Renewal Term
- Governing Law
- Non-Compete
- Exclusivity
- Termination For Convenience
- IP Ownership Assignment
- License Grant
- Audit Rights
- Cap On Liability
- ... (and 31 more)

---

## **Time Management Tips**

- **Day 1:** Focus on getting both RAG pipelines working. Don't optimize yet.
- **Day 2:** Run evaluations, implement tool-use. Collect data systematically.
- **Day 3:** Analyze, document, visualize. Don't implement new features.

**If Running Behind:**
- Prioritize RAG evaluation over tool-use
- Use smaller test set (50-100 questions)
- Skip some robustness tests

**If Ahead:**
- Add more tool-use tasks
- Test with additional frameworks (Haystack, etc.)
- Implement advanced retrieval strategies

---

## **Deliverables Checklist**

- [ ] Both RAG pipelines implemented and working
- [ ] Evaluation metrics implemented
- [ ] Baseline RAG results (LangChain vs LlamaIndex)
- [ ] Tool-use agents implemented
- [ ] Tool-use evaluation results
- [ ] Comparison matrix/table
- [ ] Evaluation report (5-10 pages)
- [ ] Visualizations
- [ ] Code repository with examples
- [ ] Presentation slides (if required)

---

**Start with Step 1 and work through systematically. Good luck! 🚀**

