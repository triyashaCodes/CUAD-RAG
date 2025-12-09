# Understanding CUAD Data Generation

## Overview

The `generate_cuad()` function creates a structured benchmark dataset from raw CUAD data. Here's how it works:

---

## Data Flow

```
Raw CUAD Data
    ↓
master_clauses.csv (contracts × 41 clause columns)
    ↓
Generate Contract Titles (LLM)
    ↓
Filter Ambiguous Contracts
    ↓
Extract Gold Answer Spans (from CSV quotes)
    ↓
Create QAGroundTruth Objects
    ↓
Save Benchmark JSON
```

---

## Step-by-Step Breakdown

### 1. **Download & Load Data**
```python
download_cuad()  # Downloads CUAD_v1.zip
df = pd.read_csv("master_clauses.csv")
```

**What's in `master_clauses.csv`?**
- Each row = one contract
- Columns:
  - `Filename`: Contract file name (e.g., "contract_123.pdf")
  - `Expiration Date`: Python list string like `["The contract expires on...", "..."]`
  - `Non-Compete`: Python list string with quotes
  - ... (41 clause columns total)

**Key insight:** The CSV already contains the **gold answer quotes** for each clause type!

---

### 2. **Generate Contract Titles**
```python
tasks = [(i, create_title(filename, text)) for ...]
titles = await asyncio.gather(*[task[1] for task in tasks])
```

**Purpose:** Creates a descriptive title for each contract (e.g., "Software License Agreement between Company A and Company B")

**Why?** The final query format is:
```
"Consider the {generated_title}; {column_query}"
```

Example: `"Consider the Software License Agreement; What is the expiration date of this contract?"`

---

### 3. **Filter Ambiguous Contracts**
```python
# Skip if filename has "agreement2", "agreement3" but title doesn't say "amendment"
# Skip if filename has "part1" or "part2"
```

**Purpose:** Removes contracts that are ambiguous (amendments, multi-part contracts) to ensure clean evaluation.

---

### 4. **Extract Gold Answer Spans** (Critical Step!)

```python
for column_name, column_query in column_queries.items():
    # Get quotes from CSV
    raw_query_quotes = row[column_name]  # e.g., '["Quote 1", "Quote 2"]'
    quotes = ast.literal_eval(raw_query_quotes)  # Parse to list
    
    # Find where quotes appear in contract text
    spans = []
    for quote in quotes:
        span = extract_quote_span(text, quote)  # Find (start_idx, end_idx)
        if span:
            spans.append(span)
    
    # Merge overlapping/nearby spans
    spans = sort_and_merge_spans(spans, max_bridge_gap_len=1)
```

**What's happening:**
1. **Parse quotes:** CSV stores quotes as Python list strings → parse to actual list
2. **Find spans:** For each quote, find its exact position in the contract text
3. **Merge spans:** If quotes are close together, merge into one span

**Example:**
- CSV has: `["The contract expires on December 31, 2024", "unless renewed"]`
- Contract text: `"...The contract expires on December 31, 2024, unless renewed..."`
- Spans: `[(100, 150), (151, 165)]` → merged to `[(100, 165)]`

---

### 5. **Create QAGroundTruth Objects**

```python
qa_list.append(
    QAGroundTruth(
        query=f"Consider the {generated_title}; {column_query}",
        snippets=[
            Snippet(
                file_path=f"cuad/{filename}",
                span=span,  # (start_idx, end_idx)
            )
            for span in spans
        ],
    )
)
```

**Structure:**
- **Query:** The question (with contract title)
- **Snippets:** List of text spans that contain the answer
  - `file_path`: Which contract file
  - `span`: Character indices `(start, end)` in that file

**Why spans?** This allows evaluation of:
- **Retrieval:** Did you retrieve the right text chunk?
- **Answer extraction:** Does your answer match the gold quote?

---

## Key Data Structures

### QAGroundTruth
```python
{
    "query": "Consider the Software License Agreement; What is the expiration date?",
    "snippets": [
        {
            "file_path": "cuad/contract_123.txt",
            "span": [100, 165]  # Character indices
        }
    ]
}
```

### Benchmark JSON
```json
{
    "tests": [
        {
            "query": "...",
            "snippets": [...]
        },
        ...
    ]
}
```

---

## Why This Design?

### ✅ Advantages:
1. **Exact ground truth:** Quotes from legal experts, not LLM-generated
2. **Span-level evaluation:** Can check if retrieval found the right location
3. **Structured format:** Easy to load and evaluate
4. **Multiple snippets:** Handles answers that span multiple locations

### 📊 Evaluation Use Cases:
- **Retrieval Recall@k:** Did top-k chunks contain the gold span?
- **Exact Match:** Does predicted answer exactly match gold quote?
- **F1 Score:** Token-level overlap between predicted and gold
- **Span Overlap:** How much does retrieved chunk overlap with gold span?

---

## For Your RAG Evaluation

When you load this benchmark:

1. **Load benchmark JSON:**
   ```python
   benchmark = Benchmark.from_json("cuad.json")
   ```

2. **For each test case:**
   - Get the query
   - Get gold snippets (file_path + span)
   - Run your RAG pipeline
   - Compare predicted answer vs. gold quote
   - Check if retrieved chunks contain the gold span

3. **Metrics:**
   - **Retrieval:** Does retrieved chunk contain `span`?
   - **Answer:** Does generated answer match gold quote?

---

## Important Notes

1. **Quotes are exact:** The CSV contains exact quotes from contracts - these are the ground truth answers
2. **Spans are character indices:** `span=(100, 165)` means characters 100-165 in the text file
3. **Multiple spans possible:** One answer can span multiple locations (merged if close)
4. **Filtered dataset:** Only includes contracts that passed ambiguity checks

---

## Example Flow

```
Contract: "contract_123.txt"
Clause: "Expiration Date"
CSV Quote: ["The Agreement shall expire on December 31, 2024"]

↓

Find in text: "The Agreement shall expire on December 31, 2024"
Span: (2450, 2505)

↓

Create QAGroundTruth:
{
    query: "Consider the Software License Agreement; What is the expiration date?",
    snippets: [{
        file_path: "cuad/contract_123.txt",
        span: [2450, 2505]
    }]
}
```

---

This design makes CUAD perfect for RAG evaluation because you have:
- ✅ Exact ground truth answers
- ✅ Exact locations in documents
- ✅ Structured format for automated evaluation

