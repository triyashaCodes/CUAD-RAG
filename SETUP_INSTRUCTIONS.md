# Setup Instructions for generate_cuad.py

## Prerequisites

This script requires the `legalbenchrag` package. You have two options:

---

## Option 1: Install legalbenchrag Package (Recommended)

### Step 1: Install the package
```bash
pip install legalbenchrag
```

Or if it's not on PyPI, install from GitHub:
```bash
pip install git+https://github.com/zeroentropy-ai/legalbenchrag.git
```

### Step 2: Install other dependencies
```bash
pip install pandas asyncio
```

### Step 3: Set up environment variables (if needed)
The `create_title()` function may require an LLM API key. Check the `legalbenchrag` documentation for required environment variables.

### Step 4: Run the script
```bash
python generate_cuad.py
```

---

## Option 2: Create Standalone Version (If package not available)

If you can't install `legalbenchrag`, I can help you create a standalone version that doesn't require the package. This would involve:

1. Implementing `Benchmark`, `QAGroundTruth`, `Snippet` classes
2. Implementing `sort_and_merge_spans()` function
3. Implementing `create_title()` function (or using a simple alternative)
4. Implementing `download_zip()` function

---

## What the Script Does

1. **Downloads CUAD dataset** from Zenodo
2. **Loads master_clauses.csv** with 41 clause categories
3. **Generates contract titles** using LLM (via `create_title()`)
4. **Filters ambiguous contracts** (amendments, multi-part files)
5. **Extracts gold answer spans** from contract text
6. **Creates benchmark JSON** with Q/A pairs

## Output

The script creates:
- `./data/corpus/cuad/` - Contract text files
- `./data/benchmarks/cuad.json` - Benchmark file with Q/A pairs
- `./tmp/cuad_titles.txt` - Generated titles (if WRITE_TITLES=True)

---

## Troubleshooting

### Error: "No module named 'legalbenchrag'"
**Solution:** Install the package (Option 1) or use standalone version (Option 2)

### Error: "OPENAI_API_KEY not found"
**Solution:** The `create_title()` function may need an API key. Set it:
```bash
export OPENAI_API_KEY=your_key_here
```

### Error: "File not found: master_clauses.csv"
**Solution:** The download may have failed. Check `./data/raw_data/cuad/CUAD_v1/` directory.

### Error: "asyncio" import issues
**Solution:** Use Python 3.7+ (asyncio is built-in)

---

## Quick Test

Before running the full script, test if dependencies are installed:

```python
# test_imports.py
try:
    from legalbenchrag.benchmark_types import Benchmark, QAGroundTruth, Snippet
    from legalbenchrag.generate.utils import create_title, download_zip
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Install with: pip install legalbenchrag")
```

Run: `python test_imports.py`

---

## Next Steps

Once the benchmark is generated:
1. Load `./data/benchmarks/cuad.json` in your evaluation script
2. Use it to test your RAG pipelines (LangChain, LlamaIndex)
3. Compare results against gold answers

