# Google Colab Setup Guide

## Directory Structure in Colab

When you clone the repo in Colab, the structure will be:

```
/content/
└── Scaling-LLM/          # Your cloned repo
    ├── data/
    │   ├── corpus/
    │   │   └── cuad/
    │   ├── benchmarks/
    │   │   └── cuad.json
    │   └── vectorstores/
    ├── results/
    ├── tmp/
    ├── generate_cuad.py
    ├── langchain_rag.py
    ├── evaluate_langchain.py
    └── ...
```

## How the Code Handles Colab

The code automatically detects Colab and adjusts paths:

```python
if 'google.colab' in sys.modules:
    BASE_DIR = Path('/content')  # Colab root
else:
    BASE_DIR = Path(__file__).parent  # Local script directory
```

**Important:** The code assumes your repo is cloned to `/content/Scaling-LLM/` (or whatever you name it).

---

## Step-by-Step Colab Setup

### 1. Clone the Repository

```python
# In Colab cell
!git clone https://github.com/your-username/your-repo-name.git
# Or if it's in a subdirectory:
!cd /content && git clone https://github.com/your-username/your-repo-name.git Scaling-LLM
```

### 2. Navigate to Project Directory

```python
import os
import sys
from pathlib import Path

# Set working directory
project_dir = Path('/content/Scaling-LLM')  # Adjust name if different
os.chdir(project_dir)
sys.path.insert(0, str(project_dir))

print(f"Working directory: {os.getcwd()}")
```

### 3. Install Dependencies

```python
!pip install -r requirements.txt
```

### 4. Set Environment Variables

```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

### 5. Upload/Download Data

**Option A: If data is in the repo (large files):**
- The data should already be there after cloning
- If `.gitignore` excludes it, you'll need to download separately

**Option B: Download CUAD dataset:**
```python
# Run generate_cuad.py to download CUAD
!python generate_cuad.py
```

**Option C: Upload from Google Drive:**
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy data from Drive
!cp -r /content/drive/MyDrive/your-data-folder/data /content/Scaling-LLM/
```

---

## Path Handling in Code

The code uses `BASE_DIR` which adapts automatically:

### Current Implementation:
- **Colab**: `BASE_DIR = Path('/content')`
- **Local**: `BASE_DIR = Path(__file__).parent`

### Issue:
If you clone to `/content/Scaling-LLM/`, the code looks for:
- `/content/data/...` ❌ (wrong - should be `/content/Scaling-LLM/data/...`)

### Solution Options:

**Option 1: Update BASE_DIR after cloning (Recommended)**
```python
# At the start of your Colab notebook
import sys
from pathlib import Path

# Override BASE_DIR for Colab
if 'google.colab' in sys.modules:
    # Set to your actual project directory
    sys.path.insert(0, '/content/Scaling-LLM')
    # You can modify the scripts to use this, or set environment variable
```

**Option 2: Clone to /content and rename**
```bash
!cd /content && git clone <repo-url> Scaling-LLM
# Then the code expects /content/data/... which won't work
```

**Option 3: Modify code to detect project directory (Best)**

---

## Recommended: Update Code for Better Colab Support

The code should detect the actual project directory. Here's what needs to be updated:

```python
# Better Colab detection
if 'google.colab' in sys.modules:
    # Try to find the project directory
    current_dir = Path.cwd()
    if (current_dir / 'generate_cuad.py').exists():
        BASE_DIR = current_dir
    else:
        # Default to /content, user can adjust
        BASE_DIR = Path('/content')
        print("Warning: Using /content as BASE_DIR. Make sure data is there or update BASE_DIR.")
else:
    BASE_DIR = Path(__file__).parent
```

---

## Quick Colab Setup Script

Create this in a Colab cell:

```python
# Colab Setup Script
import os
import sys
from pathlib import Path

# 1. Clone repo (adjust URL)
repo_url = "https://github.com/your-username/your-repo-name.git"
repo_name = "Scaling-LLM"

if not Path(f"/content/{repo_name}").exists():
    !cd /content && git clone {repo_url} {repo_name}

# 2. Set working directory
project_dir = Path(f"/content/{repo_name}")
os.chdir(project_dir)
sys.path.insert(0, str(project_dir))

# 3. Install dependencies
!pip install -q -r requirements.txt

# 4. Set API key (you'll need to set this)
# os.environ["OPENAI_API_KEY"] = "your-key"

# 5. Verify structure
print("Project structure:")
print(f"  Working dir: {os.getcwd()}")
print(f"  Data exists: {(project_dir / 'data').exists()}")
print(f"  Scripts exist: {(project_dir / 'generate_cuad.py').exists()}")

# 6. Update BASE_DIR in scripts (if needed)
# The scripts will auto-detect Colab, but make sure paths are correct
```

---

## Directory Structure After Setup

```
/content/
├── Scaling-LLM/              # Your cloned repo
│   ├── data/
│   │   ├── corpus/
│   │   │   └── cuad/         # Contract text files
│   │   ├── benchmarks/
│   │   │   └── cuad.json     # Benchmark file
│   │   ├── raw_data/         # Raw CUAD data (if downloaded)
│   │   └── vectorstores/     # Saved vector stores
│   ├── results/              # Evaluation results
│   ├── tmp/                  # Temporary files
│   ├── *.py                  # All Python scripts
│   └── requirements.txt
└── (other Colab files)
```

---

## Running Scripts in Colab

After setup, run scripts normally:

```python
# Setup pipeline
!python langchain_rag.py

# Run evaluation
!python evaluate_langchain.py

# Or use Python directly
exec(open('evaluate_langchain.py').read())
```

---

## Common Issues & Solutions

### Issue 1: "File not found: data/benchmarks/cuad.json"
**Solution:** Make sure you've run `generate_cuad.py` or uploaded the data

### Issue 2: "BASE_DIR is /content but data is in /content/Scaling-LLM/data"
**Solution:** Update BASE_DIR in the script or clone repo to `/content` root

### Issue 3: "Module not found: benchmark_types"
**Solution:** Make sure you're in the project directory: `os.chdir('/content/Scaling-LLM')`

### Issue 4: Large files in git
**Solution:** Use Git LFS or download data separately (CUAD is large)

---

## Recommended Workflow

1. **Clone repo** → `/content/Scaling-LLM/`
2. **cd to project** → `os.chdir('/content/Scaling-LLM')`
3. **Install deps** → `!pip install -r requirements.txt`
4. **Set API key** → `os.environ["OPENAI_API_KEY"] = "..."`  
5. **Download data** → `!python generate_cuad.py` (or upload from Drive)
6. **Run scripts** → `!python evaluate_langchain.py`

The code should automatically detect Colab and use the correct paths!

