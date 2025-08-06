# rag_api
Python RAG API 

## Resolve dependencies
```bash
# Create virtual environment (optional)
python -m venv cag_env
source cag_env/bin/activate

# Install core packages
pip install langchain unstructured huggingface_hub chromadb sentence-transformers llama-cpp-python pypdf
```

TBD maybe more

## Initiliase your db
```bash
python vector_db.py --source ./docs/ --db ./vector_db --init
```

## Test with query.py
```bash
python query.py
```