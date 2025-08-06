# rag_api
Python RAG API 

## Resolve dependencies
```bash
# Create virtual environment (optional)
python -m venv rag_env
source rag_env/bin/activate

# Install core packages
pip install langchain unstructured huggingface_hub chromadb sentence-transformers llama-cpp-python pypdf
```

TBD maybe more

## Vector DB

### Initiliase your db
```bash
python vector_db.py --source /path/to/docs --db /path/to/vector_db --init
```

### Add new files into db (optional)
```bash
python vector_db.py --source /path/to/newfile --db path/to/vector_db
```

After this, you will need to restart the docker compose, see command below.

## Test with query.py
```bash
python query.py
```

## Librechat endpoint
```bash
docker compose up -d --build
```

### Add in your librechat.yml:
```yaml
.....
    - name: "Personal Docs"
      apiKey: "ollama"
      # use 'host.docker.internal' instead of localhost if running LibreChat in a docker container
      baseURL: "http://<endpoint_ip>:5500/v1"
      models:
        default: [
          "mistral"
          ]
        fetch: false
      titleConvo: true
      titleModel: "current_model"
      summarize: true
      summaryModel: "current_model"
      forcePrompt: false
.....
```