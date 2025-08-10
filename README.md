# rag_api

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![CI](https://github.com/FlorentB974/rag_api/actions/workflows/ci.yml/badge.svg)
![Issues](https://img.shields.io/github/issues/FlorentB974/rag_api)
![PRs](https://img.shields.io/github/issues-pr/FlorentB974/rag_api)

A Python-based Retrieval-Augmented Generation (RAG) API for querying personal documents using a vector database (ChromaDB) and integrating with a UI like LibreChat.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Code of Conduct](#code-of-conduct)
- [Support](#support)

## Overview
The `rag_api` project enables users to ingest personal documents into a vector database (ChromaDB) and query them using a natural language interface. It integrates with tools like LibreChat for a user-friendly experience and leverages models from `ollama` for processing queries.

## Features
- Ingest and index documents into a ChromaDB vector database.
- Query documents using natural language via a Python script or LibreChat UI.
- Support for multiple document formats (PDF, text, etc.) via `unstructured`.
- Easy integration with `ollama` for language model inference.
- Dockerized deployment for the API endpoint.

## Prerequisites
- **Docker**: Required for running the API endpoint.
- **Python 3.8+**: Required for running the Python scripts.
- **Ollama**: Required for language model inference.
- A directory containing documents to be indexed.

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/FlorentB974/rag_api.git
   cd rag_api
   ```

2. **Set Up Python Virtual Environment**
   ```bash
   python -m venv rag_env
   source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install langchain langchain_community langchain_huggingface langchain_chroma langchain_ollama unstructured huggingface_hub chromadb sentence-transformers llama-cpp-python pypdf
   ```

## Usage

### Initialize the Vector Database
To create a new vector database and index documents:
```bash
python vector_db.py --source /path/to/docs --db /path/to/vector_db --init
```

### Add New Documents
To add additional documents to an existing database:
```bash
python vector_db.py --source /path/to/newfile --db /path/to/vector_db
```

### Test Queries
Run the query script to test the setup:
```bash
python query.py
```

### Deploy the API with Docker
Start the API endpoint using Docker Compose:
```bash
docker compose up -d --build
```

### Configure LibreChat
Add the following configuration to your `librechat.yml` file:
```yaml
endpoints:
  - name: "Personal Docs"
    apiKey: "ollama"
    baseURL: "http://host.docker.internal:5500/v1"  # Use endpoint_ip if not using Docker
    models:
      default:
        - "mistral"
      fetch: false
    titleConvo: true
    titleModel: "current_model"
    summarize: true
    summaryModel: "current_model"
    forcePrompt: false
```

After updating the configuration, restart LibreChat to apply the changes.

## Configuration
- **Vector DB Path**: Specify the path for the ChromaDB database using the `--db` flag in `vector_db.py`.
- **Document Source**: Provide the path to your documents using the `--source` flag.
- **API Endpoint**: The default port is `5500`. Update the `baseURL` in `librechat.yml` if you change the port in the Docker configuration.

## Contributing
Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, or suggest improvements.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Code of Conduct
We are committed to fostering an open and inclusive community. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

## Support
If you encounter issues or have questions, please file an issue on the [GitHub Issues page](https://github.com/FlorentB974/rag_api/issues).
