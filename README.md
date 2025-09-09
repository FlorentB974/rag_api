# Advanced RAG API with Intelligent Document Processing

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![CI](https://github.com/FlorentB974/rag_api/actions/workflows/ci.yml/badge.svg)
![Issues](https://img.shields.io/github/issues/FlorentB974/rag_api)
![PRs](https://img.shields.io/github/issues-pr/FlorentB974/rag_api)

A sophisticated Retrieval-Augmented Generation (RAG) system with advanced document ingestion, intelligent processing, and optimized vector storage for querying personal documents.

## Table of Contents

- [Overview](#overview)
- [🚀 New Features](#-new-features)
- [📦 Installation](#-installation)
- [🛠️ Usage](#️-usage)
- [⚙️ Configuration](#️-configuration)
- [📊 Processing Reports](#-processing-reports)
- [🏗️ Architecture](#️-architecture)
- [🔧 Advanced Features](#-advanced-features)
- [🚨 Troubleshooting](#-troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

The `rag_api` project has been completely redesigned with advanced document processing capabilities. It now supports **ANY document type** using intelligent detection, provides **automatic summarization**, and includes comprehensive **metadata enhancement** for optimal RAG performance.

## 🚀 New Features

### Document Processing

- **Universal Document Support**: Automatically detects and processes ANY document type using `libmagic`
- **Intelligent Chunking**: Optimized text splitting with context-aware separators
- **Document Summarization**: Automatic summarization using Ollama models with metadata enhancement
- **Comprehensive Metadata**: Rich document metadata including file info, content statistics, and processing timestamps
- **Deduplication**: Content-based hashing to prevent duplicate processing

### Supported Document Types

- **PDFs**: Native PDF processing with text extraction
- **Microsoft Office**: Word (.docx/.doc), Excel (.xlsx/.xls), PowerPoint (.pptx/.ppt)
- **Text Formats**: Plain text, Markdown, HTML, XML, RTF
- **Data Formats**: CSV, JSON, JSONL
- **Code Files**: Python, JavaScript, TypeScript, Java, C++, CSS, SQL, YAML, etc.
- **Email**: EML, MSG files
- **Auto-Detection**: Uses MIME type detection for unknown extensions

### Vector Database Features

- **ChromaDB Integration**: High-performance vector storage with cosine similarity
- **Ollama Embeddings**: Local embedding generation with configurable models
- **Database Management**: Initialize, update, and clear operations
- **Processing Reports**: Detailed metrics and performance analysis

## 📦 Installation

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
  pip install -r requirements.txt
  ```

4. **Install System Dependencies** (for libmagic)

   ```bash
   # macOS
   brew install libmagic
   
   # Ubuntu/Debian
   sudo apt-get install libmagic1
   
   # Windows - included with python-magic-bin (already in requirements)
   ```

## 🛠️ Usage

### Advanced Document Processing

```bash
# Initialize new database with intelligent document processing
python vector_db.py --source /path/to/documents --db vector_db --init

# Add documents to existing database
python vector_db.py --source /path/to/documents --db vector_db

# Process with custom settings and summarization
python vector_db.py --source /path/to/documents --db vector_db --init \
  --chunk-size 1500 --chunk-overlap 300 \
  --summarize-model llama3.2:3b

# Disable summarization for faster processing
python vector_db.py --source /path/to/documents --db vector_db --init --no-summary

# Generate processing report
python vector_db.py --source /path/to/documents --db vector_db --init \
  --report processing_report.json
```

### Advanced Options

The new system provides extensive configuration options:

```bash
python vector_db.py --help
```

Options include:

- `--chunk-size`: Size of text chunks (default: 1024)
- `--chunk-overlap`: Overlap between chunks (default: 200)
- `--summarize-model`: Ollama model for summarization (default: llama3.2:1b)
- `--no-summary`: Disable document summarization
- `--report`: Path to save processing report

### Direct RAG Utils Usage

```python
from rag_utils import load_and_process_documents, DocumentProcessor

# Process documents with custom settings
documents, metrics = load_and_process_documents(
    source_path="/path/to/documents",
    summarize_model="llama3.2:1b",
    chunk_size=1024,
    chunk_overlap=200,
    enable_summarization=True
)

print(f"Processed {len(documents)} chunks from {metrics.successful_docs} documents")
```

### Test Queries

Run the query script to test the setup:

```bash
python query.py
```

### Deploy the API with Docker

Start the API endpoint using Docker Compose:

```bash
cd librechat_endpoint
docker compose up -d --build
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root (use `.env.example` as template):

```env
# Vector Database Configuration
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
COLLECTION_NAME=my_documents

# Document Processing Configuration
SUMMARIZE_MODEL=llama3.2:1b

# Legacy Configuration (still supported)
VECTOR_DB_PATH=./vector_db
OLLAMA_MODEL=mistral
```

### Ollama Models

Ensure you have Ollama installed and the required models pulled:

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull embedding model
ollama pull sentence-transformers/all-MiniLM-L6-v2

# Pull summarization model
ollama pull llama3.2:1b

# Pull query model
ollama pull mistral
```

## 📊 Processing Reports

The new system generates detailed processing reports with comprehensive metrics:

```json
{
  "processing_timestamp": "2025-09-09T12:00:00",
  "metrics": {
    "total_documents": 50,
    "successful_documents": 48,
    "failed_documents": 2,
    "total_chunks": 1247,
    "processing_time_seconds": 45.67,
    "success_rate": 0.96,
    "average_chunks_per_doc": 25.98
  },
  "configuration": {
    "chunk_size": 1024,
    "chunk_overlap": 200,
    "summarization_enabled": true,
    "summarize_model": "llama3.2:1b"
  }
}
```

## 🏗️ Architecture

### Document Processing Pipeline

1. **File Detection**: Uses `libmagic` for MIME type detection
2. **Loader Selection**: Chooses optimal loader based on file type
3. **Content Extraction**: Extracts text and metadata
4. **Intelligent Chunking**: Context-aware text splitting
5. **Summarization**: Generates concise summaries using Ollama
6. **Metadata Enhancement**: Adds comprehensive metadata
7. **Vector Storage**: Stores in ChromaDB with embeddings

### Key Components

- **`rag_utils.py`**: ✨ NEW - Core document processing and utility functions
- **`vector_db.py`**: 🔄 UPDATED - Vector database management with advanced features
- **`query.py`**: Query interface for RAG operations
- **`librechat_endpoint/`**: API endpoint for LibreChat integration

## 🔧 Advanced Features

### Custom Document Processing

```python
from rag_utils import DocumentProcessor

# Create processor with custom settings
processor = DocumentProcessor(
    summarize_model="llama3.2:3b",
    chunk_size=2048,
    chunk_overlap=400,
    enable_summarization=True
)

# Process single document
documents = processor.process_single_document(Path("document.pdf"))

# Batch process with metrics
documents, metrics = processor.process_documents("/path/to/docs")
```

### Metadata-Rich Documents

Each processed document chunk now includes:

- **File Information**: Size, type, timestamps, MIME type
- **Content Statistics**: Word count, character count, content hash
- **Processing Info**: Chunk index, total chunks, processing timestamp
- **Summarization**: AI-generated summary (if enabled)
- **Deduplication**: Content hash for duplicate detection

### LibreChat Integration

Add the following configuration to your `librechat.yml` file:

```yaml
endpoints:
  - name: "Personal Docs (Advanced)"
    apiKey: "ollama"
    baseURL: "http://host.docker.internal:5500/v1"
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

## 🚨 Troubleshooting

### Common Issues

1. **Missing libmagic**: Install system dependencies

   ```bash
   # macOS
   brew install libmagic
   
   # Ubuntu/Debian
   sudo apt-get install libmagic1
   ```

2. **Ollama Connection**: Ensure Ollama is running

   ```bash
   ollama serve
   ```

3. **Memory Issues**: Reduce chunk size or disable summarization

   ```bash
   python vector_db.py --source docs --db vector_db --init --chunk-size 512 --no-summary
   ```

4. **Import Errors**: Ensure all dependencies are installed

   ```bash
   pip install -r requirements.txt
   ```

### Performance Optimization

**For Large Document Collections:**

- Use `--no-summary` for faster processing
- Increase `--chunk-size` to reduce total chunks
- Use lighter embedding models

**For Better Retrieval Quality:**

- Enable summarization with better models
- Use smaller chunk sizes (512-1024)
- Increase chunk overlap (200-400)

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, or suggest improvements.

The new architecture makes it easy to:

- Add support for new document types
- Customize processing pipelines
- Integrate additional AI models
- Extend metadata extraction

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Support

If you encounter issues or have questions, please file an issue on the [GitHub Issues page](https://github.com/FlorentB974/rag_api/issues).

For the new advanced features, check the processing reports and logs for detailed debugging information.
