# Use official Python image
FROM python:3.11-slim

# Set environment variables with defaults
ENV VECTOR_DB_PATH=/app/vector_db
ENV COLLECTION_NAME=my_documents
ENV OLLAMA_MODEL=mistral:latest
ENV EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Expose custom port
EXPOSE 5500

# Start the application
CMD ["python", "app.py"]