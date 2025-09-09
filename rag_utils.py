"""
Advanced RAG document ingestion utilities with intelligent processing and summarization.
Supports any document type using libmagic detection and provides optimized chunking strategies.
"""

import os
import json
import hashlib
import magic
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    JSONLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredXMLLoader,
    UnstructuredEmailLoader,
    UnstructuredRTFLoader
)
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


@dataclass
class ProcessingMetrics:
    """Metrics for document processing performance"""
    total_docs: int = 0
    successful_docs: int = 0
    failed_docs: int = 0
    total_chunks: int = 0
    processing_time: float = 0.0


class DocumentProcessor:
    """Advanced document processor with intelligent type detection and optimization"""
    
    def __init__(self,
                 summarize_model: str = None,
                 chunk_size: int = 1024,
                 chunk_overlap: int = 200,
                 enable_summarization: bool = True):
        """
        Initialize the document processor
        
        Args:
            summarize_model: Ollama model for summarization (default: from env or llama3.2:1b)
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            enable_summarization: Whether to generate summaries
        """
        self.summarize_model = summarize_model or os.getenv("SUMMARIZE_MODEL", "llama3.2:1b")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_summarization = enable_summarization and bool(summarize_model)
        
        # Initialize summarizer
        if self.enable_summarization:
            try:
                self.summarizer = OllamaLLM(model=self.summarize_model, temperature=0.3)
                logger.info(f"Initialized summarizer with model: {self.summarize_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize summarizer: {e}. Disabling summarization.")
                self.enable_summarization = False
        
        # Initialize text splitter with optimized settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
            separators=[
                "\n\n\n",  # Multiple newlines (section breaks)
                "\n\n",    # Paragraph breaks
                "\n",      # Line breaks
                ". ",      # Sentence endings
                "! ",      # Exclamation endings
                "? ",      # Question endings
                "; ",      # Semicolon breaks
                ", ",      # Comma breaks
                " ",       # Word breaks
                ""         # Character breaks (last resort)
            ]
        )
        
        # Document type mapping using MIME types and extensions
        self.loader_mapping = {
            # PDF documents
            'application/pdf': PyPDFLoader,
            '.pdf': PyPDFLoader,
            
            # Microsoft Office documents
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': UnstructuredWordDocumentLoader,
            'application/msword': UnstructuredWordDocumentLoader,
            '.docx': UnstructuredWordDocumentLoader,
            '.doc': UnstructuredWordDocumentLoader,
            
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': UnstructuredExcelLoader,
            'application/vnd.ms-excel': UnstructuredExcelLoader,
            '.xlsx': UnstructuredExcelLoader,
            '.xls': UnstructuredExcelLoader,
            
            'application/vnd.openxmlformats-officedocument.presentationml.presentation': UnstructuredPowerPointLoader,
            'application/vnd.ms-powerpoint': UnstructuredPowerPointLoader,
            '.pptx': UnstructuredPowerPointLoader,
            '.ppt': UnstructuredPowerPointLoader,
            
            # Text-based formats
            'text/plain': TextLoader,
            '.txt': TextLoader,
            '.text': TextLoader,
            
            'text/markdown': UnstructuredMarkdownLoader,
            '.md': UnstructuredMarkdownLoader,
            '.markdown': UnstructuredMarkdownLoader,
            
            'text/html': UnstructuredHTMLLoader,
            'application/xhtml+xml': UnstructuredHTMLLoader,
            '.html': UnstructuredHTMLLoader,
            '.htm': UnstructuredHTMLLoader,
            '.xhtml': UnstructuredHTMLLoader,
            
            # Data formats
            'text/csv': CSVLoader,
            'application/csv': CSVLoader,
            '.csv': CSVLoader,
            
            'application/json': JSONLoader,
            'text/json': JSONLoader,
            '.json': JSONLoader,
            '.jsonl': JSONLoader,
            
            # Other formats
            'application/xml': UnstructuredXMLLoader,
            'text/xml': UnstructuredXMLLoader,
            '.xml': UnstructuredXMLLoader,
            
            'message/rfc822': UnstructuredEmailLoader,
            '.eml': UnstructuredEmailLoader,
            '.msg': UnstructuredEmailLoader,
            
            'application/rtf': UnstructuredRTFLoader,
            'text/rtf': UnstructuredRTFLoader,
            '.rtf': UnstructuredRTFLoader,
            
            # Code files (treat as text)
            '.py': TextLoader,
            '.js': TextLoader,
            '.ts': TextLoader,
            '.java': TextLoader,
            '.cpp': TextLoader,
            '.c': TextLoader,
            '.h': TextLoader,
            '.css': TextLoader,
            '.sql': TextLoader,
            '.yaml': TextLoader,
            '.yml': TextLoader,
            '.toml': TextLoader,
            '.ini': TextLoader,
            '.cfg': TextLoader,
            '.conf': TextLoader,
        }

    def detect_file_type(self, file_path: Path) -> Tuple[str, str]:
        """
        Detect file type using libmagic and file extension
        
        Returns:
            Tuple of (mime_type, file_extension)
        """
        try:
            # Use libmagic to detect MIME type
            mime_type = magic.from_file(str(file_path), mime=True)
            file_extension = file_path.suffix.lower()
            logger.debug(f"Detected {file_path}: mime={mime_type}, ext={file_extension}")
            return mime_type, file_extension
        except Exception as e:
            logger.warning(f"Failed to detect file type for {file_path}: {e}")
            return "application/octet-stream", file_path.suffix.lower()

    def get_loader_for_file(self, file_path: Path) -> Optional[Any]:
        """Get appropriate document loader for a file"""
        mime_type, file_extension = self.detect_file_type(file_path)
        
        # Try MIME type first, then file extension
        loader_class = self.loader_mapping.get(mime_type) or self.loader_mapping.get(file_extension)
        
        if not loader_class:
            logger.warning(f"No loader found for {file_path} (mime: {mime_type}, ext: {file_extension})")
            return None
            
        try:
            # Special handling for specific loaders
            if loader_class == CSVLoader:
                return loader_class(str(file_path), encoding='utf-8')
            elif loader_class == JSONLoader:
                return loader_class(str(file_path), jq_schema='.')
            else:
                return loader_class(str(file_path))
        except Exception as e:
            logger.error(f"Failed to create loader for {file_path}: {e}")
            return None

    def generate_summary(self, text: str, max_length: int = 200) -> str:
        """Generate a concise summary of the text using Ollama"""
        if not self.enable_summarization or not text.strip():
            return ""
            
        try:
            # Limit input text to avoid token limits
            max_input_chars = 4000
            if len(text) > max_input_chars:
                text = text[:max_input_chars] + "..."
            
            prompt = f"""Provide a concise, informative summary of the following text in {max_length} characters or less. Focus on the main topics, key information, and purpose:

{text}

Summary:"""
            
            summary = self.summarizer.invoke(prompt)
            
            # Clean and limit the summary
            summary = summary.strip()
            if len(summary) > max_length:
                summary = summary[:max_length - 3] + "..."
                
            return summary
            
        except Exception as e:
            logger.warning(f"Failed to generate summary: {e}")
            return ""

    def extract_metadata(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Extract comprehensive metadata from document"""
        stat = file_path.stat()
        mime_type, file_extension = self.detect_file_type(file_path)
        
        # Generate content hash for deduplication
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        metadata = {
            'source': str(file_path),
            'filename': file_path.name,
            'file_extension': file_extension,
            'mime_type': mime_type,
            'file_size': stat.st_size,
            'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'content_hash': content_hash,
            'content_length': len(content),
            'word_count': len(content.split()),
            'processing_timestamp': datetime.now().isoformat()
        }
        
        # Add summary if enabled
        if self.enable_summarization:
            summary = self.generate_summary(content)
            if summary:
                metadata['summary'] = summary
                
        return metadata

    def process_single_document(self, file_path: Path) -> List[Document]:
        """Process a single document file into LangChain Document objects"""
        try:
            logger.info(f"Processing document: {file_path}")
            
            # Get appropriate loader
            loader = self.get_loader_for_file(file_path)
            if not loader:
                logger.warning(f"Skipping unsupported file: {file_path}")
                return []
            
            # Load the document
            documents = loader.load()
            if not documents:
                logger.warning(f"No content loaded from: {file_path}")
                return []
            
            # Combine all pages/content from the document
            full_content = "\n\n".join([doc.page_content for doc in documents])
            
            # Extract comprehensive metadata
            metadata = self.extract_metadata(file_path, full_content)
            
            # Add any existing metadata from the loader
            if documents[0].metadata:
                metadata.update(documents[0].metadata)
            
            # Create optimized chunks
            chunks = self.text_splitter.split_text(full_content)
            
            # Create Document objects with enhanced metadata
            processed_docs = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_id': i,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk),
                    'chunk_word_count': len(chunk.split())
                })
                
                processed_docs.append(Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                ))
            
            logger.info(f"Successfully processed {file_path}: {len(processed_docs)} chunks")
            return processed_docs
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return []

    def process_documents(self, source_path: Union[str, Path]) -> Tuple[List[Document], ProcessingMetrics]:
        """
        Process documents from a file or directory
        
        Returns:
            Tuple of (processed_documents, processing_metrics)
        """
        start_time = datetime.now()
        metrics = ProcessingMetrics()
        all_documents = []
        
        source_path = Path(source_path)
        
        if source_path.is_file():
            files_to_process = [source_path]
        elif source_path.is_dir():
            # Recursively find all files
            files_to_process = []
            for file_path in source_path.rglob('*'):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    files_to_process.append(file_path)
        else:
            logger.error(f"Source path does not exist: {source_path}")
            return [], metrics
        
        logger.info(f"Found {len(files_to_process)} files to process")
        metrics.total_docs = len(files_to_process)
        
        # Process each file
        for file_path in files_to_process:
            try:
                documents = self.process_single_document(file_path)
                if documents:
                    all_documents.extend(documents)
                    metrics.successful_docs += 1
                    metrics.total_chunks += len(documents)
                else:
                    metrics.failed_docs += 1
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                metrics.failed_docs += 1
        
        # Calculate processing time
        end_time = datetime.now()
        metrics.processing_time = (end_time - start_time).total_seconds()
        
        logger.info(f"Processing complete: {metrics.successful_docs}/{metrics.total_docs} files, "
                    f"{metrics.total_chunks} chunks, {metrics.processing_time:.2f}s")
        
        return all_documents, metrics

    def save_processing_report(self, metrics: ProcessingMetrics, output_path: str = "processing_report.json"):
        """Save processing metrics to a JSON report"""
        try:
            report = {
                "processing_timestamp": datetime.now().isoformat(),
                "metrics": {
                    "total_documents": metrics.total_docs,
                    "successful_documents": metrics.successful_docs,
                    "failed_documents": metrics.failed_docs,
                    "total_chunks": metrics.total_chunks,
                    "processing_time_seconds": metrics.processing_time,
                    "success_rate": metrics.successful_docs / metrics.total_docs if metrics.total_docs > 0 else 0,
                    "average_chunks_per_doc": metrics.total_chunks / metrics.successful_docs if metrics.successful_docs > 0 else 0
                },
                "configuration": {
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "summarization_enabled": self.enable_summarization,
                    "summarize_model": self.summarize_model if self.enable_summarization else None
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Processing report saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save processing report: {e}")


def load_and_process_documents(source_path: Union[str, Path],
                               summarize_model: str = None,
                               chunk_size: int = 1024,
                               chunk_overlap: int = 200,
                               enable_summarization: bool = True) -> Tuple[List[Document], ProcessingMetrics]:
    """
    Convenience function to load and process documents
    
    Args:
        source_path: Path to file or directory to process
        summarize_model: Ollama model for summarization
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        enable_summarization: Whether to generate summaries
        
    Returns:
        Tuple of (processed_documents, processing_metrics)
    """
    processor = DocumentProcessor(
        summarize_model=summarize_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        enable_summarization=enable_summarization
    )
    
    return processor.process_documents(source_path)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Process documents for RAG ingestion')
    parser.add_argument('--source', required=True, help='Path to document file or directory')
    parser.add_argument('--output', help='Path to save processing report')
    parser.add_argument('--model', help='Ollama model for summarization')
    parser.add_argument('--chunk-size', type=int, default=1024, help='Chunk size')
    parser.add_argument('--chunk-overlap', type=int, default=200, help='Chunk overlap')
    parser.add_argument('--no-summary', action='store_true', help='Disable summarization')
    
    args = parser.parse_args()
    
    # Process documents
    documents, metrics = load_and_process_documents(
        source_path=args.source,
        summarize_model=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        enable_summarization=not args.no_summary
    )
    
    print(f"Processed {len(documents)} document chunks")
    print(f"Success rate: {metrics.successful_docs}/{metrics.total_docs}")
    
    # Save report if requested
    if args.output:
        processor = DocumentProcessor()
        processor.save_processing_report(metrics, args.output)
