import os
import argparse
import shutil
from pathlib import Path
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from rag_utils import load_and_process_documents

load_dotenv()
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_documents")
SUMMARIZE_MODEL = os.getenv("SUMMARIZE_MODEL", "llama3.2:1b")


def clear_database(db_path: str):
    """Clear the existing vector database directory"""
    db_path_obj = Path(db_path)
    if db_path_obj.exists():
        print(f"Clearing existing database at: {db_path_obj}")
        shutil.rmtree(db_path_obj)
        print("Database cleared successfully")
    else:
        print(f"Database directory does not exist: {db_path_obj}")


def main():
    parser = argparse.ArgumentParser(description='Manage vector database with advanced document processing')
    parser.add_argument('--source', required=True,
                        help='Path to document file or directory')
    parser.add_argument('--db', required=True,
                        help='Path to vector database directory')
    parser.add_argument('--init', action='store_true',
                        help='Initialize new database (clears existing database)')
    parser.add_argument('--chunk-size', type=int, default=1024,
                        help='Size of text chunks (default: 1024)')
    parser.add_argument('--chunk-overlap', type=int, default=200,
                        help='Overlap between chunks (default: 200)')
    parser.add_argument('--summarize-model', default=SUMMARIZE_MODEL,
                        help=f'Ollama model for summarization (default: {SUMMARIZE_MODEL})')
    parser.add_argument('--no-summary', action='store_true',
                        help='Disable document summarization')
    parser.add_argument('--report',
                        help='Path to save processing report (optional)')
    args = parser.parse_args()

    # Clear database if initializing
    if args.init:
        clear_database(args.db)

    # Process documents using the advanced rag_utils
    print("Processing documents with advanced pipeline...")
    documents, metrics = load_and_process_documents(
        source_path=args.source,
        summarize_model=args.summarize_model if not args.no_summary else "",
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        enable_summarization=not args.no_summary
    )

    if not documents:
        print("No documents were processed successfully. Exiting.")
        return

    print(f"Successfully processed {metrics.successful_docs}/{metrics.total_docs} documents")
    print(f"Generated {len(documents)} document chunks")
    print(f"Processing took {metrics.processing_time:.2f} seconds")

    # Initialize embedding model
    print(f"Initializing embedding model: {EMBED_MODEL_NAME}")
    embed_model = OllamaEmbeddings(
        model=EMBED_MODEL_NAME,
        # model_kwargs={"device": "cpu"}  # Uncomment if you want to force CPU
    )

    # Handle database mode
    if args.init:
        # Create new database
        print("Creating new vector database...")
        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=embed_model,
            persist_directory=args.db,
            collection_metadata={"hnsw:space": "cosine"},
            collection_name=COLLECTION_NAME
        )
        print(f"Created new vector database with {len(documents)} chunks")
    else:
        # Add to existing database
        print("Adding documents to existing database...")
        vector_db = Chroma(
            persist_directory=args.db,
            embedding_function=embed_model,
            collection_name=COLLECTION_NAME
        )
        # Add documents to existing collection
        vector_db.add_documents(documents)
        print(f"Added {len(documents)} chunks to existing database")

    # Get document count
    try:
        count = vector_db._collection.count()
        print(f"Total entries in database: {count}")
    except AttributeError:
        print("Database updated. Collection count not available via this method.")

    # Save processing report if requested
    if args.report:
        from rag_utils import DocumentProcessor
        processor = DocumentProcessor(
            summarize_model=args.summarize_model if not args.no_summary else "",
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            enable_summarization=not args.no_summary
        )
        processor.save_processing_report(metrics, args.report)

    print("Vector database operation completed successfully!")


if __name__ == "__main__":
    main()
