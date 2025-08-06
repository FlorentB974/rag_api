import os
import argparse
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader, UnstructuredFileLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configure loader mapping for different file types
LOADER_MAPPING = {
    '.pdf': (PyPDFLoader, {}),
    '.docx': (Docx2txtLoader, {}),
    '.csv': (CSVLoader, {}),
    '.yaml': (UnstructuredFileLoader, {"mode": "single"}),
    '.yml': (UnstructuredFileLoader, {"mode": "single"}),
    '.txt': (UnstructuredFileLoader, {"mode": "single"}),
    '.md': (UnstructuredFileLoader, {"mode": "single"}),
}

def load_documents(source_path: str):
    """Load documents from a file or directory"""
    docs = []
    source_path = Path(source_path)
    
    if source_path.is_file():
        file_ext = source_path.suffix.lower()
        if file_ext in LOADER_MAPPING:
            loader_class, loader_args = LOADER_MAPPING[file_ext]
            loader = loader_class(str(source_path), **loader_args)
            docs.extend(loader.load())
            print(f"Loaded {source_path}")
        else:
            print(f"Skipped unsupported file: {source_path}")
    
    elif source_path.is_dir():
        for ext, (loader_class, loader_args) in LOADER_MAPPING.items():
            loader = DirectoryLoader(
                str(source_path),
                glob=f"**/*{ext}",
                loader_cls=loader_class,
                loader_kwargs=loader_args,
                use_multithreading=True,
                show_progress=True
            )
            docs.extend(loader.load())
        print(f"Loaded {len(docs)} documents from directory")
    
    return docs

def main():
    parser = argparse.ArgumentParser(description='Manage vector database')
    parser.add_argument('--source', required=True, 
                        help='Path to document file or directory')
    parser.add_argument('--db', required=True, 
                        help='Path to vector database directory')
    parser.add_argument('--init', action='store_true',
                        help='Initialize new database (overwrites existing)')
    args = parser.parse_args()

    # Load and split documents
    documents = load_documents(args.source)
    if not documents:
        print("No documents loaded. Exiting.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", ". ", "! ", "? ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Initialize embedding model
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # Handle database mode
    if args.init:
        # Create new database (overwrite existing)
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embed_model,
            persist_directory=args.db,
            collection_metadata={"hnsw:space": "cosine"},
            collection_name="my_documents"
        )
        print(f"Created new vector database with {len(chunks)} chunks")
    else:
        # Add to existing database
        vector_db = Chroma(
            persist_directory=args.db,
            embedding_function=embed_model,
            collection_name="my_documents"
        )
        # Chroma now automatically persists when using persist_directory
        vector_db.add_documents(chunks)
        print(f"Added {len(chunks)} chunks to existing database")
    
    # Get document count - new method
    try:
        # For newer Chroma versions
        count = vector_db._collection.count()
        print(f"Total entries in database: {count}")
    except AttributeError:
        # For older versions
        print(f"Database updated. Collection count not available via this method.")

if __name__ == "__main__":
    main()