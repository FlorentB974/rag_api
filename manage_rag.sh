#!/bin/bash

# RAG API Management Script for OpenWebUI Integration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
VECTOR_DB_PATH="$SCRIPT_DIR/vector_db"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

print_banner() {
    echo -e "${BLUE}"
    echo "=================================================="
    echo "     RAG API - OpenWebUI Integration Manager"
    echo "=================================================="
    echo -e "${NC}"
}

print_usage() {
    echo "Usage: $0 {start|stop|restart|status|setup|ingest|test|logs|clean}"
    echo
    echo "Commands:"
    echo "  start    - Start the RAG API server"
    echo "  stop     - Stop the RAG API server"  
    echo "  restart  - Restart the RAG API server"
    echo "  status   - Check server status"
    echo "  setup    - Initial setup and dependency installation"
    echo "  ingest   - Ingest documents into vector database"
    echo "  test     - Test the API endpoints"
    echo "  logs     - View server logs"
    echo "  clean    - Clean vector database and cache"
}

check_dependencies() {
    echo -e "${YELLOW}Checking dependencies...${NC}"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Python 3 is required but not installed.${NC}"
        exit 1
    fi
    
    # Check Ollama
    if ! command -v ollama &> /dev/null; then
        echo -e "${YELLOW}Warning: Ollama not found. Make sure it's running on localhost:11434${NC}"
    fi
    
    echo -e "${GREEN}Dependencies check completed.${NC}"
}

setup_environment() {
    echo -e "${YELLOW}Setting up environment...${NC}"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "rag_env" ]; then
        echo "Creating virtual environment..."
        python3 -m venv rag_env
    fi
    
    # Activate virtual environment
    source rag_env/bin/activate
    
    # Install/upgrade requirements
    if [ -f "$REQUIREMENTS_FILE" ]; then
        echo "Installing Python dependencies..."
        pip install --upgrade pip
        pip install -r "$REQUIREMENTS_FILE"
    else
        echo -e "${RED}Requirements file not found: $REQUIREMENTS_FILE${NC}"
        exit 1
    fi
    
    # Create vector database directory
    mkdir -p "$VECTOR_DB_PATH"
    
    # Create documents directory for ingestion
    mkdir -p "$SCRIPT_DIR/documents"
    
    echo -e "${GREEN}Environment setup completed.${NC}"
}

start_server() {
    echo -e "${YELLOW}Starting RAG API server...${NC}"
    
    # Check if already running
    if pgrep -f "simple_openwebui_api.py" > /dev/null; then
        echo -e "${YELLOW}Server is already running.${NC}"
        return 0
    fi
    
    # Activate virtual environment
    source rag_env/bin/activate
    
    # Start server in background
    nohup python simple_openwebui_api.py > rag_api.log 2>&1 &
    
    # Wait a moment for startup
    sleep 3
    
    # Check if started successfully
    if pgrep -f "simple_openwebui_api.py" > /dev/null; then
        echo -e "${GREEN}RAG API server started successfully.${NC}"
        echo "Server running on http://localhost:5500"
        echo "Logs: tail -f rag_api.log"
    else
        echo -e "${RED}Failed to start server. Check rag_api.log for details.${NC}"
        exit 1
    fi
}

stop_server() {
    echo -e "${YELLOW}Stopping RAG API server...${NC}"
    
    if pgrep -f "simple_openwebui_api.py" > /dev/null; then
        pkill -f "simple_openwebui_api.py"
        sleep 2
        echo -e "${GREEN}Server stopped.${NC}"
    else
        echo -e "${YELLOW}Server is not running.${NC}"
    fi
}

check_status() {
    echo -e "${YELLOW}Checking server status...${NC}"
    
    if pgrep -f "simple_openwebui_api.py" > /dev/null; then
        echo -e "${GREEN}✓ Server is running${NC}"
        
        # Test health endpoint
        if curl -s http://localhost:5500/health > /dev/null; then
            echo -e "${GREEN}✓ Health check passed${NC}"
        else
            echo -e "${RED}✗ Health check failed${NC}"
        fi
        
        # Check vector database
        if [ -d "$VECTOR_DB_PATH" ] && [ "$(ls -A $VECTOR_DB_PATH 2>/dev/null)" ]; then
            echo -e "${GREEN}✓ Vector database exists${NC}"
        else
            echo -e "${YELLOW}⚠ Vector database is empty or missing${NC}"
        fi
    else
        echo -e "${RED}✗ Server is not running${NC}"
    fi
}

ingest_documents() {
    echo -e "${YELLOW}Ingesting documents...${NC}"
    
    # Activate virtual environment
    source rag_env/bin/activate
    
    # Check if documents directory exists and has files
    if [ ! -d "$SCRIPT_DIR/documents" ] || [ -z "$(ls -A $SCRIPT_DIR/documents 2>/dev/null)" ]; then
        echo -e "${YELLOW}No documents found in $SCRIPT_DIR/documents${NC}"
        echo "Please add documents to the documents/ directory and try again."
        return 1
    fi
    
    # Run document ingestion
    echo "Processing documents from documents/ directory..."
    python vector_db.py --source "$SCRIPT_DIR/documents" --db "$VECTOR_DB_PATH"
    
    echo -e "${GREEN}Document ingestion completed.${NC}"
}

test_api() {
    echo -e "${YELLOW}Testing API endpoints...${NC}"
    
    BASE_URL="http://localhost:5500"
    
    # Test health endpoint
    echo "Testing health endpoint..."
    if curl -s "$BASE_URL/health" | jq . > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Health endpoint OK${NC}"
    else
        echo -e "${RED}✗ Health endpoint failed${NC}"
        return 1
    fi
    
    # Test models endpoint
    echo "Testing models endpoint..."
    if curl -s "$BASE_URL/v1/models" | jq . > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Models endpoint OK${NC}"
    else
        echo -e "${RED}✗ Models endpoint failed${NC}"
        return 1
    fi
    
    # Test chat completions endpoint
    echo "Testing chat completions endpoint..."
    if curl -s -X POST "$BASE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "llama3.2:1b",
            "messages": [{"role": "user", "content": "Hello, can you help me?"}],
            "stream": false
        }' | jq . > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Chat completions endpoint OK${NC}"
    else
        echo -e "${RED}✗ Chat completions endpoint failed${NC}"
        return 1
    fi
    
    echo -e "${GREEN}All API tests passed!${NC}"
}

view_logs() {
    if [ -f "rag_api.log" ]; then
        tail -f rag_api.log
    else
        echo -e "${YELLOW}No log file found.${NC}"
    fi
}

clean_database() {
    echo -e "${YELLOW}Cleaning vector database and cache...${NC}"
    
    read -p "This will delete all ingested documents. Continue? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VECTOR_DB_PATH"/*
        rm -rf __pycache__/
        rm -f rag_api.log
        echo -e "${GREEN}Cleanup completed.${NC}"
    else
        echo "Cleanup cancelled."
    fi
}

main() {
    print_banner
    
    case "${1:-}" in
        start)
            check_dependencies
            start_server
            ;;
        stop)
            stop_server
            ;;
        restart)
            stop_server
            sleep 2
            start_server
            ;;
        status)
            check_status
            ;;
        setup)
            check_dependencies
            setup_environment
            ;;
        ingest)
            ingest_documents
            ;;
        test)
            test_api
            ;;
        logs)
            view_logs
            ;;
        clean)
            clean_database
            ;;
        *)
            print_usage
            exit 1
            ;;
    esac
}

# Change to script directory
cd "$SCRIPT_DIR"

# Run main function
main "$@"
