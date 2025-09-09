"""
OpenWebUI-compatible RAG API Server (Simplified)
Provides OpenAI-compatible endpoints for integration with OpenWebUI
"""

import os
import time
import json
import logging
import asyncio
import httpx
from typing import List, Dict, Any, AsyncGenerator, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM as Ollama
from langchain.prompts import PromptTemplate
from langchain.callbacks import AsyncIteratorCallbackHandler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_documents")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "nomic-embed-text")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "5500"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for shared components
embed_model = None
vector_db = None

# In-memory conversation store: conversation_id -> List[Dict[str, str]]
conversation_store: Dict[str, List[Dict[str, str]]] = {}


# Pydantic models for OpenAI compatibility
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = False
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    # Optional conversational context storage
    conversation_id: Optional[str] = None
    # How many previous messages to include from the provided `messages` list
    max_context_messages: int = 10


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "rag-api"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# Simple prompt template that works with existing chain
PROMPT_TEMPLATE = """You are an expert assistant that answers questions based on the provided document context. Follow these strict guidelines:

1. ONLY use information explicitly stated in the provided context
2. If the answer is not in the context, clearly state: "I couldn't find that information in the provided documents"
3. Never make assumptions or add information from outside the context
4. Quote relevant parts of the context when possible
5. Be precise and accurate in your responses

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


async def initialize_components():
    """Initialize embedding model and vector database"""
    global embed_model, vector_db
    
    try:
        logger.info(f"Initializing embedding model: {EMBED_MODEL_NAME}")
        embed_model = OllamaEmbeddings(model=EMBED_MODEL_NAME)
        
        logger.info(f"Loading vector database from: {VECTOR_DB_PATH}")
        vector_db = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embed_model,
            collection_name=COLLECTION_NAME
        )
        
        # Test the database
        count = vector_db._collection.count()
        logger.info(f"Vector database loaded successfully with {count} documents")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise


async def fetch_ollama_models() -> List[Dict[str, Any]]:
    """Fetch available models from Ollama"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = []
                for model in data.get("models", []):
                    models.append({
                        "id": model["name"],
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "ollama",
                        "provider": "ollama"
                    })
                logger.info(f"Fetched {len(models)} Ollama models")
                return models
            else:
                logger.warning(f"Failed to fetch Ollama models: {response.status_code}")
                return []
    except Exception as e:
        logger.warning(f"Error fetching Ollama models: {e}")
        return []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting RAG API server...")
    await initialize_components()
    logger.info("RAG API server started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG API server...")


# Create FastAPI app
app = FastAPI(
    title="RAG API - OpenWebUI Compatible",
    description="OpenAI-compatible API for Retrieval Augmented Generation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for OpenWebUI integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def format_openai_chunk(content: str, is_final: bool = False, model: Optional[str] = None) -> Dict[str, Any]:
    """Format chunk for OpenAI-compatible streaming"""
    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model or OLLAMA_MODEL,
        "choices": [{
            "index": 0,
            "delta": {"content": content} if content else {},
            "finish_reason": "stop" if is_final else None
        }]
    }


def format_sources(source_documents: List[Any]) -> str:
    """Format source documents for display"""
    if not source_documents:
        return ""
    
    sources = "\n\n**Sources:**"
    unique_sources = set()
    
    for i, doc in enumerate(source_documents, 1):
        if hasattr(doc, 'metadata'):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            filename = os.path.basename(source)
            
            source_info = f"{filename} (page {page})"
            if source_info not in unique_sources:
                unique_sources.add(source_info)
                sources += f"\n{len(unique_sources)}. {source_info}"
    
    return sources


async def determine_model_provider(model: str) -> str:
    """Determine if a model is from Ollama"""
    # Currently only Ollama provider is supported; default to 'ollama'
    ollama_models = await fetch_ollama_models()
    ollama_model_ids = [m["id"] for m in ollama_models]

    if model in ollama_model_ids or model == OLLAMA_MODEL:
        return "ollama"

    return "ollama"


async def stream_rag_response(query: str, model: str, conversation_id: Optional[str] = None, user_message_obj: Optional[Dict[str, str]] = None) -> AsyncGenerator[str, None]:
    """Stream RAG response with sources and optionally persist conversation history"""
    if not vector_db:
        error_chunk = format_openai_chunk("Vector database is not available.", model=model)
        yield f"data: {json.dumps(error_chunk)}\n\n"
        final_chunk = format_openai_chunk("", is_final=True, model=model)
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        return

    callback = AsyncIteratorCallbackHandler()
    tokens_collected: List[str] = []

    try:
        # Create LLM with callback for streaming
        llm = Ollama(
            model=model,
            temperature=0.1,
            callbacks=[callback]
        )

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={"k": 10}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

        # Start the chain execution
        async def run_chain():
            try:
                return await qa_chain.ainvoke({"query": query})
            except Exception as e:
                logger.error(f"Error in chain execution: {e}")
                return None

        # Start background task
        task = asyncio.create_task(run_chain())

        # Stream tokens as they come and collect them to form the final assistant message
        try:
            async for token in callback.aiter():
                if token:
                    tokens_collected.append(token)
                    chunk = format_openai_chunk(token, model=model)
                    yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            logger.error(f"Error during streaming: {e}")

        # Wait for full result to get sources and full output
        result = await task

        # Build final assistant response text from collected tokens or chain result fallback
        final_response_text = "".join(tokens_collected).strip()
        if not final_response_text:
            # Try common keys used by langchain
            if isinstance(result, dict):
                final_response_text = result.get('result') or result.get('answer') or result.get('output_text') or ''

        if result and "source_documents" in result:
            sources_text = format_sources(result['source_documents'])
            if sources_text:
                sources_chunk = format_openai_chunk(sources_text, model=model)
                yield f"data: {json.dumps(sources_chunk)}\n\n"

        # Persist conversation if conversation_id and user message provided
        try:
            if conversation_id and user_message_obj is not None:
                convo = conversation_store.setdefault(conversation_id, [])
                # Append user message
                convo.append({
                    'role': user_message_obj.get('role', 'user'),
                    'content': user_message_obj.get('content', '')
                })
                # Append assistant message
                convo.append({
                    'role': 'assistant',
                    'content': final_response_text
                })
        except Exception as e:
            logger.warning(f"Failed to persist conversation {conversation_id}: {e}")

    except Exception as e:
        logger.error(f"Error in stream_rag_response: {e}")
        error_chunk = format_openai_chunk(f"Error: {str(e)}", model=model)
        yield f"data: {json.dumps(error_chunk)}\n\n"

    # Send final chunk
    final_chunk = format_openai_chunk("", is_final=True, model=model)
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/v1/models")
async def list_models():
    """List available models - OpenAI compatible"""
    models = []
    
    # Fetch Ollama models
    ollama_models = await fetch_ollama_models()
    models.extend(ollama_models)
    
    return ModelsResponse(data=models)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    try:
        # Extract the user message (last user role in messages)
        user_message = None
        last_user_index = None
        for i in range(len(request.messages) - 1, -1, -1):
            if request.messages[i].role == "user":
                user_message = request.messages[i].content
                last_user_index = i
                break

        if not user_message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No user message found in request"
            )

        # Build conversation history from provided messages (exclude the current user message)
        history_msgs = []
        if last_user_index is not None:
            start_idx = max(0, last_user_index - request.max_context_messages)
            for m in request.messages[start_idx:last_user_index]:
                history_msgs.append(f"{m.role}: {m.content}")

        conversation_context = "\n".join(history_msgs)
        if conversation_context:
            combined_question = f"Conversation history:\n{conversation_context}\n\nCurrent question: {user_message}"
        else:
            combined_question = user_message

        logger.info(f"Processing query (conversation_id={request.conversation_id}): {user_message[:100]}...")

        # Prepare user message object for potential persistence
        user_message_obj = None
        if last_user_index is not None:
            um = request.messages[last_user_index]
            user_message_obj = {"role": um.role, "content": um.content}

        # Handle Ollama model with RAG and pass conversation id for persistence
        return StreamingResponse(
            stream_rag_response(combined_question, request.model, conversation_id=request.conversation_id, user_message_obj=user_message_obj),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            }
        )

    except Exception as e:
        logger.error(f"Error in chat_completions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    is_healthy = vector_db is not None
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "timestamp": int(time.time()),
        "vector_db_loaded": is_healthy,
        "model": OLLAMA_MODEL
    }


@app.get("/documents")
async def list_documents():
    """List documents in the vector database"""
    try:
        if not vector_db:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector database is not available"
            )

        collection = vector_db.get()
        metadatas = collection.get('metadatas', [])

        unique_sources = set()
        document_info = []
        
        for meta in metadatas:
            if 'source' in meta:
                source = meta['source']
                filename = os.path.basename(source)
                if filename not in unique_sources:
                    unique_sources.add(filename)
                    document_info.append({
                        "filename": filename,
                        "source": source,
                        "created_time": meta.get('created_time'),
                        "modified_time": meta.get('modified_time'),
                        "summary": meta.get('summary', 'No summary available')
                    })

        return {
            "documents": document_info,
            "total_count": len(document_info)
        }

    except Exception as e:
        logger.error(f"Error fetching documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving documents: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "RAG API - OpenWebUI Compatible",
        "version": "1.0.0",
        "description": "OpenAI-compatible API for Retrieval Augmented Generation",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health",
            "documents": "/documents"
        },
        "vector_db_loaded": vector_db is not None,
        "default_model": OLLAMA_MODEL
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting RAG API server on {API_HOST}:{API_PORT}")
    uvicorn.run(
        "simple_openwebui_api:app",
        host=API_HOST,
        port=API_PORT,
        log_level=LOG_LEVEL.lower(),
        reload=False
    )
