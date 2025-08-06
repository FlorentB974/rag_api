import os
import time
import json
import logging
import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM as Ollama
from langchain.prompts import PromptTemplate
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import List, Dict, Any, AsyncGenerator

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global, expensive-to-load components ---
# These are loaded once at startup and are safe for concurrent reads.
PERSIST_DIRECTORY = os.getenv("VECTOR_DB_PATH", "/app/vector_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_documents")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")

# Load embeddings and vector DB globally
try:
    embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vector_db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embed_model,
        collection_name=COLLECTION_NAME
    )
    logger.info("Successfully loaded Vector DB and Embedding Model.")
except Exception as e:
    logger.error(f"Failed to load vector DB or embedding model: {e}", exc_info=True)
    vector_db = None # Handle case where DB fails to load

PROMPT_TEMPLATE = """
<|system|>
You are an expert assistant for document-based questions. Strict rules:
1. ANSWER ONLY from provided context
2. If answer isn't EXACTLY in context, say: "I couldn't find that in my documents"
3. Never invent details
4. For numerical queries, verify calculations twice
5. Quote sources verbatim when possible

Context:
{context}
</s>

<|user|>
{question}
</s>

<|assistant|>
"""
PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# --- OpenAI-compatible formatting functions ---
def format_openai_chunk(content: str, is_final: bool = False) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": OLLAMA_MODEL,
        "choices": [{
            "index": 0,
            "delta": {"content": content},
            "finish_reason": "stop" if is_final else None
        }]
    }

def format_sources(source_documents: List[Dict]) -> str:
    if not source_documents:
        return ""
    sources = "\n\n**Sources:**"
    for i, doc in enumerate(source_documents, 1):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        sources += f"\n{i}. {source} (page {page})"
    return sources

# --- Core Streaming Logic ---
async def stream_qa_response(query: str) -> AsyncGenerator[str, None]:
    """
    This is the core async generator that handles the streaming.
    It creates request-specific instances and uses asyncio to stream tokens
    while the full chain runs in the background.
    """
    if not vector_db:
        error_message = "Vector database is not available."
        error_chunk = format_openai_chunk(error_message)
        yield f"data: {json.dumps(error_chunk)}\n\n"
        final_chunk = format_openai_chunk("", is_final=True)
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        return

    callback = AsyncIteratorCallbackHandler()

    # Create request-specific instances of the LLM and Chain
    llm = Ollama(
        model=OLLAMA_MODEL,
        temperature=0.01,  # Lower for determinism
        top_k=20,          # Focus on high-probability tokens
        top_p=0.9,
        repeat_penalty=1.2,  # Discourage repetition
        num_ctx=4096,       # Match context window size        
        system="You are an expert assistant... (system prompt)",
        callbacks=[callback],  # Hook in the callback
        streaming=True
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    # Define the task to run the chain in the background
    async def run_chain():
        try:
            return await qa_chain.ainvoke({"query": query})
        except Exception as e:
            logger.error(f"Error in chain execution: {e}", exc_info=True)
            # You might want to stream an error message back to the user here
            return None

    # Start the background task
    task = asyncio.create_task(run_chain())

    # Immediately start streaming tokens from the callback handler
    try:
        async for token in callback.aiter():
            formatted_chunk = format_openai_chunk(token)
            yield f"data: {json.dumps(formatted_chunk)}\n\n"
    except Exception as e:
        logger.error(f"Error during token streaming: {e}", exc_info=True)

    # Wait for the background task to complete to get the full result
    result = await task
    
    # Stream the sources if they are available in the final result
    if result and "source_documents" in result:
        sources_text = format_sources(result['source_documents'])
        if sources_text:
            sources_chunk = format_openai_chunk(sources_text)
            yield f"data: {json.dumps(sources_chunk)}\n\n"

    # Send the final closing message
    final_chunk = format_openai_chunk("", is_final=True)
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post('/v1/chat/completions')
async def ask(request: Request):
    try:
        data = await request.json()
        streaming = data.get('stream', False)
        
        user_message = next((msg['content'] for msg in reversed(data.get('messages', [])) if msg['role'] == 'user'), None)
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found.")
            
        query = user_message
        logger.info(f"Processing query: {query}")

        if streaming:
            return StreamingResponse(stream_qa_response(query), media_type="text/event-stream")
        
        # Non-streaming response logic remains the same but uses a non-callback chain
        llm = Ollama(model=OLLAMA_MODEL, temperature=0.1)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=vector_db.as_retriever(),
            chain_type_kwargs={"prompt": PROMPT}, return_source_documents=True
        )
        result = await qa_chain.ainvoke({"query": query})
        response_content = result['result'] + format_sources(result.get('source_documents', []))
        
        return format_openai_response(
            content=response_content,
            prompt_tokens=len(query.split()),
            completion_tokens=len(response_content.split())
        )
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

# Helper for non-streaming response
def format_openai_response(content: str, prompt_tokens: int, completion_tokens: int) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": OLLAMA_MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}
    }


@app.get('/health')
def health_check():
    return {"status": "healthy" if vector_db else "unhealthy", "port": 5500}

@app.get('/documents')
def get_documents():
    """Return unique document sources from the vector database"""
    try:
        # Access the Chroma collection metadata
        collection = vector_db.get()
        metadatas = collection.get('metadatas', [])
        
        # Extract unique document sources
        unique_sources = set()
        for meta in metadatas:
            if 'source' in meta:
                # Extract filename from full path
                source = meta['source']
                filename = os.path.basename(source)
                unique_sources.add(filename)
        
        return {"documents": sorted(list(unique_sources))}
    
    except Exception as e:
        logger.exception(f"Error fetching documents: {str(e)}")
        return {"error": "Failed to retrieve documents", "detail": str(e)}

if __name__ == '__main__':
    import uvicorn
    # Use a larger number of workers if needed, but be mindful of memory usage
    uvicorn.run(app, host='0.0.0.0', port=5500, log_level="info")