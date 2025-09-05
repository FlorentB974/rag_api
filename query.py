import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM as Ollama
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_documents")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:latest")

# Initialize embedding model
embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

# Load vector DB
vector_db = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embed_model,
    collection_name=COLLECTION_NAME  # Read collection name from env
)

# Initialize Ollama
llm = Ollama(
    model=OLLAMA_MODEL,  # Read model from env
    temperature=0.1,
    system="You are a precise and careful assistant that answers questions "
    "about the user's personal documents. You must: "
    "1. Base answers SOLELY on the provided context\n"
    "2. If the context doesn't contain enough information, say so\n"
    "3. Never make assumptions or add information not present in the context\n"
    "4. Quote relevant parts of the context to support your answers\n"
    "5. Double-check your response against the context before providing it"
)

# Custom prompt template for better context awareness
prompt_template = """
<|system|>
You are a precise and careful assistant. Follow these steps for EVERY response:

1. First Reading:
   - Read the context carefully
   - Identify specific facts and information relevant to the question
   - Note exact quotes that support your answer

2. Analysis:
   - Consider ONLY information explicitly stated in the context
   - Do not make assumptions or add external knowledge
   - If the context lacks sufficient information, prepare to acknowledge this

3. Double-Check:
   - Re-read the context to verify each point you plan to make
   - Confirm that every statement in your answer is directly supported by the context
   - Remove any statements that cannot be verified from the context

4. Response Formation:
   - Start with a clear, direct answer to the question
   - Support your answer with specific references from the context
   - Use phrases like "According to the context..." or "The document states..."
   - If information is missing or unclear, explicitly say so

Context for this response:
{context}</s>

<|user|>
{question}</s>

<|assistant|>
Let me analyze your question using the provided context...</s>
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Create retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# Initialize conversation context
conversation_context = []

# Interactive query loop
while True:
    query = input("\nYour question (type 'exit' to quit): ")
    if query.lower() == 'exit':
        break

    # Add previous context to the query
    context_string = "\n\nPrevious conversation context:\n" + "\n".join([
        f"Q: {item['question']}\nA: {item['answer']}"
        for item in conversation_context
    ]) if conversation_context else ""
    
    result = qa_chain.invoke({
        "query": query + context_string
    })
    
    # Store the Q&A pair in context
    conversation_context.append({
        "question": query,
        "answer": result['result']
    })
    
    print(f"\nAnswer: {result['result']}")

    # Display sources
    print("\nSources:")
    for i, doc in enumerate(result['source_documents']):
        print(f"{i + 1}. {doc.metadata['source']} (page {doc.metadata.get('page', 'N/A')})")
