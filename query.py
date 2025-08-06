from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM as Ollama
from langchain.prompts import PromptTemplate

# Initialize embedding model
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load vector DB
vector_db = Chroma(
    persist_directory="/Users/florentbaillif/cag/vector_db",
    embedding_function=embed_model,
    collection_name="my_documents"  # Add collection name
)

# Initialize Ollama
llm = Ollama(
    model="mistral:latest",  # Match the model you pulled
    temperature=0.1,
    system="You are an expert assistant specialized in answering questions about the user's personal documents. Provide detailed responses based strictly on the following context:"
)

# Custom prompt template for better context awareness
prompt_template = """
<|system|>
You are answering questions about the user's personal documents.
Base your response ONLY on the following context:
{context}</s>
<|user|>
{question}</s>
<|assistant|>
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

# Interactive query loop
while True:
    query = input("\nYour question (type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
        
    result = qa_chain.invoke({"query": query})
    print(f"\nAnswer: {result['result']}")
    
    # Display sources
    print("\nSources:")
    for i, doc in enumerate(result['source_documents']):
        print(f"{i+1}. {doc.metadata['source']} (page {doc.metadata.get('page', 'N/A')})")