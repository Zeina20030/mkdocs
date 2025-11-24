from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import chromadb.utils.embedding_functions as embedding_functions
from transformers import pipeline

# Initialize Chroma
chroma_client = PersistentClient(path="db/")  # local persistent DB
embedding_model_name = "all-MiniLM-L6-v2"
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model_name)
collection = chroma_client.get_collection(name="MkDocsRAG", embedding_function=embedding_function)

# Load LLM (you can replace with your preferred HuggingFace model)
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")

def get_answer(prompt: str) -> str:
    # Query the vector DB
    results = collection.query(
        query_texts=[prompt],
        n_results=5
    )
    documents = results["documents"][0]
    
    # Build prompt for LLM
    context = "\n\n".join(documents)
    full_prompt = f"""You are an expert assistant for MkDocs documentation.
Use the following context to answer the question.

Context:
{context}

Question:
{prompt}
"""
    response = qa_pipeline(full_prompt, max_length=500)
    return response[0]['generated_text'].strip()
