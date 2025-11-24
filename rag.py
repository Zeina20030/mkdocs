# rag.py
import os
from google import genai  # Google Gemini client
from chromadb import PersistentClient
import chromadb.utils.embedding_functions as embedding_functions

# ----------------- Configure Gemini -----------------
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Set the GOOGLE_API_KEY environment variable first!")

client = genai.Client(api_key=api_key)

# ----------------- Connect to Chroma -----------------
chroma_client = PersistentClient(path="db/")
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Try to get existing collection; if not exists, create it
try:
    collection = chroma_client.get_collection(name="MkDocsRAG")
except:
    collection = chroma_client.create_collection(
        name="MkDocsRAG", embedding_function=embedding_function
    )

# ----------------- Main RAG function -----------------
def get_answer(question: str, n_results: int = 5) -> str:
    """
    Returns an answer from MkDocs documentation using vector search and Gemini.
    """
    # 1️⃣ Query Chroma for most relevant chunks
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    documents = results["documents"][0]

    if not documents:
        context = "No relevant documentation found."
    else:
        context = "\n\n".join(documents)

    # 2️⃣ Build prompt for Gemini
    full_prompt = f"""
You are a helpful assistant that answers questions about MkDocs documentation.

Context:
{context}

Question:
{question}
"""

    # 3️⃣ Call Gemini model
    response = client.chat.completions.create(
        model="gemini-2.0",  # or "gemini-2.0-chat"
        messages=[{"role": "user", "content": full_prompt}]
    )

    answer = response.last.message["content"].strip()
    return answer
