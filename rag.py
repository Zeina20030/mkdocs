# rag.py
import os
from google import genai
from chromadb import PersistentClient
import chromadb.utils.embedding_functions as embedding_functions

# ----------------- Gemini client -----------------
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Set the GOOGLE_API_KEY environment variable first!")
client = genai.Client(api_key=api_key)

# ----------------- Chroma client -----------------
chroma_client = PersistentClient(path="db/")
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Get or create collection
try:
    collection = chroma_client.get_collection(name="MkDocsRAG")
except:
    collection = chroma_client.create_collection(
        name="MkDocsRAG", embedding_function=embedding_function
    )

# ----------------- Main function -----------------
def get_answer(question: str, n_results: int = 5) -> str:
    # Retrieve relevant chunks from Chroma
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    documents = results["documents"][0]
    context = "No relevant documentation found." if not documents else "\n\n".join(documents)

    full_prompt = f"""
Context:
{context}

Question:
{question}
"""

    # Create chat and send message using a valid Gemini model
    chat = client.chats.create(model="models/gemini-2.5-flash")
    response = chat.send_message(full_prompt)

    return response.text
