# embedding.py - fully working without LangChain imports
import os
import re
from chromadb import PersistentClient
import chromadb.utils.embedding_functions as embedding_functions
from sentence_transformers import SentenceTransformer

# --- 1. Load MkDocs documentation ---
docs_path = r"C:\Users\zeina\Downloads\mkdocs-master\mkdocs-master\docs"
all_text = ""
for root, dirs, files in os.walk(docs_path):
    for file in files:
        if file.endswith(".md"):
            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                all_text += f.read() + "\n\n"

print(f"Total characters loaded: {len(all_text)}")

# --- 2. Clean text ---
def clean_text(text):
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)  # remove links
    text = re.sub(r"#+", "", text)               # remove headers
    text = re.sub(r"\s+", " ", text)            # remove extra spaces
    return text.strip()

all_text = clean_text(all_text)

# --- 3. Chunk text (custom splitter) ---
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

chunks = chunk_text(all_text)
print(f"Total chunks created: {len(chunks)}")

# --- 4. Embed and add to Chroma ---
chroma_client = PersistentClient(path="db/")
embedding_model_name = "all-MiniLM-L6-v2"
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model_name)
collection = chroma_client.create_collection(name="MkDocsRAG", embedding_function=embedding_function)

collection.add(
    ids=[str(i) for i in range(len(chunks))],
    documents=chunks,
    metadatas=[{"source": "mkdocs_docs"} for _ in chunks]
)

print("Embedding completed successfully!")
