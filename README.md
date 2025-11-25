# MkDocs RAG System with FastAPI & Google Gemini

This project implements a Retrieval-Augmented Generation (RAG) system for helping users navigate and understand **MkDocs documentation**. It uses:

* **ChromaDB** as the vector database
* **SentenceTransformer (all-MiniLM-L6-v2)** for embedding generation
* **Google Gemini 2.5 Flash** for answering questions
* **FastAPI** as an API interface

---

## Features

* Ask any MkDocs-related question via the `/ask` API endpoint
* Adjustable `k` (number of retrieved chunks)
* RAG implementation to enhance accuracy using real MkDocs documentation
* Local server with Swagger UI: `http://127.0.0.1:8000/docs`
* Auto-adjusts when fewer chunks exist in the vector DB

---

##  Project Structure

```
project/
│
├── app.py              # FastAPI application
├── rag.py              # RAG logic (embeddings, retrieval, LLM)
├── db/                 # ChromaDB persistent storage
├── static/             # Optional frontend assets
└── README.md           # Project documentation
```

---

##  RAG Pipeline

### 1. **Chunking Method**

* Uses **semantic chunking** of MkDocs documentation sections.
* Reason: MkDocs docs contain short, structured paragraphs that work best when stored in mid-sized semantic chunks.

### 2. **Chunk Cleaning**

* Removed Markdown noise, code block fencing, and headings where needed.
* Normalized whitespace for cleaner embedding.

### 3. **Embedding Model**

* **all-MiniLM-L6-v2** (Sentence Transformers)
* Fast, lightweight, high semantic accuracy.

### 4. **Vector Database**

* **ChromaDB (PersistentClient)**
* Provides fast and local semantic search.

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install fastapi uvicorn chromadb google-genai sentence-transformers
```

### 2. Set Google API Key

```bash
set GOOGLE_API_KEY=your_key_here
```

(Use `export` on macOS/Linux)

### 3. Run the Server

```bash
uvicorn app:app --reload
```

Then open:

```
http://127.0.0.1:8000/docs
```

---

## 📤 Example Request

```json
{
  "question": "How do I add a new page to mkdocs?",
  "k": 4
}
```

cURL example:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "How do I add a new page?", "k": 4}'
```

---

##  API Endpoints

### **GET /**

Check if API is running.

### **POST /ask**

Ask a question related to MkDocs.

| Field    | Type   | Description                |
| -------- | ------ | -------------------------- |
| question | string | User's question            |
| k        | int    | Number of retrieved chunks |

---

##  Sample Questions for Testing

* "How do I add a page to MkDocs navigation?"
* "How do I configure the mkdocs.yml file?"
* "What themes does MkDocs support?"
* "How do I deploy MkDocs on GitHub Pages?"

---

##  Notes

* Ensure ChromaDB contains embedded MkDocs documentation before querying.
* Higher `k` values may retrieve more context but could reduce answer focus.

---
Author : Zeina Elshenawy
---

## 👤 Author

Generated for **Zeina Elshenawy**
