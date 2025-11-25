from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import get_answer
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="MkDocs RAG System")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Allow all origins for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model with adjustable k
class Query(BaseModel):
    question: str
    k: int = 5  # default number of context chunks to retrieve

@app.get("/")
def root():
    return {"message": "MkDocs RAG system is running with Google Gemini!"}

@app.post("/ask")
def ask(query: Query):
    # Validate k
    if query.k <= 0:
        raise HTTPException(
            status_code=400,
            detail="Parameter 'k' must be a positive integer."
        )
    
    # Call get_answer with validated k
    answer = get_answer(query.question, n_results=query.k)
    
    return {
        "answer": answer,
        "used_k": query.k
    }
