from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import get_answer
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="MkDocs RAG System")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "MkDocs RAG system is running with Google Gemini!"}

@app.post("/ask")
def ask(query: Query):
    answer = get_answer(query.question)
    return {"answer": answer}
