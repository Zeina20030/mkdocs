# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from rag import get_answer

app = FastAPI()

class Query(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "MkDocs RAG system is running!"}

@app.post("/ask")
def ask(query: Query):
    answer = get_answer(query.question)
    return {"answer": answer}
