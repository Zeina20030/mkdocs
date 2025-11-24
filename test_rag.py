from rag import get_answer

question = "How do I add a new page in MkDocs?"
answer = get_answer(question)
print("Question:", question)
print("Answer:", answer)
