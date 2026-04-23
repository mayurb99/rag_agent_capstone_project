from fastapi import FastAPI
from rag_pipeline import get_rag_answer

app = FastAPI(title="RAG Service")


@app.get("/query")
def query(q: str):
    answer = get_rag_answer(q)
    return {"answer": answer}


@app.get("/health")
def health():
    return {"status": "ok"}
