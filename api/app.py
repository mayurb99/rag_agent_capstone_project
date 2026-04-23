import httpx

import os
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from db import init_db
from agent import agent_loop
from logger import log

# In-memory session store (keyed by patient_id / session_id)
# For demo: single hardcoded user. Expand to dict keyed by user_id for multi-user.
SESSION_STORE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Medical Assistant API", lifespan=lifespan)

RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag-service:8001")


class ChatRequest(BaseModel):
    patient_id: str = "demo_user"
    message: str


class ChatResponse(BaseModel):
    response: str


def rag_answer(query: str) -> str:
    """Call the RAG microservice synchronously."""
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(f"{RAG_SERVICE_URL}/query", params={"q": query})
            resp.raise_for_status()
            return resp.json().get("answer", "No answer returned.")
    except Exception as e:
        log({"error": f"RAG service call failed: {str(e)}"})
        return "I couldn't retrieve medical information right now. Please try again."


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    log({"step": "chat_start", "input": req.message})
    session_key = req.patient_id
    session_state = SESSION_STORE.get(session_key, {})
    log({"step": "before_agent_loop"})


    response_text, updated_state = agent_loop(
        user_input=req.message,
        session_state=session_state,
        rag_answer_fn=rag_answer,
    )
    log({"step": "after_agent_loop", "response": response_text})

    SESSION_STORE[session_key] = updated_state
    log({"patient": req.patient_id, "message": req.message, "response": response_text})
    return ChatResponse(response=response_text)


@app.get("/appointments/{patient_id}")
def list_appointments(patient_id: int):
    from tools import get_appointments
    return {"appointments": get_appointments(patient_id, raw=True)}


@app.get("/doctors")
def list_doctors(specialty: str = None):
    from tools import get_doctors
    return {"doctors": get_doctors(specialty=specialty)}


@app.get("/health")
def health():
    return {"status": "ok"}
