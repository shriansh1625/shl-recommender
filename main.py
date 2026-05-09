"""
SHL Assessment Recommender — FastAPI Service
POST /chat   → conversational agent
GET  /health → readiness check
"""
import logging
import os

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="SHL Assessment Recommender", version="1.0.0")

# Agent is initialised once at startup (loads catalog + builds FAISS index)
_agent = None


@app.on_event("startup")
async def _startup():
    global _agent
    logger.info("Initialising SHL agent …")
    from agent import SHLAgent
    _agent = SHLAgent()
    logger.info("Agent ready.")


# ── Request / Response models ───────────────────────────────────────────────

class Message(BaseModel):
    role: str           # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_items=1)


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = []
    end_of_conversation: bool = False


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    result = _agent.respond(messages)
    return ChatResponse(**result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
