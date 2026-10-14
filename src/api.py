"""FastAPI 推荐服务"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from agent import chat

load_dotenv()

app = FastAPI(title="电影推荐 Agent API")


class ChatRequest(BaseModel):
    message: str
    user_id: Optional[int] = None


class ChatResponse(BaseModel):
    response: str
    user_id: Optional[int] = None


@app.get("/")
def root():
    return {"message": "电影推荐 Agent API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """对话接口"""
    try:
        response = chat(request.message, request.user_id)
        return ChatResponse(response=response, user_id=request.user_id)
    except Exception as e:
        return ChatResponse(response=f"错误: {str(e)}", user_id=request.user_id)


@app.get("/recommend/{user_id}")
def recommend_endpoint(user_id: int):
    """推荐接口"""
    try:
        response = chat(f"为我推荐电影", user_id)
        return {"user_id": user_id, "response": response}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
