from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional

# 请求体模型：前端post json传 {"query":"你的问题","thread_id":"session_xxx","file_ids":[1,2]}
class ChatRequest(BaseModel):
    query: str
    thread_id: str  # 会话id，用来做langgraph checkpointer thread_id
    file_ids: Optional[List[int]] = None  # 上传文件 ID 列表，解析内容拼接到 query 传入 LLM
