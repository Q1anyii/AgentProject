
from pydantic import BaseModel

# 请求体模型：前端post json传 {"query":"你的问题","thread_id":"session_xxx"}
class ChatRequest(BaseModel):
    query: str
    thread_id: str  # 会话id，用来做langgraph checkpointer thread_id
