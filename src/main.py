from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from loguru import logger
from graph import  ChatService
from fastapi import FastAPI
from schemas import ChatRequest

from dotenv import load_dotenv
load_dotenv()

chat_service = ChatService()

@asynccontextmanager
async def lifespan(app: FastAPI):

    # ===== 启动阶段：yield 之前 =====
    logger.info("正在初始化 LangGraph 资源...")
    chat_service.open()

    logger.info("资源初始化完成")
    yield                                # ===== 应用运行期间（yield 挂起）=====
    # ===== 关闭阶段：yield 之后 =====
    logger.info("正在释放资源...")
    chat_service.close(timeout=10)                         # 替代现在的 finally 逻辑
    logger.info("资源已释放")

app = FastAPI(title="智能AI客服", lifespan=lifespan)

@app.post("/api/chat/")
async def chat(request_body: ChatRequest):
    query = request_body.query
    thread_id = request_body.thread_id
    answer = await chat_service.a_invoke("user_001", thread_id, query)
    return {"thread_id": thread_id, "answer": answer}


@app.get("/api/chat/{thread_id}/history")
def get_history_session(thread_id: str):
    history_session = chat_service.get_history_session(thread_id)
    return history_session

@app.get("/api/users/{user_id}/memory")
def get_memory(user_id: str):
    memory = chat_service.get_memory(user_id)
    return memory

@app.delete("/api/chat/{thread_id}")
def delete_session_by_id(thread_id: str):
    success = chat_service.delete_session_by_id(thread_id)
    return success

@app.get("/api/users/{user_id}/sessions")
def get_sessions_by_user_id(user_id: str):
    sessions = chat_service.get_user_sessions(user_id)
    return sessions

@app.get("/health")
def check_db_health():
    status = chat_service.check_db_health()
    return status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)