from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from loguru import logger
from graph import build_chat_graph
from fastapi import FastAPI
from schemas import ChatRequest

from dotenv import load_dotenv
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ===== 启动阶段：yield 之前 =====
    logger.info("正在初始化 LangGraph 资源...")
    graph, pool = build_chat_graph()          # 建连接池 → setup() 迁移 → compile
    app.state.graph = graph              # 关键：资源挂到 app.state
    app.state.pool = pool
    logger.info("资源初始化完成")
    yield                                # ===== 应用运行期间（yield 挂起）=====
    # ===== 关闭阶段：yield 之后 =====
    logger.info("正在释放资源...")
    pool.close(timeout=10)                         # 替代现在的 finally 逻辑
    logger.info("资源已释放")

app = FastAPI(title="智能AI客服", lifespan=lifespan)

from fastapi import Request

#  bug修复(1.新建会话仍读取旧数据，2.刷新页面会话消失)


@app.post("/api/chat/")
def chat(request: Request, request_body: ChatRequest):
    graph = request.app.state.graph
    query = request_body.query
    thread_id = request_body.thread_id

    # 演示用固定用户 ID；thread_id 决定短期记忆（会话内），user_id 决定长期记忆（跨会话）
    user_id = "user_001"

    config = {
        "configurable": {"thread_id": thread_id, "user_id": user_id},
        "metadata": {"user_id": user_id},  # 随 checkpoint 写入 metadata
    }

    result = graph.invoke({"input_str": query}, config=config)
    ai_msg = result["messages"][-1]
    return {"thread_id": thread_id, "answer": ai_msg.content}


@app.get("/api/chat/{thread_id}/history")
def get_history_session(request: Request, thread_id: str):
    graph = request.app.state.graph
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    snapshot = graph.get_state(config)
    if not snapshot or len(snapshot) == 0:
        logger.error(f"会话:{thread_id}记录不存在")
        return {"code": 404, "message": f"会话:{thread_id}记录不存在"}

    state_data = snapshot.values
    history_messages = state_data.get("messages",[])
    if not history_messages:
        logger.error(f"会话:{thread_id}记录不存在")
        return []
    history_msg = []
    for message in history_messages:
        history_msg.append(
            {
              "role": f"{message.type}",
              "content": message.content
            }
        )
    return history_msg


@app.get("/api/users/{user_id}/memory")
def get_memory(request: Request, user_id: str):
    pool = request.app.state.pool
    store = PostgresStore(pool)
    memory = ""
    item = store.get(("rag_chat", user_id), "user_profile")
    if item and item.value.get("profile"):
        memory = item.value["profile"]
    return memory


@app.delete("/api/chat/{thread_id}")
def delete_session_by_id(request: Request, thread_id: str):
    pool = request.app.state.pool
    checkpointer = PostgresSaver(pool)
    history_msg = get_history_session(request, thread_id)
    if not history_msg:
        logger.error(f"会话:{thread_id}记录不存在")
        return f"会话:{thread_id}记录不存在"
    checkpointer.delete_thread(thread_id)
    logger.info(f"删除会话:{thread_id}成功")
    return {
        "title": f"删除会话:{thread_id}成功",
    }

@app.get("/api/users/{user_id}/sessions")
def get_sessions_by_user_id(request: Request, user_id: str):
    pool = request.app.state.pool
    checkpointer = PostgresSaver(pool)
    # 1. 列出所有 checkpoint（跨所有 thread），按线程分组取最新
    from langgraph.checkpoint.base import CheckpointTuple
    latest_by_thread: dict[str, CheckpointTuple] = {}
    for item in checkpointer.list(None):  # config=None 列出全部
        tid = item.config["configurable"]["thread_id"]
        if tid not in latest_by_thread:
            latest_by_thread[tid] = item  # list() 按 checkpoint_id 升序返回

    sessions = []
    for tid, item in latest_by_thread.items():
        messages = item.checkpoint["channel_values"].get("messages", [])
        first_user = next((m for m in messages if m.type == "human"), None)
        sessions.append({
            "thread_id": tid,
            "title": first_user.content[:20] if first_user else "新会话",
            "last_updated": item.checkpoint["ts"],
        })
    sessions.sort(key=lambda s: s["last_updated"], reverse=True)
    return sessions


import psycopg
from fastapi import Request
from psycopg_pool import PoolTimeout

@app.get("/health")
def check_db_health(request: Request):
    pool = request.app.state.pool
    try:
        with pool.connection() as conn:      # 从池中借连接（空闲不足会抛 PoolTimeout）
            conn.execute("SELECT 1")          # 真正发一条查询验证链路
        return {"status": "ok", "db": True}
    except PoolTimeout:
        logger.warning("数据库连接池已满或无法建立连接")
        return {"status": "degraded", "db": False}
    except psycopg.OperationalError as e:     # psycopg3 的异常就在 psycopg 顶层
        logger.error(f"数据库不可用: {e}")
        return {"status": "degraded", "db": False}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)