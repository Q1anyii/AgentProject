import os

import pymysql
import requests
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain.chat_models.base import init_chat_model
from FlagEmbedding import FlagReranker
from langchain_community.document_loaders.text import TextLoader
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import OpenAIEmbeddings
from langgraph.checkpoint.postgres import PostgresSaver
from dotenv import load_dotenv
from pymysql.cursors import DictCursor
from constant.embedding_constants import COLLECTION_NAME

load_dotenv(override=True)

model = init_chat_model(
    model="deepseek-v4-flash",  # 指定混元模型，如 hunyuan-turbos-latest[reference:4]
    model_provider="openai",  # 关键：使用 OpenAI 兼容模式
    api_key=os.getenv("HUNYUAN_API_KEY"),  # 你的混元 API Key
    base_url="https://tokenhub.tencentmaas.com/v1",  # 混元的 Base URL[reference:5]
)

embed_model = OpenAIEmbeddings(
    model="BAAI/bge-m3", # 免费模型 ID: BAAI/bge-m3
    base_url=os.getenv("SILICONFLOW_BASE_URL"),
    api_key=os.getenv("SILICONFLOW_API_KEY")
)

embedding_function = OpenAIEmbeddingFunction(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    api_base=os.getenv("SILICONFLOW_BASE_URL"),
    model_name="BAAI/bge-m3"
)
current_dir = os.path.dirname(os.path.abspath(__file__))
prompt_file = os.path.join(
    current_dir,
    "..",
    "resources",
    "system_prompt",
    "default_system_prompt.txt"
)
loader = TextLoader(prompt_file, encoding="utf-8")
system_prompt = loader.load()[0].page_content
def get_effective_system_prompt(base_prompt: str, user_id: str, store) -> str:
    """组装用户级 system prompt：基础 + 用户全局自定义"""
    item = store.get(("user_global", user_id), "custom_prompt")
    if item and item.value.get("content"):
        return f"{base_prompt}\n\n【用户自定义设定】\n{item.value['content']}"
    return base_prompt


def online_rerank(query: str, documents: list[str], top_n: int = 10) -> list[dict]:
    """调用 SiliconFlow 在线重排，返回按相关性降序的 [{index, relevance_score}, ...]"""
    resp = requests.post(
        f"{os.getenv('SILICONFLOW_BASE_URL')}/rerank",
        headers={"Authorization": f"Bearer {os.getenv('SILICONFLOW_API_KEY')}"},
        json={
            "model": "BAAI/bge-reranker-v2-m3",
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return sorted(resp.json()["results"], key=lambda r: r["relevance_score"], reverse=True)


class CustomPostgresSaver(PostgresSaver):
    def list(
        self,
        config: RunnableConfig | None = None,
        *,
        thread_id: str | None = None,   # 扩展：便捷定位单线程（等价于 config 传 thread_id）
        user_id: str | None = None,     # 扩展：按 metadata 内 user_id 过滤会话
        filter: dict | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ):
        """兼容父类签名，并扩展支持 thread_id / user_id 便捷过滤。

        必须保持与 PostgresSaver.list(config, *, filter, before, limit) 签名兼容：
        LangGraph 内部及既有调用会按父类签名传参，签名不兼容会直接 TypeError。
        """
        if filter is None:
            filter = {}
        # 如果传入 user_id，合并进 filter（数据库层 metadata @> '{"user_id": ...}' 过滤）
        if user_id is not None:
            filter["user_id"] = user_id
        # 便捷参数：合并进 config，等价于 {"configurable": {"thread_id": thread_id}}
        if thread_id is not None:
            if config is None:
                config = {}
            config.setdefault("configurable", {})["thread_id"] = thread_id
        return super().list(config, filter=filter, before=before, limit=limit)

if __name__ =="__main__":
    resp = model.invoke("简单介绍一下自己")
    print(resp.content)


# ============================================================
# 用户级 System Prompt 组装
# 作用：基础 system_prompt + 用户自定义内容（从 MySQL user_profile 表读取）
# 调用方：main_graph.py llm_node，每次对话时按 user_id 动态组装
# ============================================================

def get_user_system_prompt(user_id: str, base_prompt: str = None) -> str:
    """组装用户级 system prompt：基础默认 + 用户自定义内容。

    Args:
        user_id: 用户 ID
        base_prompt: 基础 system prompt，默认使用模块级 system_prompt

    Returns:
        组装后的 system prompt 字符串
    """
    if base_prompt is None:
        base_prompt = system_prompt

    # 延迟导入避免循环依赖
    try:
        from service.user_profile_service import user_profile_service
        custom_prompt = user_profile_service.get_system_prompt(user_id)
        if custom_prompt and custom_prompt.strip():
            return f"{base_prompt}\n\n【用户自定义设定】\n{custom_prompt.strip()}"
    except Exception as e:
        # 获取用户自定义内容失败时，降级使用基础 prompt，不影响对话
        from loguru import logger
        logger.warning(f"获取用户自定义 system prompt 失败 user_id={user_id}: {e}，使用基础 prompt")

    return base_prompt