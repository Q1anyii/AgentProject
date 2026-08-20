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
from load_dotenv import load_dotenv
from pymysql.cursors import DictCursor

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
    "recourses",
    "system_prompt",
    "online_learning_platform_customer_temp.txt"
)
loader = TextLoader(prompt_file, encoding="utf-8")
system_prompt = loader.load()[0].page_content

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

COLLECTION_NAME = "FAQ_KNOWLEDGE_BASE"

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