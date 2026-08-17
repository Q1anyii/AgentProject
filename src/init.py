import os

import requests
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain.chat_models.base import init_chat_model
from FlagEmbedding import FlagReranker
from langchain_community.document_loaders.text import TextLoader
from langchain_openai import OpenAIEmbeddings
from load_dotenv import load_dotenv

load_dotenv(override=True)

model = init_chat_model(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("BASE_URL"),
    extra_body={
        "thinking":{
            "type": "disabled"
        }
    }
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

loader = TextLoader("../recourses/prompt/system_prompt.txt", encoding="utf-8")
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