from typing import Annotated
from operator import add

from chromadb import QueryResult
from langchain_core.messages import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph.message import MessagesState
import chromadb
from langgraph.graph.state import StateGraph
from loguru import logger
from pydantic import Field
from rich import print as rprint
from init import model, system_prompt, embedding_function

client = chromadb.PersistentClient("../recourses/chroma_db")
collection = client.get_collection(
    name="FAQ_KNOWLEDGE_BASE",
    embedding_function=embedding_function,
)


class OverAllState(MessagesState):
    input_str: Annotated[str, add]
    retrieval_res: Annotated[QueryResult, "检索结果"]
    llm_output: Annotated[str, Field(description="模型输出")]

def retrieval_node(state: OverAllState) -> OverAllState:
    input_str = state["input_str"]

    retrieval_res = collection.query(
    query_texts=[input_str], # 查询文本
    n_results=3                     # 返回最相似的两个结果
    )
    return {
        "retrieval_res": retrieval_res
    }

def llm_node(state: OverAllState) -> OverAllState:

    retrieval_res = state["retrieval_res"]
    input_str = state["input_str"]

    docs = retrieval_res["documents"][0] if retrieval_res["documents"] else []
    distances = retrieval_res["distances"][0] if retrieval_res["distances"] else []

    # 只保留语义相关的检索结果（l2² < 1.0 约等于余弦相似度 > 0.5）
    filtered = [(d, s) for d, s in zip(docs, distances) if s < 1.0]
    if filtered:
        context = "\n\n".join(f"[文档 {i + 1}] {doc}" for i, (doc, _) in enumerate(filtered))
        logger.info(context)
    else:
        context = "（知识库中未检索到相关内容）"


    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"请严格依据下面检索到的资料回答用户问题，资料中没有的内容不要编造。\n\n"
                    f"【检索资料】\n{context}\n\n"
                    f"【用户问题】\n{input_str}"
        )
    ]

    response = model.invoke(messages)
    llm_output = response.content

    return {
        "llm_output": llm_output,
        "messages": [HumanMessage(content=input_str), AIMessage(content=llm_output)],

    }

builder = StateGraph(state_schema=OverAllState)

builder.add_node("retrieval_node", retrieval_node)
builder.add_node("llm_node", llm_node)
builder.add_edge(START, "retrieval_node")
builder.add_edge("retrieval_node", "llm_node")
builder.add_edge("llm_node", END)

checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": "123"}}
graph = builder.compile(checkpointer = checkpointer)

response = graph.invoke({"input_str": "咋买"},config = config)

rprint(response)
