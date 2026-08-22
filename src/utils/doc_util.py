from typing import List, Any, Dict
from loguru import logger
from langchain_core.documents import Document

def unpack_query_results(results: List[Any]) -> List[List[Document]]:
    """
    将 List[QueryResult] 拆解为 List[List[Document]]，
    每个内层列表对应一个查询的文档列表。
    """
    all_query_docs = []

    for result in results:
        # 防御：某些版本/配置下 ChromaDB 会再包一层 list，先解包
        while isinstance(result, list) and len(result) == 1:
            result = result[0]

        if not isinstance(result, dict):
            logger.warning(
                f"Unexpected query result type: {type(result)}, value: {result[:200] if isinstance(result, (list, str)) else result}")
            all_query_docs.append([])
            continue

        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        # ChromaDB 返回的是 batch 嵌套结构：List[List[...]]
        if ids and isinstance(ids[0], list):
            for sub_ids, sub_docs, sub_metas in zip(ids, documents, metadatas):
                query_docs = []
                for i, doc_id in enumerate(sub_ids):
                    meta = dict(sub_metas[i]) if i < len(sub_metas) else {}
                    meta["id"] = doc_id
                    text = sub_docs[i] if i < len(sub_docs) else ""
                    query_docs.append(Document(page_content=text, metadata=meta))
                all_query_docs.append(query_docs)
        else:
            # 兜底：一维结构
            query_docs = []
            for i, doc_id in enumerate(ids):
                meta = dict(metadatas[i]) if i < len(metadatas) else {}
                meta["id"] = doc_id
                text = documents[i] if i < len(documents) else ""
                query_docs.append(Document(page_content=text, metadata=meta))
            all_query_docs.append(query_docs)

    return all_query_docs

def documents_to_dicts(docs: List[Any]) -> List[Dict[str, Any]]:
    # 兼容两种形态：langchain Document（page_content）与 vector.RetrievedDoc（text）
    return [
        {
            "page_content": getattr(doc, "page_content", None) or getattr(doc, "text", ""),
            "metadata": doc.metadata,
        }
        for doc in docs
    ]

def dict_to_documents(dictionary: list[Dict[Any, Any]]) -> List[Document]:
    try:
        doc_list = [
            Document(page_content=item["page_content"], metadata=item["metadata"])
            for item in dictionary
        ]
        return doc_list
    except (AttributeError, KeyError, TypeError) as e:
        logger.warning(f"缓存文档反序列化失败：{dictionary}，错误：{e}")
        return []