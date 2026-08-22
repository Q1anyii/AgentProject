from dataclasses import dataclass


@dataclass
class RetrievedDoc:
    """统一检索结果结构：屏蔽具体向量库（Chroma/Milvus）的返回格式差异。

    distance 统一语义：距离，越小越近（0 = 完全匹配）。
    """
    text: str
    distance: float
    metadata: dict
    id: str
