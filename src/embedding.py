import asyncio
from pathlib import Path

import chromadb
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from loguru import logger

class Meta:
    """文档元数据：来源（source）与分类（category）。"""

    def __init__(self, source: str, category: str):
        self.source = source
        self.category = category

"""
EmbeddingProcessor（类）
├── __init__(persist_path)        # 连接 Chroma，复用 init.embedding_function（延迟导入防循环）
│
├── 静态方法（纯函数）
│   ├── doc_2_str(docs)           # 文档拼接
│   ├── load_docs(file_path)      # 按后缀解析 .md/.txt/.pdf
│   └── split_docs(file_path)     # 300/50 切分
│
├── 同步入口
│   └── embed(file_path, meta) -> int    # 解析→切分→入库→返回条数
│
└── 异步协程入口
    ├── aembed(...)               # async 版 embed
    ├── aload_docs(...)           # async 版 load_docs
    └── asplit_docs(...)          # async 版 split_docs
"""

class EmbeddingProcessor:
    """文档解析 → 切分 → 向量化入库。

    提供同步（embed）与异步（aembed）两套入口：
    协程内部用 asyncio.to_thread 托管阻塞调用，避免卡住事件循环。
    """

    COLLECTION_NAME = "FAQ_KNOWLEDGE_BASE"
    CHUNK_SIZE = 300
    CHUNK_OVERLAP = 50

    def __init__(self, persist_path: str | Path = "../recourses/chroma_db"):
        # 延迟导入：init.py 会初始化模型等重资源
        from init import embedding_function

        self.embedding_function = embedding_function
        self.client = chromadb.PersistentClient(path=str(persist_path))
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_function,  # 与 RAG 侧保持一致
            metadata={"hnsw:space": "cosine"},
        )

    # ---------- 文档解析与切分（纯函数） ----------

    @staticmethod
    def doc_2_str(docs) -> str:
        """拼接多个文档为单个字符串。"""
        return "".join(content.page_content for content in docs)

    @staticmethod
    def load_docs(file_path) -> list[Document]:
        """按后缀加载文档：.md 按标题切分，.txt/.pdf 原样加载。"""
        suffix = Path(file_path).suffix  # 带点后缀，如 ".md"，与 case 分支对齐
        parent_docs: list[Document] = []

        match suffix:
            case ".md":
                loader = UnstructuredMarkdownLoader(file_path)
                docs = loader.load()
                headers_to_split_on = [
                    ("#", "Header 1"),
                    ("##", "Header 2"),  # 二级标题作为分块边界
                ]
                document = EmbeddingProcessor.doc_2_str(docs)
                splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
                parent_docs = splitter.split_text(document)
            case ".txt":
                parent_docs = TextLoader(file_path).load()
            case ".pdf":
                parent_docs = PyPDFLoader(file_path).load()
            case _:
                logger.error("暂不支持解析该类文档")

        return parent_docs

    @staticmethod
    def split_docs(file_path) -> list[Document] | None:
        """加载后按 chunk 大小切分为子文档。"""
        try:
            parent_docs = EmbeddingProcessor.load_docs(file_path=file_path)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=EmbeddingProcessor.CHUNK_SIZE,
                chunk_overlap=EmbeddingProcessor.CHUNK_OVERLAP,
            )
            return text_splitter.split_documents(parent_docs)
        except FileNotFoundError as e:
            logger.error(f"路径文件不存在：{e}")
            return None

    # ---------- 同步入口 ----------

    def embed(self, file_path: str | Path, meta: Meta) -> int:
        """解析文档并写入向量库，返回写入的文档条数。"""
        child_docs = self.split_docs(file_path=file_path)
        if not child_docs:
            logger.warning("没有可写入的文档，跳过入库")
            return 0

        ids = [f"C{i}" for i in range(len(child_docs))]
        metadatas = []
        for doc in child_docs:
            metadata = {"source": meta.source, "category": meta.category}
            metadata.update(doc.metadata)  # 合并 Header 1 / Header 2 标题信息
            metadatas.append(metadata)

        self.collection.upsert(
            ids=ids,
            documents=[doc.page_content for doc in child_docs],
            metadatas=metadatas,  # 与 ids 等长
        )
        count = self.collection.count()
        if count:
            logger.info(f"信息嵌入成功，当前集合共 {count} 条")
        return count

    # ---------- 异步协程入口 ----------

    async def aembed(self, file_path: str | Path, meta: Meta) -> int:
        """异步版 embed：阻塞操作放入线程池，不阻塞事件循环。"""
        return await asyncio.to_thread(self.embed, file_path, meta)

    async def aload_docs(self, file_path) -> list[Document]:
        """异步版 load_docs。"""
        return await asyncio.to_thread(self.load_docs, file_path)

    async def asplit_docs(self, file_path) -> list[Document] | None:
        """异步版 split_docs。"""
        return await asyncio.to_thread(self.split_docs, file_path)



async def main():
    """异步 CLI 入口：输入文档路径与元数据后入库。"""
    while True:
        try:
            file_path = input("输入需要解析的文档路径: ").strip()
            source, category = input("输入 source 与 category（空格分隔）: ").split()
            meta = Meta(source, category)

            processor = EmbeddingProcessor()
            count = await processor.aembed(file_path=file_path, meta=meta)
            check = input(f"嵌入完成，共 {count} 条,是否继续嵌入？(y/n):")
            if check == "y" :
                continue
            elif check == "n":
                break
            else:
                logger.error("错误选择,视为推出")
                break
        except ValueError as e:
            logger.error(f"路径或分类格式出错，重新输入{e}\n")
            continue


if __name__ == "__main__":
    asyncio.run(main())
