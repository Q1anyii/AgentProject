import asyncio
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from loguru import logger
import hashlib

from config import load_vector_db_config
from constant.embedding_constants import CHUNK_SIZE, CHUNK_OVERLAP
from vector.vector_store import create_vector_store


class Meta:
    """文档元数据：来源（source）与分类（category）。"""

    def __init__(self, source: str, category: str):
        self.source = source
        self.category = category

def compute_doc_hash_with_meta(docs: list[Document]) -> list[str]:

    ids = []
    for doc in docs:
        content = doc.page_content.strip().replace("\r\n", "\n").replace("\r", "\n")
        # 挑选需要参与哈希的元数据，不要全部meta（时间、随机字段不要混入）
        meta_keys = ["source", "file_name", "url"]
        meta_part = []
        for k in meta_keys:
            meta_part.append(f"{k}:{doc.metadata.get(k, '')}")

        raw = content + "\n" + "\n".join(meta_part)
        ids.append(hashlib.sha256(raw.encode("utf‑8")).hexdigest())
    return ids


"""
EmbeddingProcessor（类）
├── __init__()                     # 通过 create_vector_store 工厂创建向量库（复用 init.embedding_function）
│
├── 静态方法（纯函数）
│   ├── doc_2_str(docs)           # 文档拼接
│   ├── load_docs(file_path)      # 按后缀解析 .md/.txt/.pdf
│   └── split_docs(file_path)     # 300/50 切分
│
├── 同步入口
│   ├── embed(file_path, meta) -> int    # 解析→切分→入库→返回条数
│   └── count() -> int                    # 当前集合总数
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

    常量（COLLECTION_NAME/CHUNK_SIZE/CHUNK_OVERLAP）已移至 constant/embedding_constants.py 统一管理。
    """

    def __init__(self):
        # 工厂内部会延迟导入 init.embedding_function（init.py 初始化模型等重资源）
        self.vector_store = create_vector_store(load_vector_db_config())

    # ---------- 文档解析与切分（纯函数） ----------

    @staticmethod
    def doc_2_str(docs) -> str:
        """拼接多个文档为单个字符串。"""
        return "".join(content.page_content for content in docs)

    @staticmethod
    def load_docs(file_path) -> list[Document]:
        """按后缀加载文档：.md 按标题切分，.txt/.pdf 原样加载。"""
        # suffix 转小写，避免 .PDF 大写不匹配 case 分支
        suffix = Path(file_path).suffix.lower()
        parent_docs: list[Document] = []

        match suffix:
            case ".md":
                # 注意：不能用 UnstructuredMarkdownLoader——它会解析掉 markdown 语法（# 标题 → 纯文本），
                # 导致后续 MarkdownHeaderTextSplitter 找不到 # 标记，切分结果为空（入库 0 条）。
                # 改用 TextLoader 读取原始 markdown 文本，保留 # 标题标记。
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()
                headers_to_split_on = [
                    ("#", "Header 1"),
                    ("##", "Header 2"),  # 二级标题作为分块边界
                ]
                document = EmbeddingProcessor.doc_2_str(docs)
                splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
                parent_docs = splitter.split_text(document)
                # 兜底：文档无 # / ## 标题时，按标题切分结果为空，直接用原始文档
                if not parent_docs:
                    logger.warning(f"{Path(file_path).name} 无 # / ## 标题，跳过标题切分，直接按 chunk 切分")
                    parent_docs = docs
            case ".txt":
                parent_docs = TextLoader(str(file_path), encoding="utf-8").load()
            case ".pdf":
                # PyPDFLoader 要求字符串路径，Path 对象在某些版本下会报 ValueError
                loader = PyPDFLoader(str(file_path))
                parent_docs = loader.load()
            case _:
                logger.error(f"暂不支持解析该类文档: {suffix}")

        return parent_docs

    @staticmethod
    def split_docs(file_path) -> list[Document] | None:
        """加载后按 chunk 大小切分为子文档。"""
        try:
            parent_docs = EmbeddingProcessor.load_docs(file_path=file_path)
            if not parent_docs:
                logger.warning(f"{Path(file_path).name} 加载结果为空，可能是不支持的格式或文件损坏")
                return None
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
            return text_splitter.split_documents(parent_docs)
        except FileNotFoundError as e:
            logger.error(f"路径文件不存在：{e}")
            return None
        except Exception as e:
            # PyPDFLoader 加载 PDF 失败时抛 ValueError/PdfReadError，
            # 统一捕获避免单文件失败导致整个入库脚本崩溃
            logger.error(f"文档解析失败 [{Path(file_path).name}]: {type(e).__name__}: {e}")
            return None

    # ---------- 同步入口 ----------

    def embed(self, file_path: str | Path, meta: Meta) -> int:
        """解析文档并写入向量库，返回写入的文档条数。"""
        child_docs = self.split_docs(file_path=file_path)
        if not child_docs:
            logger.warning("没有可写入的文档，跳过入库")
            return 0

        ids = compute_doc_hash_with_meta(child_docs)

        metadatas = []
        for doc in child_docs:
            metadata = {"source": meta.source, "category": meta.category}
            metadata.update(doc.metadata)  # 合并 Header 1 / Header 2 标题信息
            metadatas.append(metadata)

        self.vector_store.upsert(
            ids=ids,
            documents=[doc.page_content for doc in child_docs],
            metadatas=metadatas,  # 与 ids 等长
        )
        # 返回本次写入的条数，而非集合总数
        # 注意：Milvus 的 get_collection_stats 在 upsert 后有异步 flush 延迟，
        # 立即调用 count() 可能返回 0，因此用 len(child_docs) 作为本次写入数
        written = len(child_docs)
        logger.info(f"信息嵌入成功，本次写入 {written} 条")
        return written

    def count(self) -> int:
        """当前向量集合文档总数（ingest_knowledge.py 汇总用）。"""
        return self.vector_store.count()

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
