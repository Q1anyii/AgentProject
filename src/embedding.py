from langchain_text_splitters.character import RecursiveCharacterTextSplitter


from init import embed_model, embedding_function
import chromadb
from langchain_text_splitters import MarkdownHeaderTextSplitter

embed_model = embed_model
COLLECTION_NAME = "FAQ_KNOWLEDGE_BASE"

# 3. 创建 Chroma 客户端和集合
chroma_client = chromadb.PersistentClient(path="../recourses/chroma_db")
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function,  # 与 RAG 侧保持一致
    metadata={"hnsw:space": "cosine"}
)


with open("../recourses/FAQ/在线学习平台FAQ知识库（智能客服RAG专用）.md", "r", encoding="utf-8") as f:
    markdown_text = f.read()

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),  # 这就是我们需要的二级标题
    # ("###", "Header 3"), # 如果有需要，可以继续添加
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
parent_docs = splitter.split_text(markdown_text)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

child_docs = text_splitter.split_documents(parent_docs)
ids = [f"C{i}" for i in range(len(child_docs))]

metadatas = []
for doc in child_docs:
    meta = {"source": "智能客服", "category": "常见问题"}
    meta.update(doc.metadata)  # 合并 Header 1 / Header 2 标题信息
    metadatas.append(meta)

# embed_documents 需要的是字符串列表，不能直接传 Document 对象
vectors = embed_model.embed_documents([doc.page_content for doc in child_docs])

collection.upsert(
    ids=ids,
    documents=[doc.page_content for doc in child_docs],
    metadatas=metadatas,  # 与 ids 等长
)
count = collection.count()
print(count)
if count:
    print("信息嵌入成功")