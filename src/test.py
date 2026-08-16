# src/check_db.py
from init import embedding_function
import chromadb

client = chromadb.PersistentClient("../recourses/chroma_db")
collection = client.get_collection(name="FAQ_KNOWLEDGE_BASE", embedding_function=embedding_function)

result = collection.get(include=["documents", "metadatas"])
print(f"共 {len(result['ids'])} 个块\n")
for doc_id, doc, meta in zip(result["ids"], result["documents"], result["metadatas"]):
    print(f"--- {doc_id} | {meta.get('Header 1', '')} / {meta.get('Header 2', '')}")
    print(doc[:60].replace("\n", " "), "\n")