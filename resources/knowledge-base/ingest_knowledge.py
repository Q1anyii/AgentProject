#!/usr/bin/env python3
"""
编程知识库向量库入库脚本

功能：遍历 knowledge-base 目录下的所有 .md 文件，调用项目的 EmbeddingProcessor 入库。
使用：cd src && python ../resources/knowledge-base/ingest_knowledge.py

注意：
- 需要先配置好 .env 中的 SILICONFLOW_API_KEY 和 SILICONFLOW_BASE_URL
- 入库会调用 embedding API，产生一定费用
- 重复入库会更新已有文档（基于内容哈希去重）
"""

import sys
from pathlib import Path

# 确保 src 目录在 Python 路径中
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from loguru import logger

# 知识库目录
KB_DIR = SCRIPT_DIR

# 文件分类映射：文件名前缀 -> category
CATEGORY_MAP = {
    "01": "python",
    "02": "fastapi",
    "03": "langgraph",
    "04": "rag",
    "05": "database",
    "06": "architecture",
    "07": "exception",
    "08": "security",
    "09": "frontend",
    "10": "engineering",
    "README": "knowledge_base_index",
}

# test-qa 子目录下的文件分类
QA_CATEGORY_MAP = {
    "01": "test_qa_basic",
    "02": "test_qa_debugging",
    "03": "test_qa_architecture",
    "04": "test_qa_badcase",
}


def get_category(file_path: Path) -> str:
    """根据文件路径确定 category。"""
    # test-qa 子目录
    if "test-qa" in file_path.parts:
        prefix = file_path.stem.split("-")[0]
        return QA_CATEGORY_MAP.get(prefix, "test_qa")

    # 根目录文件
    prefix = file_path.stem.split("-")[0]
    return CATEGORY_MAP.get(prefix, "knowledge_base")


def ingest_file(processor, file_path: Path) -> int:
    """入库单个文件，返回写入的文档条数。"""
    from vector.embedding import Meta

    category = get_category(file_path)
    source = "knowledge_base"
    meta = Meta(source=source, category=category)

    try:
        count = processor.embed(file_path, meta)
        logger.success(f"[OK] {file_path.name} -> category={category}, 共 {count} 条")
        return count
    except Exception as e:
        logger.error(f"[FAIL] {file_path.name} 入库失败: {e}")
        return 0


def main():
    """主函数：遍历所有 .md 文件并入库。"""
    logger.info("=" * 60)
    logger.info("编程知识库向量库入库开始")
    logger.info(f"知识库目录: {KB_DIR}")
    logger.info("=" * 60)

    # 初始化 EmbeddingProcessor
    try:
        from vector.embedding import EmbeddingProcessor
        processor = EmbeddingProcessor()
        logger.info(f"向量库初始化成功")
    except Exception as e:
        logger.error(f"向量库初始化失败: {e}")
        logger.error("请检查 vector_db.json 配置和网络连接")
        sys.exit(1)

    # 收集所有 .md 文件
    md_files = sorted(KB_DIR.rglob("*.md"))
    # 排除本脚本和 README（可选，README 也可以入库）
    md_files = [f for f in md_files if f.name != "ingest_knowledge.py"]

    logger.info(f"找到 {len(md_files)} 个 Markdown 文件")
    logger.info("-" * 60)

    # 逐个入库
    total_docs = 0
    success_count = 0
    fail_count = 0

    for i, file_path in enumerate(md_files, 1):
        logger.info(f"[{i}/{len(md_files)}] 处理: {file_path.name}")
        count = ingest_file(processor, file_path)
        if count > 0:
            total_docs += count
            success_count += 1
        else:
            fail_count += 1

    # 汇总
    logger.info("=" * 60)
    logger.info("入库完成")
    logger.info(f"  成功文件: {success_count}/{len(md_files)}")
    logger.info(f"  失败文件: {fail_count}")
    logger.info(f"  总文档数: {total_docs}")
    logger.info(f"  集合总数: {processor.count()}")
    logger.info("=" * 60)

    if fail_count > 0:
        logger.warning(f"有 {fail_count} 个文件入库失败，请检查日志")
        sys.exit(1)


if __name__ == "__main__":
    main()
