"""
RAGAS 评估脚本：对 RAG 检索+生成链路进行自动化质量评估。

用法：
    cd src
    python -m test.ragas_eval                    # 评估全部 45 条
    python -m test.ragas_eval --limit 10         # 只评估前 10 条
    python -m test.ragas_eval --category basic   # 只评估基础概念类
    python -m test.ragas_eval --threshold 0.5    # 自定义距离阈值（默认0.5）

输出：
    - test/ragas_report.csv          逐条评分明细
    - test/ragas_summary.json        汇总统计（含按分类分组）

评估指标：
    - context_precision : 检索结果中相关片段的排名精度
    - context_recall    : 标准答案信息在检索结果中的覆盖率
    - faithfulness      : 生成答案是否忠实于上下文（幻觉率反向指标）
    - answer_relevancy  : 答案与问题的相关性
    - answer_correctness: 答案与标准答案的语义一致度
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# 必须在 import 项目模块前把 src 加入 sys.path
SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv
load_dotenv(SRC_DIR.parent / ".env")

from datasets import Dataset
from loguru import logger
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import load_vector_db_config
from constant.retrieval_constants import TOP_K, RRF_K, REWRITE_PROMPT
from vector.vector_store import create_vector_store, RetrievedDoc
from init import model, online_rerank


# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = SRC_DIR.parent
DATASET_PATH = PROJECT_ROOT / "resources" / "knowledge-base" / "test-qa" / "eval_dataset.json"
REPORT_CSV = SRC_DIR / "test" / "ragas_report.csv"
SUMMARY_JSON = SRC_DIR / "test" / "ragas_summary.json"

# 分类关键词映射（用于 --category 过滤）
CATEGORY_MAP = {
    "basic": "基础概念",
    "debug": "代码调试",
    "architecture": "架构设计",
    "badcase": "刁钻 Badcase",
}


def load_test_dataset(limit: int = 0, category: str = "") -> list[dict]:
    """加载 JSON 测试集，支持按条数和分类过滤。"""
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)

    if category:
        keyword = CATEGORY_MAP.get(category, category)
        data = [d for d in data if keyword in d.get("category", "")]

    if limit > 0:
        data = data[:limit]

    logger.info(f"加载测试集 {len(data)} 条" + (f"（分类: {category}）" if category else ""))
    return data


def extract_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON（容错：去除 markdown 代码块包裹）。"""
    text = text.strip()
    # 去除 ```json ... ``` 包裹
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return json.loads(text[start:end + 1])
    raise json.JSONDecodeError("No JSON object found", text, 0)


def rewrite_queries(question: str) -> list[str]:
    """Query 改写：生成主查询+子查询，容错处理 LLM 返回格式不一致的情况。"""
    prompt = REWRITE_PROMPT.format(question=question, history="无")
    try:
        resp = model.invoke(prompt, response_format={"type": "json_object"})
        raw = extract_json(resp.content)
        # 兼容多种返回格式
        if "主查询" in raw:
            main_q = raw["主查询"]
            sub_qs = raw.get("子查询", [])
        elif "main_query" in raw:
            main_q = raw["main_query"]
            sub_qs = raw.get("sub_queries", [])
        elif "queries" in raw and isinstance(raw["queries"], list):
            # LLM 有时直接返回 {"queries": [...]}
            qs = raw["queries"]
            main_q = qs[0] if qs else question
            sub_qs = qs[1:] if len(qs) > 1 else []
        else:
            main_q = question
            sub_qs = []
        queries = [main_q] + sub_qs
        logger.info(f"  改写: {queries}")
        return queries
    except Exception as e:
        logger.warning(f"  Query 改写失败，用原始问题: {e}")
        return [question]


def rrf_fusion(results: list[list[RetrievedDoc]], k: int = RRF_K) -> list[RetrievedDoc]:
    """Reciprocal Rank Fusion：多查询结果融合去重。"""
    scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            key = doc.id or doc.text
            if key not in scores:
                scores[key] = {"doc": doc, "score": 0.0}
            scores[key]["score"] += 1.0 / (k + rank + 1)
    return [item["doc"] for item in sorted(scores.values(), key=lambda x: x["score"], reverse=True)]


def run_rag(vector_store, question: str, threshold: float) -> tuple[str, list[str]]:
    """对单个问题执行 改写→检索→RRF→重排→生成，返回 (answer, contexts)。"""
    # 1. Query 改写
    queries = rewrite_queries(question)

    # 2. 多查询向量检索（评估时用更宽松的阈值）
    filtered_results = vector_store.query(queries, TOP_K, distance_threshold=threshold)
    total_raw = sum(len(r) for r in filtered_results)
    logger.info(f"  检索: {len(queries)} 查询, 原始命中 {total_raw} 段 (threshold={threshold})")

    # 3. RRF 融合
    merged_docs = rrf_fusion(filtered_results)
    logger.info(f"  RRF 融合后: {len(merged_docs)} 段")

    # 4. 在线重排
    contexts = []
    if merged_docs:
        try:
            rerank_results = online_rerank(queries[0], [doc.text for doc in merged_docs], top_n=10)
            top_docs = [merged_docs[r["index"]] for r in rerank_results]
            contexts = [doc.text for doc in top_docs]
            logger.info(f"  重排后: {len(contexts)} 段")
        except Exception as e:
            logger.warning(f"  重排失败，用融合结果: {e}")
            contexts = [doc.text for doc in merged_docs[:10]]

    # 5. 生成答案
    context_text = "\n\n".join(contexts) if contexts else "（未检索到相关内容）"
    prompt = (
        "你是 Mitta，一位智能 AI 助理。请严格基于以下检索到的知识库内容回答问题，"
        "不要编造知识库中没有的信息。如果知识库内容不足以回答问题，请明确告知。\n\n"
        f"【知识库内容】\n{context_text}\n\n"
        f"【问题】\n{question}"
    )
    answer = model.invoke(prompt).content
    return answer, contexts


def main():
    parser = argparse.ArgumentParser(description="RAGAS 评估")
    parser.add_argument("--limit", type=int, default=0, help="只评估前 N 条（0=全部）")
    parser.add_argument("--category", type=str, default="",
                        help="按分类过滤: basic/debug/architecture/badcase")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="向量检索距离阈值（默认0.5，越小越严格；生产环境用0.3）")
    args = parser.parse_args()

    # 1. 加载测试集
    test_data = load_test_dataset(limit=0, category=args.category)
    if not test_data:
        logger.error("测试集为空，退出")
        return

    # 2. 构建向量库连接
    cfg = load_vector_db_config()
    vector_store = create_vector_store(cfg)
    logger.info(f"向量库: {cfg.get('type')}, 集合: {cfg.get('collection')}, 阈值: {args.threshold}")

    # 3. 逐条执行 RAG，收集结果
    results = []
    for i, item in enumerate(test_data, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        category = item.get("category", "")

        logger.info(f"[{i}/{len(test_data)}] {question[:60]}...")
        start = time.time()
        try:
            answer, contexts = run_rag(vector_store, question, args.threshold)
            elapsed = time.time() - start
            results.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
                "category": category,
                "context_count": len(contexts),
                "latency_sec": round(elapsed, 2),
            })
            logger.info(f"  完成: {len(contexts)} 段上下文, {len(answer)} 字, {elapsed:.1f}s")
        except Exception as e:
            logger.error(f"  执行失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "question": question,
                "answer": f"[ERROR] {e}",
                "contexts": [],
                "ground_truth": ground_truth,
                "category": category,
                "context_count": 0,
                "latency_sec": 0,
            })

    # 4. 配置 RAGAS 评判 LLM 和 Embedding
    # 注意：必须与 init.py 中的模型配置一致（腾讯混元代理），
    # 不能用 .env 的 BASE_URL（DeepSeek 官方地址）+ HUNYUAN_API_KEY，会 401
    # 评判用 flash 模型：评分任务不需要 pro 级别推理，flash 快 3-5 倍
    ragas_llm = ChatOpenAI(
        model="deepseek-v4-flash",
        base_url="https://tokenhub.tencentmaas.com/v1",
        api_key=os.getenv("HUNYUAN_API_KEY"),
        request_timeout=120,
        max_retries=3,
    )
    ragas_embeddings = OpenAIEmbeddings(
        model="BAAI/bge-m3",
        base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        api_key=os.getenv("SILICONFLOW_API_KEY"),
    )

    # 降低 RAGAS 并发数，避免腾讯混元 API 限流/超时
    os.environ["RAGAS_MAX_CONCURRENCY"] = "2"

    # 5. 运行 RAGAS 评估（timeout=120 秒/指标，max_workers=2 降低并发）
    from ragas.run_config import RunConfig
    dataset = Dataset.from_list(results)
    logger.info("开始 RAGAS 评分（可能需要几分钟）...")

    score = evaluate(
        dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
            answer_correctness,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=True,
        run_config=RunConfig(timeout=180, max_retries=3, max_workers=2),
    )

    # 6. 输出逐条明细 CSV
    df = score.to_pandas()
    # RAGAS 返回的 df 不含输入数据中的 category 列，从 results 按行补回
    df["category"] = [r["category"] for r in results]
    df.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")
    logger.info(f"逐条报告已保存: {REPORT_CSV}")

    # 7. 汇总统计（总体 + 按分类）
    metrics_cols = ["context_precision", "context_recall", "faithfulness",
                    "answer_relevancy", "answer_correctness"]
    summary = {
        "total_questions": len(results),
        "distance_threshold": args.threshold,
        "avg_context_count": round(sum(r["context_count"] for r in results) / len(results), 2),
        "avg_latency_sec": round(sum(r["latency_sec"] for r in results) / len(results), 2),
        "overall": {},
        "by_category": {},
    }
    for col in metrics_cols:
        vals = df[col].dropna()
        summary["overall"][col] = round(float(vals.mean()), 4) if len(vals) else None

    for cat in df["category"].unique():
        cat_df = df[df["category"] == cat]
        summary["by_category"][cat] = {}
        for col in metrics_cols:
            vals = cat_df[col].dropna()
            summary["by_category"][cat][col] = round(float(vals.mean()), 4) if len(vals) else None

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"汇总报告已保存: {SUMMARY_JSON}")

    # 8. 控制台输出
    print("\n" + "=" * 60)
    print("RAGAS 评估结果")
    print("=" * 60)
    print(f"评估问题数: {summary['total_questions']}")
    print(f"距离阈值: {summary['distance_threshold']}")
    print(f"平均检索段数: {summary['avg_context_count']}")
    print(f"平均耗时: {summary['avg_latency_sec']}s")
    print("-" * 60)
    for metric, val in summary["overall"].items():
        print(f"  {metric:25s}: {val}")
    print("-" * 60)
    for cat, metrics in summary["by_category"].items():
        print(f"  [{cat}]")
        for metric, val in metrics.items():
            print(f"    {metric:23s}: {val}")
    print("=" * 60)


if __name__ == "__main__":
    main()
