from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from rag_core import (
    EmbeddingRetriever,
    HybridRetriever,
    Retriever,
    TfidfRetriever,
    load_markdown_chunks,
)


ROOT = Path(__file__).parent
KNOWLEDGE_DIR = ROOT / "knowledge_base"
DEFAULT_EMBEDDING_MODEL = ROOT / "models" / "bge-small-zh-v1.5"


@dataclass(frozen=True)
class EvalCase:
    question: str
    expected_sources: set[str]


@dataclass(frozen=True)
class EvalRow:
    method: str
    question: str
    expected_sources: set[str]
    retrieved_sources: list[str]
    hits: set[str]
    precision: float
    recall: float


EVAL_CASES = [
    EvalCase(
        question="风很大但发电不多，可能是什么原因？",
        expected_sources={
            "03_限功率.md",
            "04_偏航误差.md",
            "05_桨距角.md",
            "10_SCADA异常与数据清洗论文.md",
        },
    ),
    EvalCase(
        question="SHAP 值高说明什么？",
        expected_sources={
            "08_SHAP解释.md",
            "11_SHAP可解释性论文.md",
        },
    ),
    EvalCase(
        question="为什么不能只用 embedding 检索？",
        expected_sources={
            "12_RAG检索策略与混合检索.md",
        },
    ),
    EvalCase(
        question="SCADA 异常为什么会影响预测？",
        expected_sources={
            "06_SCADA异常.md",
            "10_SCADA异常与数据清洗论文.md",
        },
    ),
    EvalCase(
        question="超短期风电功率预测误差为什么会突然变大？",
        expected_sources={
            "02_风速波动.md",
            "07_预测误差.md",
            "09_风电功率预测论文综述.md",
        },
    ),
    EvalCase(
        question="限功率会怎样影响模型预测？",
        expected_sources={
            "03_限功率.md",
            "06_SCADA异常.md",
            "10_SCADA异常与数据清洗论文.md",
        },
    ),
    EvalCase(
        question="桨距角异常为什么会影响功率输出？",
        expected_sources={
            "05_桨距角.md",
            "10_SCADA异常与数据清洗论文.md",
        },
    ),
    EvalCase(
        question="短期风电功率预测有哪些常用方法？",
        expected_sources={
            "09_风电功率预测论文综述.md",
        },
    ),
]


EVAL_CASES_2 = EVAL_CASES + [
    EvalCase(
        question="为什么 RMSE 很高但 MAE 看起来还可以？",
        expected_sources={
            "07_预测误差.md",
            "02_风速波动.md",
            "06_SCADA异常.md",
        },
    ),
    EvalCase(
        question="MAPE 在低功率或停机样本里为什么不稳定？",
        expected_sources={
            "07_预测误差.md",
            "06_SCADA异常.md",
        },
    ),
    EvalCase(
        question="RRF 在混合检索里是怎么融合排序的？",
        expected_sources={
            "12_RAG检索策略与混合检索.md",
        },
    ),
    EvalCase(
        question="同一个风速为什么会对应多个不同功率？",
        expected_sources={
            "01_风速功率关系.md",
            "03_限功率.md",
            "04_偏航误差.md",
            "05_桨距角.md",
            "06_SCADA异常.md",
        },
    ),
    EvalCase(
        question="偏航误差应该怎么判断，为什么会让模型高估？",
        expected_sources={
            "04_偏航误差.md",
            "01_风速功率关系.md",
            "06_SCADA异常.md",
        },
    ),
    EvalCase(
        question="风向快速变化后预测误差变大，可能和什么有关？",
        expected_sources={
            "02_风速波动.md",
            "04_偏航误差.md",
            "07_预测误差.md",
        },
    ),
    EvalCase(
        question="桨距角 SHAP 贡献高，工程上应该怎么解释？",
        expected_sources={
            "05_桨距角.md",
            "08_SHAP解释.md",
            "11_SHAP可解释性论文.md",
        },
    ),
    EvalCase(
        question="为什么 RAG 能减少大模型胡编，检索结果差会有什么影响？",
        expected_sources={
            "12_RAG检索策略与混合检索.md",
        },
    ),
    EvalCase(
        question="预测曲线趋势差不多但整体提前或滞后，应该先检查什么？",
        expected_sources={
            "06_SCADA异常.md",
            "02_风速波动.md",
            "07_预测误差.md",
        },
    ),
    EvalCase(
        question="为什么训练前要清洗风机 SCADA 功率曲线异常点？",
        expected_sources={
            "10_SCADA异常与数据清洗论文.md",
            "06_SCADA异常.md",
            "01_风速功率关系.md",
        },
    ),
    EvalCase(
        question="有风但功率接近 0 的样本会怎样影响模型？",
        expected_sources={
            "10_SCADA异常与数据清洗论文.md",
            "03_限功率.md",
            "06_SCADA异常.md",
        },
    ),
    EvalCase(
        question="高风速下功率形成固定平台，怎么区分额定平台和限功率？",
        expected_sources={
            "03_限功率.md",
            "01_风速功率关系.md",
            "05_桨距角.md",
        },
    ),
    EvalCase(
        question="为什么模型在风速突增时容易低估峰值？",
        expected_sources={
            "02_风速波动.md",
            "07_预测误差.md",
            "09_风电功率预测论文综述.md",
        },
    ),
    EvalCase(
        question="短期和超短期风电功率预测有什么区别？",
        expected_sources={
            "09_风电功率预测论文综述.md",
            "02_风速波动.md",
        },
    ),
    EvalCase(
        question="物理模型、统计模型和深度学习模型各适合什么预测场景？",
        expected_sources={
            "09_风电功率预测论文综述.md",
        },
    ),
    EvalCase(
        question="点预测和区间预测有什么区别，为什么风电预测要表达不确定性？",
        expected_sources={
            "09_风电功率预测论文综述.md",
            "07_预测误差.md",
        },
    ),
    EvalCase(
        question="历史功率 SHAP 贡献高但工况突然变化时，为什么可能预测滞后？",
        expected_sources={
            "08_SHAP解释.md",
            "11_SHAP可解释性论文.md",
            "02_风速波动.md",
        },
    ),
    EvalCase(
        question="为什么不能把 SHAP 特征重要性直接当成真实原因？",
        expected_sources={
            "08_SHAP解释.md",
            "11_SHAP可解释性论文.md",
            "06_SCADA异常.md",
        },
    ),
    EvalCase(
        question="TF-IDF 和 embedding 检索各有什么优势，为什么要混合？",
        expected_sources={
            "12_RAG检索策略与混合检索.md",
        },
    ),
    EvalCase(
        question="如果某个传感器长时间不变，为什么会误导预测和解释？",
        expected_sources={
            "06_SCADA异常.md",
            "10_SCADA异常与数据清洗论文.md",
            "08_SHAP解释.md",
        },
    ),
    EvalCase(
        question="功率曲线下方出现密集散点，通常可能是什么工况？",
        expected_sources={
            "01_风速功率关系.md",
            "03_限功率.md",
            "04_偏航误差.md",
            "05_桨距角.md",
            "10_SCADA异常与数据清洗论文.md",
        },
    ),
    EvalCase(
        question="检索结果混入无关文档时，会怎样影响 RAG 回答质量？",
        expected_sources={
            "12_RAG检索策略与混合检索.md",
        },
    ),
]


def main() -> None:
    args = parse_args()
    chunks = load_markdown_chunks(args.knowledge_dir)
    retrievers = build_retrievers(chunks, args.embedding_model)

    for eval_name, eval_cases in [
        ("Eval set 1", EVAL_CASES),
        ("Eval set 2", EVAL_CASES_2),
    ]:
        rows = evaluate_cases(retrievers, eval_cases, top_k=args.top_k)
        print_summary(rows, top_k=args.top_k, title=eval_name)
        if args.details:
            print_details(rows, top_k=args.top_k, title=eval_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TF-IDF, embedding, and hybrid retrieval on curated RAG questions."
    )
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--knowledge-dir",
        type=Path,
        default=KNOWLEDGE_DIR,
        help="Knowledge base directory.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=str(DEFAULT_EMBEDDING_MODEL),
        help="Sentence-transformers model name or local model path.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print per-question retrieved sources.",
    )
    return parser.parse_args()


def build_retrievers(chunks, embedding_model: str) -> dict[str, Retriever]:
    return {
        "TF-IDF": TfidfRetriever(chunks),
        "Embedding": EmbeddingRetriever(chunks, model_name=embedding_model),
        "Hybrid": HybridRetriever(chunks, embedding_model_name=embedding_model),
    }


def evaluate_cases(
    retrievers: dict[str, Retriever],
    eval_cases: list[EvalCase],
    top_k: int,
) -> list[EvalRow]:
    return [
        evaluate_case(method, retriever, case, top_k=top_k)
        for method, retriever in retrievers.items()
        for case in eval_cases
    ]


def evaluate_case(
    method: str,
    retriever: Retriever,
    case: EvalCase,
    top_k: int,
) -> EvalRow:
    results = retriever.search(case.question, top_k=top_k)
    retrieved_sources = dedupe_preserve_order([result.chunk.source for result in results])
    hits = set(retrieved_sources) & case.expected_sources
    precision = len(hits) / top_k if top_k else 0.0
    recall = len(hits) / len(case.expected_sources) if case.expected_sources else 0.0
    return EvalRow(
        method=method,
        question=case.question,
        expected_sources=case.expected_sources,
        retrieved_sources=retrieved_sources,
        hits=hits,
        precision=precision,
        recall=recall,
    )


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def print_summary(rows: list[EvalRow], top_k: int, title: str) -> None:
    print(f"\n{title}: file-level Precision@{top_k} / Recall@{top_k}")
    print("-" * 72)
    print(f"{'Method':<12} {'Cases':>5} {'Precision':>10} {'Recall':>10} {'Hit cases':>10}")
    print("-" * 72)

    methods = list(dict.fromkeys(row.method for row in rows))
    for method in methods:
        method_rows = [row for row in rows if row.method == method]
        avg_precision = sum(row.precision for row in method_rows) / len(method_rows)
        avg_recall = sum(row.recall for row in method_rows) / len(method_rows)
        hit_cases = sum(1 for row in method_rows if row.hits)
        print(
            f"{method:<12} {len(method_rows):>5} "
            f"{avg_precision:>10.3f} {avg_recall:>10.3f} {hit_cases:>10}"
        )


def print_details(rows: list[EvalRow], top_k: int, title: str) -> None:
    print(f"\n{title}: per-question details, top_k={top_k}")
    print("=" * 72)
    for row in rows:
        print(f"\n[{row.method}] {row.question}")
        print(f"Expected: {', '.join(sorted(row.expected_sources))}")
        print(f"Retrieved: {', '.join(row.retrieved_sources) or '(none)'}")
        print(f"Hits: {', '.join(sorted(row.hits)) or '(none)'}")
        print(f"Precision: {row.precision:.3f}  Recall: {row.recall:.3f}")


if __name__ == "__main__":
    main()
