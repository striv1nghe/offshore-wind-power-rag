from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag_core import SearchResult, TfidfRetriever
from wind_data import (
    dataset_summary,
    feature_error_correlations,
    high_error_rows,
    list_datasets,
    load_merged_dataset,
    make_error_summary,
)


@dataclass(frozen=True)
class AgentToolTrace:
    name: str
    purpose: str
    output: str


@dataclass(frozen=True)
class AgentRun:
    dataset: str
    tool_traces: list[AgentToolTrace]
    search_results: list[SearchResult]
    final_prompt: str


def run_wind_analysis_agent(
    *,
    question: str,
    selected_dataset: str | None,
    retriever: TfidfRetriever,
    data_dir: str | Path = "data",
) -> AgentRun:
    datasets = list_datasets(data_dir)
    dataset = choose_dataset(question, selected_dataset, datasets)
    df = load_merged_dataset(dataset, data_dir)

    traces = [
        AgentToolTrace(
            name="choose_dataset",
            purpose="确定本次分析的数据集",
            output=f"选择数据集：{dataset}",
        ),
        AgentToolTrace(
            name="get_dataset_summary",
            purpose="读取数据集规模、时间范围和整体误差指标",
            output=format_dataset_summary(dataset, data_dir),
        ),
        AgentToolTrace(
            name="get_error_summary",
            purpose="生成误差摘要和高误差样本",
            output=make_error_summary(dataset, df),
        ),
        AgentToolTrace(
            name="get_feature_correlations",
            purpose="分析输入变量与绝对误差的 Spearman 相关性",
            output=format_top_correlations(df),
        ),
    ]

    search_query = build_agent_search_query(question, traces)
    search_results = retriever.search(search_query, top_k=5)
    traces.append(
        AgentToolTrace(
            name="search_knowledge_base",
            purpose="检索风电预测解释知识库",
            output=format_search_results(search_results),
        )
    )

    return AgentRun(
        dataset=dataset,
        tool_traces=traces,
        search_results=search_results,
        final_prompt=build_agent_final_prompt(question, dataset, traces, search_results),
    )


def choose_dataset(
    question: str,
    selected_dataset: str | None,
    datasets: list[str],
) -> str:
    if selected_dataset and selected_dataset != "自动识别":
        return selected_dataset

    normalized_question = question.lower()
    for dataset in datasets:
        if dataset.lower() in normalized_question:
            return dataset

    if not datasets:
        raise ValueError("没有找到可用数据集。")
    return datasets[0]


def format_dataset_summary(dataset: str, data_dir: str | Path) -> str:
    summary = dataset_summary(data_dir)
    row = summary[summary["dataset"] == dataset]
    if row.empty:
        return f"没有找到 {dataset} 的概览信息。"

    item = row.iloc[0]
    return "\n".join(
        [
            f"数据集：{item['dataset']}",
            f"数据总量：{int(item['data_rows'])}",
            f"预测样本数：{int(item['prediction_rows'])}",
            f"时间范围：{item['start_time']} 至 {item['end_time']}",
            f"RMSE：{item['RMSE_KW']:.3f} kW",
            f"MAE：{item['MAE_KW']:.3f} kW",
            f"R2：{item['R2']:.6f}",
            f"MAPE：{item['MAPE']:.6f}",
        ]
    )


def format_top_correlations(df, count: int = 8) -> str:
    corr = feature_error_correlations(df).head(count)
    if corr.empty:
        return "没有可用的数值特征相关性结果。"

    lines = ["与绝对误差 Spearman 相关性较高的变量："]
    for _, row in corr.iterrows():
        lines.append(
            f"- {row['feature']}："
            f"abs_error={row['spearman_abs_error']:.3f}，"
            f"signed_error={row['spearman_error']:.3f}"
        )
    return "\n".join(lines)


def format_search_results(results: list[SearchResult]) -> str:
    if not results:
        return "没有召回知识库片段。"

    lines = []
    for result in results:
        lines.append(
            f"- {result.chunk.source} / {result.chunk.title} "
            f"(score={result.score:.3f})"
        )
    return "\n".join(lines)


def build_agent_search_query(question: str, traces: list[AgentToolTrace]) -> str:
    evidence = "\n".join(trace.output for trace in traces)
    return (
        question
        + "\n"
        + evidence
        + "\nRMSE MAE MAPE 预测误差 风速波动 偏航误差 桨距角 SCADA异常 SHAP 特征解释"
    )


def build_agent_final_prompt(
    question: str,
    dataset: str,
    traces: list[AgentToolTrace],
    results: list[SearchResult],
) -> str:
    tool_evidence = "\n\n".join(
        f"工具：{trace.name}\n目的：{trace.purpose}\n输出：\n{trace.output}"
        for trace in traces
    )
    knowledge_context = "\n\n".join(
        f"来源：{result.chunk.source} / {result.chunk.title}\n{result.chunk.text}"
        for result in results
    )
    return f"""你是一个海上风电功率预测解释 Agent。
你已经调用了一组数据分析工具和知识库检索工具。请只根据工具输出和知识库片段回答。
如果证据不足，请明确说明不足，不要编造数据。

用户问题：
{question}

当前数据集：
{dataset}

工具调用证据：
{tool_evidence}

知识库片段：
{knowledge_context}

请用中文输出结构化分析报告，格式如下：
1. 分析目标
2. Agent 调用了哪些工具
3. 关键数据现象
4. 可能原因
5. 知识库依据
6. 建议下一步验证"""
