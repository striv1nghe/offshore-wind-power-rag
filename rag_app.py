from __future__ import annotations

import os
from pathlib import Path


import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from rag_core import (
    EmbeddingRetriever,
    HybridRetriever,
    Retriever,
    TfidfRetriever,
    answer_ollama,
    answer_openai_compatible,
    append_reference_footer,
    build_data_explanation_prompt,
    build_direct_prompt,
    build_prompt,
    load_markdown_chunks,
)
from wind_data import (
    ABS_ERROR_COL,
    ERROR_COL,
    PRED_COL,
    TIME_COL,
    TRUE_COL,
    dataset_summary,
    downsample_frame,
    feature_error_correlations,
    high_error_rows,
    input_feature_columns,
    list_datasets,
    load_loss,
    load_merged_dataset,
    make_error_summary,
    suggested_error_questions,
)


ROOT = Path(__file__).parent
KNOWLEDGE_DIR = ROOT / "knowledge_base"
DATA_DIR = ROOT / "data"
HYBRID_RETRIEVAL = "Hybrid 混合检索"
TFIDF_RETRIEVAL = "TF-IDF 字符检索"
EMBEDDING_RETRIEVAL = "Embedding 语义检索"
DEFAULT_EMBEDDING_MODEL = str(ROOT / "models" / "bge-small-zh-v1.5")

DISPLAY_COLUMN_NAMES = {
    TRUE_COL: "真实值_KW",
    PRED_COL: "预测值_KW",
    ERROR_COL: "误差_KW",
    ABS_ERROR_COL: "绝对误差_KW",
}
TIME_DISPLAY_FORMAT = "%Y-%m-%d %H:%M:%S"


@st.cache_resource
def get_retriever(method: str, embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> Retriever:
    chunks = load_markdown_chunks(KNOWLEDGE_DIR)
    if method == HYBRID_RETRIEVAL:
        return HybridRetriever(chunks, embedding_model_name=embedding_model)
    if method == EMBEDDING_RETRIEVAL:
        return EmbeddingRetriever(chunks, model_name=embedding_model)
    return TfidfRetriever(chunks)


@st.cache_data
def get_dataset_summary():
    return dataset_summary(DATA_DIR)


@st.cache_data
def get_dataset_names() -> list[str]:
    return list_datasets(DATA_DIR)


@st.cache_data
def get_merged_dataset(dataset: str):
    return load_merged_dataset(dataset, DATA_DIR)


@st.cache_data
def get_loss(dataset: str):
    return load_loss(dataset, DATA_DIR)


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="TRAG 风电 RAG", page_icon="TRAG", layout="wide")
    init_history()

    st.title("RAG 风电功率预测")
    st.caption("可以用知识库 RAG 回答，也可以绕过知识库直接问大模型。")

    page = st.sidebar.radio(
        "页面",
        ["知识库问答", "数据概览", "数据分析与解释"],
    )

    provider_options = ["请选择", "OpenAI / 兼容 API", "DeepSeek", "Ollama 本地模型"]
    provider = st.sidebar.selectbox(
        "选择大模型类型",
        provider_options,
        index=provider_options.index(infer_default_provider()),
        help="会根据 .env 中已填写的配置自动选择，也可以在这里手动切换。",
    )

    retrieval_config = collect_retrieval_config(page)

    st.sidebar.divider()
    model_config = collect_model_config(provider) if provider != "请选择" else {}

    if page == "知识库问答":
        render_rag_page(provider, model_config, retrieval_config)
    elif page == "数据概览":
        render_data_overview_page()
    elif page == "数据分析与解释":
        render_data_analysis_page(provider, model_config, retrieval_config)


def init_history() -> None:
    st.session_state.setdefault("rag_history", [])
    st.session_state.setdefault("direct_history", [])


def render_rag_page(
    provider: str,
    model_config: dict[str, str],
    retrieval_config: dict[str, str],
) -> None:
    st.subheader("知识库问答")
    answer_mode = st.radio(
        "问答模式",
        ["知识库 RAG", "直接问模型"],
        horizontal=True,
    )

    if answer_mode == "直接问模型":
        render_direct_page(provider, model_config)
        return

    st.caption("先检索本地风电知识库，再让大模型基于召回片段回答。")

    question = st.text_area(
        "输入一个风电预测问题",
        value="为什么风速波动时，风电功率预测误差会变大？",
        height=90,
        key="rag_question",
    )
    top_k = st.slider("召回片段数量", min_value=1, max_value=8, value=4)
    retrieve_only = st.checkbox("只测试检索，不调用大模型")

    if st.button("检索并生成回答", type="primary"):
        if not question.strip():
            st.warning("请先输入问题。")
            return
        if provider == "请选择" and not retrieve_only:
            st.warning("请先在左侧选择要接入的大模型类型，或勾选只测试检索。")
            return

        retriever = load_selected_retriever(retrieval_config)
        if retriever is None:
            return

        with st.spinner("正在检索知识库..."):
            results = retriever.search(question, top_k=top_k)
        if not results:
            st.warning("知识库里暂时没有召回到相关片段。")
            return

        prompt = build_prompt(question, results)
        show_results(results)

        if retrieve_only:
            st.subheader("已生成 Prompt")
            st.code(prompt, language="markdown")
            st.session_state.rag_history.insert(
                0,
                {
                    "question": question,
                    "answer": "只测试检索，未调用大模型。",
                    "results": results,
                },
            )
        else:
            with st.spinner("正在调用大模型，请稍等..."):
                answer = generate_answer(provider, model_config, prompt)
            if is_model_answer(answer):
                answer = append_reference_footer(answer, results)
                st.session_state.rag_history.insert(
                    0,
                    {"question": question, "answer": answer, "results": results},
                )
            st.subheader("回答")
            st.markdown(answer)

    render_rag_history()


def render_direct_page(provider: str, model_config: dict[str, str]) -> None:
    st.subheader("直接问模型")
    st.caption("这个页面不检索知识库，问题会直接发送给当前选择的大模型。")

    question = st.text_area(
        "输入你想直接问大模型的问题",
        value="请用通俗语言解释一下 RMSE 是什么。",
        height=120,
        key="direct_question",
    )

    if st.button("直接生成回答", type="primary"):
        if not question.strip():
            st.warning("请先输入问题。")
            return
        if provider == "请选择":
            st.warning("请先在左侧选择要接入的大模型类型。")
            return

        prompt = build_direct_prompt(question)
        with st.spinner("正在调用大模型，请稍等..."):
            answer = generate_answer(provider, model_config, prompt)
        if is_model_answer(answer):
            st.session_state.direct_history.insert(
                0,
                {"question": question, "answer": answer},
            )
        st.subheader("回答")
        st.markdown(answer)

    render_direct_history()


def render_data_overview_page() -> None:
    st.subheader("数据概览")
    st.caption("直接读取 data/ 下的 CSV，当前还没有导入 SQL 数据库。")

    summary = get_dataset_summary()
    if summary.empty:
        st.warning("没有在 data/ 目录下找到可用数据。")
        return

    metric_cols = st.columns(5)
    metric_cols[0].metric("数据集数量", len(summary))
    metric_cols[1].metric("数据总量", f"{int(summary['data_rows'].sum()):,}")
    metric_cols[2].metric("预测样本总数", f"{int(summary['prediction_rows'].sum()):,}")
    metric_cols[3].metric("平均 RMSE", f"{summary['RMSE_KW'].mean():.2f} kW")
    metric_cols[4].metric("平均 MAE", f"{summary['MAE_KW'].mean():.2f} kW")

    display = summary.copy()
    display["start_time"] = display["start_time"].dt.strftime("%Y-%m-%d %H:%M")
    display["end_time"] = display["end_time"].dt.strftime("%Y-%m-%d %H:%M")
    display = display[
        [
            "dataset",
            "data_rows",
            "prediction_rows",
            "feature_rows",
            "input_columns",
            "start_time",
            "end_time",
            "RMSE_KW",
            "MAE_KW",
            "R2",
            "MAPE",
        ]
    ]
    display = display.rename(
        columns={
            "dataset": "数据集",
            "data_rows": "数据总量",
            "prediction_rows": "预测样本数",
            "feature_rows": "特征数据行数",
            "input_columns": "输入变量数",
            "start_time": "开始时间",
            "end_time": "结束时间",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("数据随时间变化")
    dataset = st.selectbox("选择数据集", get_dataset_names(), key="overview_dataset")
    df = get_merged_dataset(dataset)
    numeric_columns = [
        TRUE_COL,
        PRED_COL,
        ERROR_COL,
        ABS_ERROR_COL,
        *input_feature_columns(df),
    ]
    value_col = st.selectbox(
        "选择要查看的数据列",
        numeric_columns,
        format_func=display_column_name,
        key="overview_value_col",
    )
    chart_df = downsample_frame(df[[TIME_COL, value_col]], max_rows=2000)
    chart = (
        alt.Chart(chart_df)
        .mark_line()
        .encode(
            x=time_x(),
            y=alt.Y(f"{value_col}:Q", title=display_column_name(value_col)),
            tooltip=[
                time_tooltip(),
                alt.Tooltip(f"{value_col}:Q", title=display_column_name(value_col), format=".3f"),
            ],
        )
        .interactive()
        .properties(height=360)
    )
    st.altair_chart(chart, use_container_width=True)


def render_data_analysis_page(
    provider: str,
    model_config: dict[str, str],
    retrieval_config: dict[str, str],
) -> None:
    st.subheader("数据分析与解释")
    dataset = select_dataset()
    if not dataset:
        return

    df = get_merged_dataset(dataset)
    st.caption(f"当前数据集：{dataset}，共 {len(df):,} 条预测样本。")
    tab_prediction, tab_feature, tab_explain = st.tabs(
        ["预测可视化", "特征分析", "误差解释"]
    )
    with tab_prediction:
        render_prediction_visual(dataset, df)
    with tab_feature:
        render_feature_analysis(dataset, df)
    with tab_explain:
        render_error_explanation(provider, model_config, retrieval_config, dataset, df)


def render_prediction_visual(dataset: str, df) -> None:
    st.subheader("预测可视化")
    st.caption(f"当前数据集：{dataset}，共 {len(df):,} 条预测样本。")

    reset_token_key = f"visual_reset_token_{dataset}"
    st.session_state.setdefault(reset_token_key, 0)
    reset_token = st.session_state[reset_token_key]

    selected_time = render_high_error_selector(dataset, df, reset_token)
    left, _ = st.columns([1, 3])
    if left.button(
        "返回全局视图",
        key=f"clear_focus_{dataset}",
        help="清除高误差时间点聚焦，恢复完整时间范围。",
    ):
        st.session_state.pop(f"selected_error_time_{dataset}", None)
        st.session_state[reset_token_key] += 1
        st.rerun()
    left.caption("恢复鼠标缩放前的初始图表状态。")

    input_col, help_col = st.columns([1, 3])
    focus_hours = input_col.number_input(
        "高误差点窗口（小时）",
        min_value=1,
        max_value=48,
        value=2,
        step=1,
        help="点选误差最大时间点后，图表会显示该时间点前后各多少小时。",
    )
    help_col.caption("只在点选高误差时间点后生效；返回全局视图会恢复完整时间范围。")
    chart_source = filter_time_window(df, selected_time, hours=focus_hours)
    if selected_time is not None:
        st.info(f"已聚焦到 {selected_time} 前后 {focus_hours} 小时。")

    chart_df = downsample_frame(
        chart_source[[TIME_COL, TRUE_COL, PRED_COL, ERROR_COL, ABS_ERROR_COL]]
    )
    chart_df = chart_df.copy()
    chart_df["_reset_token"] = reset_token
    power_long = chart_df[[TIME_COL, TRUE_COL, PRED_COL]].melt(
        id_vars=TIME_COL,
        value_vars=[TRUE_COL, PRED_COL],
        var_name="类型",
        value_name="功率",
    )
    power_long["类型"] = power_long["类型"].map(display_column_name)
    st.markdown("**真实功率、预测功率与误差联动图**")
    zoom = alt.selection_interval(bind="scales", encodings=["x"])
    power_chart = (
        alt.Chart(power_long)
        .mark_line()
        .encode(
            x=time_x(),
            y=alt.Y("功率:Q", title="功率 kW"),
            color=alt.Color(
                "类型:N",
                title="曲线",
                scale=alt.Scale(
                    domain=["真实值_KW", "预测值_KW"],
                    range=["#9ecae1", "#08519c"],
                ),
            ),
            tooltip=[time_tooltip(), "类型", alt.Tooltip("功率:Q", format=".2f")],
        )
        .properties(name=f"power_chart_{dataset}_{reset_token}")
        .properties(height=360, title="真实功率与预测功率")
    )
    if selected_time is not None:
        power_chart = power_chart + selected_time_rule(selected_time)

    error_chart = (
        alt.Chart(chart_df)
        .mark_line(color="#d62728")
        .encode(
            x=time_x(),
            y=alt.Y(f"{ERROR_COL}:Q", title="预测误差 kW"),
            tooltip=[
                time_tooltip(),
                alt.Tooltip(f"{ERROR_COL}:Q", format=".2f"),
                alt.Tooltip(f"{ABS_ERROR_COL}:Q", format=".2f"),
            ],
        )
        .properties(name=f"error_chart_{dataset}_{reset_token}")
        .properties(height=260, title="预测误差曲线")
    )
    if selected_time is not None:
        error_chart = error_chart + selected_time_rule(selected_time)

    linked_chart = (
        alt.vconcat(power_chart, error_chart)
        .add_params(zoom)
        .resolve_scale(x="shared")
    )
    st.altair_chart(
        linked_chart,
        use_container_width=True,
        key=f"linked_power_error_chart_{dataset}_{reset_token}",
    )
    st.divider()

    st.markdown("**绝对误差分布**")
    hist = (
        alt.Chart(df[[ABS_ERROR_COL]])
        .mark_bar()
        .encode(
            x=alt.X(f"{ABS_ERROR_COL}:Q", bin=alt.Bin(maxbins=40), title="绝对误差 kW"),
            y=alt.Y("count():Q", title="样本数"),
            tooltip=[alt.Tooltip("count():Q", title="样本数")],
        )
        .properties(height=300)
    )
    st.altair_chart(
        hist,
        use_container_width=True,
        key=f"abs_error_hist_{dataset}_{reset_token}",
    )
    st.divider()

    st.markdown("**训练与验证 Loss**")
    loss = get_loss(dataset)
    loss_long = loss.melt(
        id_vars="epoch",
        value_vars=["train_loss", "val_loss"],
        var_name="类型",
        value_name="loss",
    )
    loss_chart = (
        alt.Chart(loss_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("epoch:Q", title="训练轮次"),
            y=alt.Y("loss:Q", title="Loss"),
            color=alt.Color("类型:N", title="曲线"),
            tooltip=["epoch", "类型", alt.Tooltip("loss:Q", format=".6f")],
        )
        .properties(height=300)
    )
    st.altair_chart(
        loss_chart,
        use_container_width=True,
        key=f"loss_chart_{dataset}_{reset_token}",
    )


def render_high_error_selector(dataset: str, df, reset_token: int) -> pd.Timestamp | None:
    st.subheader("误差最大的时间点")
    st.caption("可以点表格行，下面曲线会用竖线标出该高误差时间点。")

    table = high_error_rows(df, count=10).copy()
    display_table = table.rename(columns=DISPLAY_COLUMN_NAMES)
    display_table[TIME_COL] = display_table[TIME_COL].dt.strftime(TIME_DISPLAY_FORMAT)
    event = st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
        height=180,
        selection_mode="single-row",
        on_select="rerun",
        key=f"high_error_table_{dataset}_{reset_token}",
    )

    selected_time = selected_time_from_event(event, table)
    state_key = f"selected_error_time_{dataset}"
    if selected_time is not None:
        st.session_state[state_key] = selected_time
        return selected_time
    return st.session_state.get(state_key)


def selected_time_from_event(event, table) -> pd.Timestamp | None:
    try:
        selected_rows = event.selection.rows
    except AttributeError:
        return None

    if not selected_rows:
        return None
    row_index = selected_rows[0]
    if row_index >= len(table):
        return None
    return pd.to_datetime(table.iloc[row_index][TIME_COL])


def filter_time_window(df, selected_time: pd.Timestamp | None, hours: int | float):
    if selected_time is None:
        return df
    start = selected_time - pd.Timedelta(hours=hours)
    end = selected_time + pd.Timedelta(hours=hours)
    window = df[(df[TIME_COL] >= start) & (df[TIME_COL] <= end)]
    return window if not window.empty else df


def selected_time_rule(selected_time: pd.Timestamp):
    return (
        alt.Chart(pd.DataFrame({TIME_COL: [selected_time]}))
        .mark_rule(color="#111111", strokeDash=[6, 4])
        .encode(x=time_x())
    )


def display_column_name(column: str) -> str:
    return DISPLAY_COLUMN_NAMES.get(column, column.replace("_Step1", ""))


def time_x():
    return alt.X(
        f"{TIME_COL}:T",
        title="时间",
        axis=alt.Axis(format=TIME_DISPLAY_FORMAT),
    )


def time_tooltip():
    return alt.Tooltip(
        f"{TIME_COL}:T",
        title="时间",
        format=TIME_DISPLAY_FORMAT,
    )


def build_suggested_questions(dataset: str, df) -> list[str]:
    questions = suggested_error_questions(dataset, df, count=5)
    selected_time = st.session_state.get(f"selected_error_time_{dataset}")
    if selected_time is None:
        return questions

    matched = df[df[TIME_COL] == selected_time]
    if matched.empty:
        return questions

    row = matched.iloc[0]
    selected_question = (
        f"请重点解释我选中的高误差时间点 {row[TIME_COL]}："
        f"真实值 {row[TRUE_COL]:.3f} kW，预测值 {row[PRED_COL]:.3f} kW，"
        f"误差 {row[ERROR_COL]:.3f} kW，绝对误差 {row[ABS_ERROR_COL]:.3f} kW。"
    )
    return [selected_question] + [
        question for question in questions if question != selected_question
    ]


def render_feature_analysis(dataset: str, df) -> None:
    st.subheader("特征分析")
    corr = feature_error_correlations(df)
    if corr.empty:
        st.warning("没有找到可用于相关性分析的数值特征列。")
        return

    st.caption("这里展示的是输入变量与绝对误差的 Spearman 相关性，用来辅助判断变量的单调变化是否和误差变化有关。")
    st.dataframe(corr, use_container_width=True, hide_index=True)

    bar = (
        alt.Chart(corr)
        .mark_bar()
        .encode(
            x=alt.X("spearman_abs_error:Q", title="与绝对误差的 Spearman 相关系数"),
            y=alt.Y("feature:N", sort="-x", title="输入变量"),
            tooltip=[
                "feature",
                alt.Tooltip("spearman_abs_error:Q", format=".3f"),
                alt.Tooltip("spearman_error:Q", format=".3f"),
            ],
        )
        .properties(height=360)
    )
    st.altair_chart(bar, use_container_width=True)

    feature = st.selectbox("选择一个变量查看它和绝对误差的关系", corr["feature"].tolist())
    scatter_df = downsample_frame(df[[TIME_COL, feature, ABS_ERROR_COL]], max_rows=2000)
    scatter = (
        alt.Chart(scatter_df)
        .mark_circle(size=22, opacity=0.45)
        .encode(
            x=alt.X(f"{feature}:Q", title=feature),
            y=alt.Y(f"{ABS_ERROR_COL}:Q", title="绝对误差 kW"),
            tooltip=[
                time_tooltip(),
                alt.Tooltip(f"{feature}:Q", format=".3f"),
                alt.Tooltip(f"{ABS_ERROR_COL}:Q", format=".3f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(scatter, use_container_width=True)


def render_error_explanation(
    provider: str,
    model_config: dict[str, str],
    retrieval_config: dict[str, str],
    dataset: str,
    df,
) -> None:
    st.subheader("误差解释")
    st.caption("把数据摘要和知识库片段一起发给大模型，不会发送完整 CSV。")
    data_summary = make_error_summary(dataset, df)
    st.text_area("将发送给模型的数值摘要", value=data_summary, height=260)

    questions = build_suggested_questions(dataset, df)
    selected_question = st.selectbox("建议问答库", questions)
    question = st.text_area(
        "你想让模型解释什么？可以直接使用建议问题，也可以在这里修改。",
        value=selected_question,
        height=110,
        key=f"error_question_{dataset}_{questions.index(selected_question)}",
    )

    if st.button("生成误差解释", type="primary"):
        if provider == "请选择":
            st.warning("请先在左侧选择大模型类型，再生成误差解释。")
            return
        retriever = load_selected_retriever(retrieval_config)
        if retriever is None:
            return
        search_query = (
            question
            + "\nRMSE MAE 预测误差 风速波动 桨距角 偏航 变桨 电网有功功率"
        )
        with st.spinner("正在检索知识库..."):
            results = retriever.search(search_query, top_k=4)
        if not results:
            st.warning("知识库里暂时没有召回到相关片段。")
            return

        prompt = build_data_explanation_prompt(
            question=question,
            data_summary=data_summary,
            results=results,
        )
        show_results(results)

        with st.spinner("正在调用大模型，请稍等..."):
            answer = generate_answer(provider, model_config, prompt)
        if is_model_answer(answer):
            answer = append_reference_footer(answer, results)
        st.subheader("解释结果")
        st.markdown(answer)


def select_dataset() -> str | None:
    datasets = get_dataset_names()
    if not datasets:
        st.warning("没有在 data/INPUT 下找到特征数据。")
        return None
    return st.selectbox("选择数据集", datasets)


def infer_default_provider() -> str:
    if os.getenv("DEEPSEEK_API_KEY") and os.getenv("DEEPSEEK_MODEL"):
        return "DeepSeek"
    if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MODEL"):
        return "OpenAI / 兼容 API"
    if os.getenv("OLLAMA_MODEL"):
        return "Ollama 本地模型"
    return "请选择"


def collect_model_config(provider: str) -> dict[str, str]:
    if provider == "DeepSeek":
        return {
            "api_key": st.sidebar.text_input(
                "DeepSeek API Key",
                value=os.getenv("DEEPSEEK_API_KEY", ""),
                type="password",
            ),
            "base_url": st.sidebar.text_input(
                "Base URL",
                value=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            ),
            "model": st.sidebar.text_input(
                "模型名",
                value=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            ),
        }

    if provider == "Ollama 本地模型":
        return {
            "base_url": st.sidebar.text_input(
                "Ollama 地址",
                value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ),
            "model": st.sidebar.text_input(
                "模型名",
                value=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            ),
        }

    if provider == "OpenAI / 兼容 API":
        return {
            "api_key": st.sidebar.text_input(
                "API Key",
                value=os.getenv("OPENAI_API_KEY", ""),
                type="password",
            ),
            "base_url": st.sidebar.text_input(
                "Base URL",
                value=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            ),
            "model": st.sidebar.text_input(
                "模型名",
                value=os.getenv("OPENAI_MODEL", ""),
                placeholder="例如 gpt-4.1-mini，或兼容平台的模型名",
            ),
        }

    return {}


def collect_retrieval_config(page: str) -> dict[str, str]:
    if page not in {"知识库问答", "数据分析与解释"}:
        return {"method": HYBRID_RETRIEVAL, "embedding_model": DEFAULT_EMBEDDING_MODEL}

    st.sidebar.divider()
    method = st.sidebar.selectbox(
        "知识库检索方式",
        [HYBRID_RETRIEVAL, EMBEDDING_RETRIEVAL, TFIDF_RETRIEVAL],
        help="Hybrid 会融合关键词和语义检索；TF-IDF 适合关键词匹配，Embedding 适合语义相近但用词不同的问题。",
    )
    embedding_model = DEFAULT_EMBEDDING_MODEL
    if method in {HYBRID_RETRIEVAL, EMBEDDING_RETRIEVAL}:
        embedding_model = st.sidebar.text_input(
            "Embedding 模型",
            value=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            help="第一次使用本地未缓存的模型时，sentence-transformers 会下载模型文件。",
        ).strip() or DEFAULT_EMBEDDING_MODEL
    return {"method": method, "embedding_model": embedding_model}


def load_selected_retriever(retrieval_config: dict[str, str]) -> Retriever | None:
    method = retrieval_config.get("method", TFIDF_RETRIEVAL)
    embedding_model = resolve_embedding_model_path(
        retrieval_config.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
    )
    try:
        with st.spinner(f"正在加载{method}..."):
            return get_retriever(method, embedding_model)
    except Exception as exc:
        st.error(f"检索器加载失败：{exc}")
        return None


def resolve_embedding_model_path(model_name: str) -> str:
    path = Path(model_name).expanduser()
    if path.is_absolute() or "/" not in model_name:
        return str(path)

    project_path = ROOT / path
    if project_path.exists():
        return str(project_path)
    return model_name


def generate_answer(provider: str, config: dict[str, str], prompt: str) -> str:
    try:
        if provider in {"OpenAI / 兼容 API", "DeepSeek"}:
            missing = [key for key in ("api_key", "base_url", "model") if not config.get(key)]
            if missing:
                return f"还缺少配置：{', '.join(missing)}。请在左侧填好后再生成。"
            return answer_openai_compatible(
                api_key=config["api_key"],
                base_url=config["base_url"],
                model=config["model"],
                prompt=prompt,
            )

        if provider == "Ollama 本地模型":
            missing = [key for key in ("base_url", "model") if not config.get(key)]
            if missing:
                return f"还缺少配置：{', '.join(missing)}。请在左侧填好后再生成。"
            return answer_ollama(
                base_url=config["base_url"],
                model=config["model"],
                prompt=prompt,
            )
    except Exception as exc:
        return f"模型调用失败：{exc}"

    return "当前模型类型还没有配置生成逻辑。"


def is_model_answer(answer: str) -> bool:
    return not answer.startswith(("还缺少配置", "模型调用失败", "当前模型类型"))


def render_rag_history() -> None:
    st.divider()
    left, right = st.columns([1, 1])
    left.subheader("知识库问答历史")
    if right.button("清空知识库问答历史"):
        st.session_state.rag_history = []
        st.rerun()

    if not st.session_state.rag_history:
        st.caption("本次运行期间还没有知识库问答历史。")
        return

    for index, item in enumerate(st.session_state.rag_history, start=1):
        with st.expander(f"{index}. {item['question']}", expanded=index == 1):
            st.markdown(item["answer"])
            if item.get("results"):
                st.caption("本次召回片段")
                for result in item["results"]:
                    st.write(
                        f"{result.chunk.source} / {result.chunk.title} - 相关度 {result.score:.3f}"
                    )


def render_direct_history() -> None:
    st.divider()
    left, right = st.columns([1, 1])
    left.subheader("直接问模型历史")
    if right.button("清空直接问模型历史"):
        st.session_state.direct_history = []
        st.rerun()

    if not st.session_state.direct_history:
        st.caption("本次运行期间还没有直接问模型历史。")
        return

    for index, item in enumerate(st.session_state.direct_history, start=1):
        with st.expander(f"{index}. {item['question']}", expanded=index == 1):
            st.markdown(item["answer"])


def show_results(results) -> None:
    st.subheader("召回片段")
    for result in results:
        with st.expander(
            f"{result.chunk.source} / {result.chunk.title} - 相关度 {result.score:.3f}"
        ):
            st.write(result.chunk.text)


if __name__ == "__main__":
    main()
