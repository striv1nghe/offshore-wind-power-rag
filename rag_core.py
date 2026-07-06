from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
import requests
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Chunk:
    source: str
    title: str
    text: str


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 4) -> list[SearchResult]:
        ...


def load_markdown_chunks(knowledge_dir: str | Path) -> list[Chunk]:
    root = Path(knowledge_dir)
    chunks: list[Chunk] = []

    for path in sorted(root.glob("*.md")):
        if path.name == "README.md":
            continue
        text = path.read_text(encoding="utf-8")
        chunks.extend(_split_markdown(path.name, text))

    if not chunks:
        raise ValueError(f"No markdown chunks found in {root}")
    return chunks


def _split_markdown(source: str, text: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    current_title = Path(source).stem
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            _append_chunk(chunks, source, current_title, current_lines)
            current_title = line.removeprefix("## ").strip() or current_title
            current_lines = []
        else:
            current_lines.append(raw_line)

    _append_chunk(chunks, source, current_title, current_lines)
    return chunks


def _append_chunk(chunks: list[Chunk], source: str, title: str, lines: list[str]) -> None:
    cleaned_lines = [line for line in lines if not line.strip().startswith("# ")]
    body = "\n".join(cleaned_lines).strip()
    if body:
        chunks.append(Chunk(source=source, title=title, text=body))


class TfidfRetriever:
    def __init__(self, chunks: Iterable[Chunk]):
        self.chunks = list(chunks)
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
        self.matrix = self.vectorizer.fit_transform(
            [chunk_to_search_text(chunk) for chunk in self.chunks]
        )

    def search(self, query: str, top_k: int = 4) -> list[SearchResult]:
        query_vector = self.vectorizer.transform([expand_query(query)])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked = scores.argsort()[::-1][:top_k]
        return [
            SearchResult(chunk=self.chunks[index], score=float(scores[index]))
            for index in ranked
            if scores[index] > 0
        ]


class EmbeddingRetriever:
    def __init__(
        self,
        chunks: Iterable[Chunk],
        model_name: str = "BAAI/bge-small-zh-v1.5",
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Embedding 检索需要安装 sentence-transformers。"
                "请运行 pip install -r requirements.txt 后重试。"
            ) from exc

        self.chunks = list(chunks)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.matrix = self.model.encode(
            [chunk_to_search_text(chunk) for chunk in self.chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def search(self, query: str, top_k: int = 4) -> list[SearchResult]:
        query_vector = self.model.encode(
            [expand_query(query)],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = np.asarray(self.matrix @ query_vector).ravel()
        ranked = scores.argsort()[::-1][:top_k]
        return [
            SearchResult(chunk=self.chunks[index], score=float(scores[index]))
            for index in ranked
            if scores[index] > 0
        ]


class HybridRetriever:
    def __init__(
        self,
        chunks: Iterable[Chunk],
        embedding_model_name: str = "BAAI/bge-small-zh-v1.5",
        rrf_k: int = 60,
        candidate_multiplier: int = 3,
    ):
        self.chunks = list(chunks)
        self.tfidf_retriever = TfidfRetriever(self.chunks)
        self.embedding_retriever = EmbeddingRetriever(
            self.chunks,
            model_name=embedding_model_name,
        )
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier

    def search(self, query: str, top_k: int = 4) -> list[SearchResult]:
        candidate_k = max(top_k * self.candidate_multiplier, 8)
        result_lists = [
            self.tfidf_retriever.search(query, top_k=candidate_k),
            self.embedding_retriever.search(query, top_k=candidate_k),
        ]
        scores: dict[Chunk, float] = {}

        for results in result_lists:
            for rank, result in enumerate(results, start=1):
                scores[result.chunk] = scores.get(result.chunk, 0.0) + (
                    1.0 / (self.rrf_k + rank)
                )

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [
            SearchResult(chunk=chunk, score=float(score))
            for chunk, score in ranked[:top_k]
        ]


def chunk_to_search_text(chunk: Chunk) -> str:
    return f"{chunk.title}\n{chunk.text}"


def expand_query(query: str) -> str:
    """Add domain terms for common wind-power diagnostic phrasings."""
    compact_query = "".join(query.split())
    expansions: list[str] = []

    high_wind_low_power = any(
        phrase in compact_query
        for phrase in [
            "风很大但发电不多",
            "风大但发电不多",
            "风速高但功率低",
            "高风速低功率",
            "风速高但发电少",
            "风速高但功率不上升",
            "风速高但功率没有继续上升",
            "风很大但功率低",
        ]
    )
    if high_wind_low_power:
        expansions.append(
            "高风速低功率 限功率 限电 调度限发 偏航误差 桨距角 "
            "变桨控制 设备保护 停机维护 SCADA异常 功率曲线偏离 模型高估"
        )

    return "\n".join([query, *expansions])


def build_prompt(question: str, results: list[SearchResult]) -> str:
    context = "\n\n".join(
        f"[{idx}] 来源: {result.chunk.source} / {result.chunk.title}\n{result.chunk.text}"
        for idx, result in enumerate(results, start=1)
    )
    return f"""你是风电功率预测领域的 RAG 助手。请只根据给定知识库回答问题。
如果知识库没有足够信息，请明确说不知道，并说明还需要什么资料。

知识库片段:
{context}

用户问题:
{question}

请用中文回答。不要在正文中使用 [1]、[2] 这样的引用编号，也不要自己编写参考来源部分。
系统会在回答后自动追加统一格式的参考来源。"""


def build_direct_prompt(question: str) -> str:
    return f"""请用中文回答用户问题。
如果问题涉及工程或数据分析，请尽量给出清晰、结构化、可操作的解释。

用户问题:
{question}"""


def build_data_explanation_prompt(
    *,
    question: str,
    data_summary: str,
    results: list[SearchResult],
) -> str:
    context = "\n\n".join(
        f"[{idx}] 来源: {result.chunk.source} / {result.chunk.title}\n{result.chunk.text}"
        for idx, result in enumerate(results, start=1)
    )
    return f"""你是风电功率预测误差分析助手。请结合数值摘要和知识库片段解释预测误差。
不要编造数值摘要中没有出现的数据；如果证据不足，请明确说明还需要哪些数据。

数值摘要:
{data_summary}

知识库片段:
{context}

用户问题:
{question}

请用中文回答，重点说明：
1. 当前数据的预测误差表现；
2. 哪些输入变量可能与误差相关；
3. 结合风电机理给出可能原因；
4. 建议下一步如何验证。"""


def format_reference_footer(results: list[SearchResult]) -> str:
    source_counts: dict[str, int] = {}
    for result in results:
        source_counts[result.chunk.source] = source_counts.get(result.chunk.source, 0) + 1

    has_repeated_source = any(count > 1 for count in source_counts.values())
    seen: set[tuple[str, str]] = set()
    lines = ["参考来源："]

    for result in results:
        if has_repeated_source and source_counts[result.chunk.source] == 1:
            continue
        key = (result.chunk.source, result.chunk.title)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"\\- {result.chunk.source}：{result.chunk.title}")

    return "\n".join(lines)


def append_reference_footer(answer: str, results: list[SearchResult]) -> str:
    return f"{answer.strip()}\n\n{format_reference_footer(results)}"


def answer_openai_compatible(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float = 0.2,
) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def answer_ollama(
    *,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float = 0.2,
) -> str:
    response = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("message", {}).get("content", "")
