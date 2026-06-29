# TRAG 风电 RAG

这是一个最小可运行的风电功率预测 RAG 项目。当前已完成到“大模型选择”阶段：

- 读取 `knowledge_base/` 下的 Markdown 知识库。
- 使用本地文本相似度完成知识库召回。
- 提供 Streamlit 界面选择大模型类型。
- 预留 OpenAI 兼容 API、DeepSeek、本地 Ollama 三种生成方式。

## 启动

```bash
conda activate TRAG
streamlit run rag_app.py
```

## 配置

如需保存 API Key，可以复制 `.env.example` 为 `.env` 后填写。也可以直接在网页左侧输入。

## 当前推荐选择

如果先验证流程，选择“只测试检索”。

如果你已有云端 API Key，选择“OpenAI / 兼容 API”或“DeepSeek”。

如果你想本地运行大模型，选择“Ollama 本地模型”。
