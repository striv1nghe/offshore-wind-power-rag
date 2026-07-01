# Offshore Wind Power Prediction RAG

基于 Streamlit、DeepSeek 和本地知识库的海上风电功率预测解释系统。项目支持知识库问答、直接问大模型、预测结果可视化、特征分析和误差解释。

## 功能概览

- 知识库问答：读取 `knowledge_base/` 中的 Markdown 文档，检索相关片段后交给大模型生成回答。
- 直接问模型：绕过知识库，直接调用 DeepSeek、OpenAI 兼容接口或本地 Ollama。
- 数据概览：读取本地 CSV，展示数据集数量、数据总量、预测样本数、RMSE、MAE 等信息。
- 预测可视化：展示真实功率、预测功率、预测误差、绝对误差分布和训练/验证 loss。
- 特征分析：计算输入变量与绝对误差的 Spearman 相关性。
- 误差解释：将数据摘要和知识库片段一起发送给大模型，生成误差原因解释。

## 项目结构

```text
.
├── rag_app.py              # Streamlit 页面入口
├── rag_core.py             # RAG 检索、Prompt 构造和模型调用逻辑
├── wind_data.py            # 风电 CSV 数据读取、合并、统计和摘要逻辑
├── knowledge_base/         # 风电预测解释知识库
├── data/                   # 本地数据目录，真实 CSV 未上传
├── requirements.txt        # pip 依赖
├── environment.yml         # Conda 环境配置
├── .env.example            # API 配置模板
└── .gitignore              # Git 忽略规则
```

## 环境安装

推荐使用 Conda：

```bash
conda env create -f environment.yml
conda activate TRAG
```

如果环境已经创建过：

```bash
conda activate TRAG
```

也可以用 pip 安装依赖：

```bash
pip install -r requirements.txt
```

## 配置 DeepSeek

复制 `.env.example` 为 `.env`，然后填写自己的 API Key：

```env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

也可以不写 `.env`，在 Streamlit 页面左侧手动输入 API Key。



## 运行项目

在项目根目录运行：

```bash
conda activate TRAG
streamlit run rag_app.py
```

浏览器打开：

```text
http://localhost:8501
```

关闭 Streamlit：

```text
Ctrl + C
```

## 数据目录

项目默认从 `data/` 读取 CSV。

期望目录结构：

```text
data/
├── INPUT/
│   └── *_10min_top10_features.csv
├── PRED_RESULT/
│   └── *_step1_true_pred_inverse.csv
├── METRIC_RESULT/
│   └── *_step1_metrics_loss.csv
└── LOSS_DATA/
    └── *_loss.csv
```

主要字段：

- `时间`
- `True_Step1_KW`
- `Pred_Step1_KW`
- `Error_KW`
- `Abs_Error_KW`
- `APE`
- 皮尔逊筛选后的输入变量列

页面展示时会把 `True_Step1_KW`、`Pred_Step1_KW` 等列名转成更易读的中文名称。

## 页面说明

### 知识库问答

适合回答风电预测解释类问题，例如：

```text
为什么风速波动会导致风电功率预测误差变大？
```

系统会先检索 `knowledge_base/`，再让大模型基于召回片段回答。

### 直接问模型

不使用知识库，直接向当前选择的大模型提问。适合问通用概念，例如：

```text
请解释 RMSE 和 MAE 的区别。
```

### 数据概览

展示数据集总量、预测样本数量、平均 RMSE、平均 MAE，并支持选择某个数据集和变量查看时间序列。

### 数据分析与解释

包含三个标签页：

- 预测可视化：真实功率与预测功率联动图、预测误差曲线、绝对误差分布、loss 曲线。
- 特征分析：输入变量与绝对误差的 Spearman 相关性。
- 误差解释：基于数据摘要和知识库片段生成解释。

### Agent 分析助手

在 `feature/agent` 分支中新增。当前版本是轻量工作流 Agent，会按固定流程调用工具：

```text
识别数据集
-> 读取数据概览
-> 生成误差摘要
-> 分析 Spearman 相关性
-> 检索知识库
-> 调用大模型生成结构化报告
```

这版 Agent 暂时不依赖 LangChain，目的是先理解工具调用、证据组织和最终报告生成流程。


## 当前检索方式

当前知识库检索使用的是 TF-IDF 字符相似度：

```text
Markdown 文档
-> 按二级标题切分
-> TF-IDF 检索
-> 拼接 Prompt
-> 调用大模型
```

目前没有使用 embedding 或向量数据库。后续如果知识库规模变大，可以升级为 embedding + FAISS / Chroma。

## 后续方向

- 将 CSV 数据导入 SQLite，支持上传和长期管理。
- 增加 embedding 检索，提高语义召回能力。
- 增加更多高误差工况筛选规则。
- 保存问答历史和误差解释记录。
- 支持多模型结果对比。
