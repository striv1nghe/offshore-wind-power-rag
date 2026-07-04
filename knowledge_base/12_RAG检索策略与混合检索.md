# RAG 检索策略与混合检索

## 适用问题
- RAG 的基本流程是什么？
- 为什么风电预测解释系统需要 RAG？
- TF-IDF、BM25、embedding 和 dense retrieval 有什么区别？
- 为什么要做 hybrid retrieval，而不是只用关键词检索或只用向量检索？
- RRF、rerank、粗排和精排分别解决什么问题？

## 核心概念
- RAG 是检索增强生成流程，通常包括文档切分、索引构建、查询编码、候选召回、排序、上下文拼接和答案生成。
- TF-IDF 根据词频和逆文档频率衡量关键词匹配程度，适合专业术语、故障名、变量名和精确短语检索。
- BM25 是经典稀疏检索方法，在关键词匹配中考虑词频饱和和文档长度归一化，常作为强基线。
- Embedding 检索把查询和文档编码成向量，通过相似度召回语义相关内容，适合自然语言改写、同义表达和语义近邻问题。
- Dense retrieval 通常使用神经网络编码器学习查询和文档的语义匹配关系，可用于开放域问答召回。
- Hybrid retrieval 将关键词检索和向量检索结合，兼顾术语精确匹配和语义泛化能力。
- RRF 是 Reciprocal Rank Fusion，用于融合多个检索器的排名结果，不需要额外训练，适合工程系统快速增强稳健性。
- 粗排负责快速召回候选文档，精排或 rerank 负责对候选文档做更细粒度相关性判断。
- Rerank 可使用交叉编码器、大模型打分或规则打分，但需要权衡准确率、延迟和成本。

## 近年论文/资料结论
- Ram 等在 TACL 中研究 in-context retrieval-augmented language models，说明检索到的示例或文档可在上下文中改变语言模型输出，检索质量会直接影响生成质量。
- Park 和 Lee 在 TACL 中研究不完美检索对 retrieval-augmented language models 的影响，强调 RAG 系统必须关注错误召回、噪声文档和检索鲁棒性。
- Bruch、Gai 和 Ingber 在 ACM TOIS 中系统分析 hybrid retrieval 的融合函数，说明稀疏检索和稠密检索的融合不是简单相加，融合策略会显著影响排序质量。
- 近年 RAG/RALM 文献共同指向：RAG 不是“把文档塞进提示词”，而是由文档质量、切块、召回、融合排序、rerank 和生成约束共同决定效果。

## 工程含义
- 风电功率预测解释 RAG 系统适合保留 TF-IDF 检索，因为用户可能直接问“切入风速”“限功率”“桨距角异常”“SHAP 值”等精确术语。
- 系统也需要 embedding 检索，因为用户可能用口语化表达，例如“为什么风很大但发电不多”，这类问题和“限功率、偏航误差、桨距角控制”存在语义关联。
- Hybrid retrieval 可以先分别用 TF-IDF/BM25 和 embedding 召回候选，再用 RRF、加权融合或学习排序方法合并结果。
- 粗排阶段应追求召回全面，避免漏掉关键机制；精排阶段应追求相关性，减少无关知识进入大模型上下文。
- 如果使用 BAAI/bge-small-zh-v1.5，适合中文语义召回，但仍应搭配关键词检索处理英文缩写、变量名、论文术语和精确故障名。
- RAG 知识卡片应按 `##` 切分，保证每个 chunk 主题集中，降低检索时混入无关内容的概率。

## 可用于问答的模板
- 问：为什么不能只用 embedding 检索？
  - 回答思路：embedding 擅长语义相似，但可能漏掉精确变量名、缩写和专业术语；风电场景中这些术语往往是关键证据。
- 问：为什么不能只用 TF-IDF 或 BM25？
  - 回答思路：关键词检索依赖字面匹配，用户换一种说法时可能召回不到同义知识，例如“风很大但功率低”可能对应限功率或偏航误差。
- 问：RRF 在混合检索中有什么作用？
  - 回答思路：RRF 可以把多个检索器的排名融合起来，让同时被多个检索器认为相关的文档排得更靠前。
- 问：粗排和精排有什么区别？
  - 回答思路：粗排快速召回较多候选，精排或 rerank 再从候选中挑出最适合回答当前问题的证据。
- 问：RAG 为什么能减少幻觉？
  - 回答思路：RAG 让模型基于检索到的知识片段回答，而不是完全依赖参数记忆；但前提是检索结果相关、来源可靠且排序正确。

## 正式来源
- 标题：In-Context Retrieval-Augmented Language Models
  - 作者：Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay 等
  - 年份：2023
  - 期刊：Transactions of the Association for Computational Linguistics
  - DOI / 链接：https://doi.org/10.1162/tacl_a_00605
  - 分区：JCR Q1
- 标题：Toward Robust RALMs: Revealing the Impact of Imperfect Retrieval on Retrieval-Augmented Language Models
  - 作者：Seong-Il Park, Jay-Yoon Lee
  - 年份：2024
  - 期刊：Transactions of the Association for Computational Linguistics
  - DOI / 链接：https://doi.org/10.1162/tacl_a_00724
  - 分区：JCR Q1
- 标题：An Analysis of Fusion Functions for Hybrid Retrieval
  - 作者：Sebastian Bruch, Siyu Gai, Amir Ingber
  - 年份：2024
  - 期刊：ACM Transactions on Information Systems
  - DOI / 链接：https://doi.org/10.1145/3596512
  - 分区：未查到分区

## 经典背景来源
- 标题：The Probabilistic Relevance Framework: BM25 and Beyond
  - 作者：Stephen Robertson, Hugo Zaragoza
  - 年份：2009
  - 来源：Foundations and Trends in Information Retrieval
  - DOI / 链接：https://doi.org/10.1561/1500000019
  - 说明：BM25 与概率相关性框架的经典背景来源。
- 标题：Reciprocal rank fusion outperforms condorcet and individual rank learning methods
  - 作者：Gordon V. Cormack, Charles L. A. Clarke, Stefan Buettcher
  - 年份：2009
  - 来源：SIGIR
  - DOI / 链接：https://doi.org/10.1145/1571941.1572114
  - 说明：RRF 排名融合的经典顶会背景来源。
- 标题：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
  - 作者：Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin 等
  - 年份：2020
  - 来源：NeurIPS
  - DOI / 链接：无 DOI；会议论文链接：https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html
  - 说明：RAG 经典顶会背景来源，不作为近年正式期刊来源。
- 标题：Dense Passage Retrieval for Open-Domain Question Answering
  - 作者：Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih
  - 年份：2020
  - 来源：EMNLP
  - DOI / 链接：https://doi.org/10.18653/v1/2020.emnlp-main.550
  - 说明：DPR 稠密检索经典顶会背景来源。
- 标题：ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT
  - 作者：Omar Khattab, Matei Zaharia
  - 年份：2020
  - 来源：SIGIR
  - DOI / 链接：https://doi.org/10.1145/3397271.3401075
  - 说明：late interaction 神经检索经典背景来源。
- 标题：SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking
  - 作者：Thibault Formal, Benjamin Piwowarski, Stephane Clinchant
  - 年份：2021
  - 来源：SIGIR
  - DOI / 链接：https://doi.org/10.1145/3404835.3463098
  - 说明：稀疏神经检索与词项扩展的经典背景来源。
