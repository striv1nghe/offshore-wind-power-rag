# SHAP 可解释性论文

## 适用问题
- SHAP 是什么，如何解释风电功率预测模型？
- 全局解释和局部解释有什么区别？
- 特征重要性如何用于风电预测误差归因？
- SHAP 结果如何与风速-功率曲线、限功率和 SCADA 异常结合？
- 为什么不能把 SHAP 值直接等同于物理因果关系？

## 核心概念
- SHAP 基于 Shapley value 思想，用于衡量每个特征对模型输出的边际贡献。
- 局部解释关注单个样本：例如某一时刻预测偏高，是哪些特征把预测值推高或拉低。
- 全局解释关注一批样本：例如在测试集中，风速、历史功率、风向、桨距角等特征整体上谁更重要。
- SHAP 值为正通常表示该特征把模型输出推高，SHAP 值为负通常表示该特征把模型输出拉低。
- SHAP 解释的是模型行为，不是直接证明真实世界因果关系。
- 在风电场景中，SHAP 应与风电机理结合解释：风速贡献高可联系功率曲线非线性，桨距角贡献异常可联系控制策略或限功率，偏航相关特征异常可联系有效入流减少。
- SHAP 可用于误差归因，但需要同时检查 SCADA 状态、数据质量、风速区间和功率曲线位置。

## 近年论文/资料结论
- Liao 等在 Applied Energy 中专门讨论风电功率预测中的 XAI 可信性问题，提醒使用者不能只把解释图当作“正确原因”，还要检查解释稳定性、数据分布和模型行为。
- Letzgus 和 Müller 在 Energy and AI 中提出面向风机功率曲线模型的 XAI 框架，强调可解释性可用于检查数据驱动模型是否学到合理的风机功率曲线行为。
- van Zyl、Ye 和 Naidoo 在 Applied Energy 中比较 SHAP 与 Grad-CAM 在能源时间序列预测特征选择中的作用，说明 XAI 不仅可用于结果解释，也可辅助特征筛选和模型诊断。
- 近年能源领域 XAI 论文共同强调：SHAP 输出需要转换成工程语言，并结合风速-功率曲线、SCADA 状态和异常工况进行解释。

## 工程含义
- RAG 系统回答 SHAP 问题时，应采用“SHAP 数值解释 + 风电机理解释 + 异常工况检查”的组合表达。
- 如果风速 SHAP 值很大，应结合功率曲线说明不同风速区间对功率变化的敏感性不同。
- 如果历史功率 SHAP 值很大，应说明模型可能依赖短期惯性或自相关，但这不代表未来功率一定延续历史趋势。
- 如果桨距角、偏航角、状态码相关特征的 SHAP 值异常，应检索 SCADA 异常、限功率、偏航误差和桨距角异常知识。
- 系统应提醒用户：SHAP 是模型解释工具，不能单独替代故障诊断或机理分析。
- 对用户友好的解释可以按三步组织：这个特征如何影响模型输出、该影响是否符合风电机理、还需要检查哪些数据或状态。

## 可用于问答的模板
- 问：SHAP 值高说明什么？
  - 回答思路：说明该特征对模型当前预测贡献较大，但不等于它一定是物理因果原因，需要结合机理和数据状态解释。
- 问：风速 SHAP 值很大，如何解释？
  - 回答思路：风速是风电功率核心驱动变量，尤其在功率曲线快速上升区，小风速变化会导致较大功率变化。
- 问：为什么实际功率低，但模型预测高，SHAP 显示风速贡献很大？
  - 回答思路：模型可能根据高风速推高预测，但实际机组可能处于限功率、停机、偏航误差或桨距控制异常状态。
- 问：全局 SHAP 和局部 SHAP 有什么区别？
  - 回答思路：全局 SHAP 用于看整体特征重要性，局部 SHAP 用于解释某一个时间点或样本的预测原因。
- 问：如何避免误用 SHAP？
  - 回答思路：不要只看特征排名，应结合功率曲线、SCADA 状态、异常检测和实际运行机理共同判断。

## 正式来源
- 标题：Can we trust explainable artificial intelligence in wind power forecasting?
  - 作者：Wenlong Liao, Jiannong Fang, Lin Ye, Birgitte Bak-Jensen
  - 年份：2024
  - 期刊：Applied Energy
  - DOI / 链接：https://doi.org/10.1016/j.apenergy.2024.124273
  - 分区：JCR Q1
- 标题：An explainable AI framework for robust and transparent data-driven wind turbine power curve models
  - 作者：Simon Letzgus, Klaus-Robert Müller
  - 年份：2024
  - 期刊：Energy and AI
  - DOI / 链接：https://doi.org/10.1016/j.egyai.2023.100328
  - 分区：JCR Q1
- 标题：Harnessing eXplainable artificial intelligence for feature selection in time series energy forecasting: A comparative analysis of Grad-CAM and SHAP
  - 作者：Corne van Zyl, Xianming Ye, Raj Naidoo
  - 年份：2024
  - 期刊：Applied Energy
  - DOI / 链接：https://doi.org/10.1016/j.apenergy.2023.122079
  - 分区：JCR Q1

## 经典背景来源
- 标题：A Unified Approach to Interpreting Model Predictions
  - 作者：Scott M. Lundberg, Su-In Lee
  - 年份：2017
  - 来源：NeurIPS
  - DOI / 链接：无 DOI；会议论文链接：https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
  - 说明：SHAP 原理的经典顶会背景来源，不作为近年正式期刊来源。
