# SCADA 异常与数据清洗论文

## 适用问题
- SCADA 数据中哪些异常会导致风电功率预测误差变大？
- 限功率、停机、传感器异常、偏航误差和桨距角异常如何影响功率曲线？
- 为什么训练预测模型前需要清洗风机运行数据？
- 如何根据功率曲线散点识别异常工况？
- 为什么模型预测功率很高，但实际功率明显偏低？

## 核心概念
- SCADA 数据通常包括风速、风向、有功功率、转速、桨距角、偏航角、温度、状态码等机组运行变量。
- 正常运行数据应大体符合风机功率曲线：低于切入风速时功率接近 0，额定风速附近快速上升，达到额定功率后进入平台区。
- 限功率会让实际功率低于可发功率，常在功率曲线中表现为低于额定功率的水平带、截断带或异常平台。
- 停机或待机会导致在有风条件下功率接近 0，如果混入训练数据，会让模型错误学习风速和功率的关系。
- 传感器异常可能表现为风速、功率、风向、桨距角等变量突变、冻结、缺失或互相矛盾。
- 偏航误差会降低风轮捕获的有效风能，使同等风速下实际功率偏低。
- 桨距角异常或控制策略变化会改变叶片气动效率，导致功率曲线偏移、额定区间异常或功率被主动压低。
- 数据清洗的目标是识别非正常运行点，而不是机械删除所有离群点；异常点本身也可作为误差解释证据。

## 近年论文/资料结论
- Morrison、Liu 和 Lin 在 Renewable Energy 中研究 SCADA 异常检测与功率曲线清洗，指出异常运行点会污染功率曲线建模，影响后续预测模型学习正常风速-功率关系。
- Long、Xu 和 Gu 在 Applied Energy 中把异常风机数据清洗转化为颜色空间转换和图像特征检测问题，说明功率曲线散点形态本身就是异常识别线索。
- Luo 等在 IEEE Transactions on Sustainable Energy 中提出基于密度聚类和边界提取的异常功率曲线清洗方法，适合识别功率曲线中的异常簇和异常边界。
- 近年功率曲线清洗研究共同强调，异常点可能来自限功率、停机、传感器问题、低效运行、控制策略变化或复杂风况。
- 在风电预测解释中，功率曲线下方的密集异常点常提示限功率、偏航问题、桨距角控制介入或非正常运行。
- 高风速下功率没有达到额定平台时，需要结合桨距角、状态码、风向/偏航信息和机组保护逻辑判断，而不能只把误差归因给预测模型。
- 风速正常但功率为 0 的点，通常需要结合状态码判断是否停机、故障、维护、通信异常或数据缺失。
- 该主题 2023-2026 年可核验的高水平正式期刊来源相对少，本文件正式来源主要集中在 2022 年；旧一些的 IEEE TSTE 研究放在经典背景来源中。

## 工程含义
- 风电功率预测解释 RAG 系统应把 SCADA 异常作为误差归因的重要证据，而不是只解释模型结构。
- 当用户问“预测偏高但实际偏低”时，系统应优先检索限功率、停机、偏航误差、桨距角异常和功率曲线异常相关知识。
- 当用户问“为什么需要数据清洗”时，系统可解释：清洗是为了让模型学习正常功率曲线，而不是把异常工况当成正常规律。
- 异常数据不一定要完全丢弃；它们可以作为误差解释、异常告警和模型失效分析的知识证据。
- 预测解释可采用“功率曲线位置 + SCADA 状态 + 异常类型 + SHAP 特征贡献”的组合框架。

## 可用于问答的模板
- 问：为什么模型预测功率很高，但实际功率很低？
  - 回答思路：可能是限功率、停机、偏航误差、桨距角控制异常或传感器异常，需结合功率曲线和 SCADA 状态码判断。
- 问：什么样的功率曲线散点可能表示限功率？
  - 回答思路：在额定功率以下出现明显水平带、截断带或同一风速下功率被压低，常提示限功率或控制策略介入。
- 问：停机数据为什么会影响模型训练？
  - 回答思路：停机点会让模型看到“有风但无功率”的样本，如果未标记或清洗，会破坏正常风速-功率映射。
- 问：偏航误差如何导致预测误差？
  - 回答思路：偏航误差使风轮不能正对来流，实际捕获能量下降，同等风速下实际功率低于预测。
- 问：桨距角异常为什么会改变功率输出？
  - 回答思路：桨距角影响叶片气动效率和机组控制状态，异常或保护控制会使实际功率偏离正常功率曲线。

## 正式来源
- 标题：Anomaly detection in wind turbine SCADA data for power curve cleaning
  - 作者：Rory Morrison, Xiaolei Liu, Zi Lin
  - 年份：2022
  - 期刊：Renewable Energy
  - DOI / 链接：https://doi.org/10.1016/j.renene.2021.11.118
  - 分区：JCR Q1
- 标题：An abnormal wind turbine data cleaning algorithm based on color space conversion and image feature detection
  - 作者：Huan Long, Shaohui Xu, Wei Gu
  - 年份：2022
  - 期刊：Applied Energy
  - DOI / 链接：https://doi.org/10.1016/j.apenergy.2022.118594
  - 分区：JCR Q1
- 标题：Method for Cleaning Abnormal Data of Wind Turbine Power Curve Based on Density Clustering and Boundary Extraction
  - 作者：Zhihong Luo, Chengyue Fang, Changliang Liu, Shuai Liu
  - 年份：2022
  - 期刊：IEEE Transactions on Sustainable Energy
  - DOI / 链接：https://doi.org/10.1109/TSTE.2021.3138757
  - 分区：JCR Q1

## 经典背景来源
- 标题：Image-Based Abnormal Data Detection and Cleaning Algorithm via Wind Power Curve
  - 作者：Huan Long, Linwei Sang, Zaijun Wu, Wei Gu
  - 年份：2020
  - 来源：IEEE Transactions on Sustainable Energy
  - DOI / 链接：https://doi.org/10.1109/TSTE.2019.2914089
  - 说明：经典功率曲线图像化异常清洗研究，略早于近 4 年窗口。
- 标题：A Combined Algorithm for Cleaning Abnormal Data of Wind Turbine Power Curve Based on Change Point Grouping Algorithm and Quartile Algorithm
  - 作者：Xiaojun Shen, Xuejiao Fu, Chongcheng Zhou
  - 年份：2019
  - 来源：IEEE Transactions on Sustainable Energy
  - DOI / 链接：https://doi.org/10.1109/TSTE.2018.2822682
  - 说明：经典功率曲线异常清洗方法，可作为变点与分位数清洗背景。
