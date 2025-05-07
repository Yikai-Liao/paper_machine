---
title: "Representation Learning of Limit Order Book: A Comprehensive Study and Benchmarking"
pubDatetime: 2025-05-04T15:00:00+00:00
slug: "2025-05-lob-representation-benchmark"
type: "arxiv"
id: "2505.02139"
score: 0.5975647138274283
author: "grok-3-latest"
authors: ["Muyao Zhong", "Yushi Lin", "Peng Yang"]
tags: ["Representation Learning", "Time Series", "Financial Modeling", "Benchmark Framework", "Feature Extraction"]
institution: ["Harbin Institute of Technology", "Southern University of Science and Technology"]
description: "本文通过引入 LOBench 基准，系统研究限价订单簿（LOB）表示学习，证明其在捕捉市场信息和支持下游任务中的充分性和必要性，为金融建模提供可复用的解决方案。"
---

> **Summary:** 本文通过引入 LOBench 基准，系统研究限价订单簿（LOB）表示学习，证明其在捕捉市场信息和支持下游任务中的充分性和必要性，为金融建模提供可复用的解决方案。 

> **Keywords:** Representation Learning, Time Series, Financial Modeling, Benchmark Framework, Feature Extraction

**Authors:** Muyao Zhong, Yushi Lin, Peng Yang

**Institution(s):** Harbin Institute of Technology, Southern University of Science and Technology


## Problem Background

限价订单簿（LOB）作为金融市场核心数据结构，记录未成交买卖订单，反映市场动态，但其高维、非平稳、强自相关、跨特征约束和特征尺度差异等特性使得传统深度学习模型难以有效建模。
现有研究多将表示学习与特定下游任务耦合，缺乏对表示本身的独立分析，限制了模型复用性和泛化能力。
本文旨在探索如何提取可转移、紧凑的 LOB 表示，捕捉其本质属性，提升模型在不同任务和数据集上的表现。

## Method

*   **核心框架：LOBench 基准**：基于中国 A 股市场真实数据，提出标准化框架，提供统一数据格式、预处理流程和评估指标，旨在解耦表示学习与下游任务。
*   **数据预处理**：从深圳证券交易所（SZSE）获取 2019 年 LOB 数据，选择五只代表性股票，以 3 秒频率采样，构建 10 级深度快照；采用全局 z 分数归一化（Global Z-Score）而非特征级归一化，以保留价格层级间的结构约束；通过前向填充处理缺失数据，剔除集中竞价阶段噪声数据。
*   **模型设计与对比**：包括三类模型：基础模型（如 CNN、LSTM、Transformer）用于通用基准；通用时间序列模型（如 iTransformer、TimesNet、TimeMixer）测试跨领域适用性；LOB 专用模型（如 DeepLOB、TransLOB、SimLOB）针对领域特性优化。
*   **下游任务设置**：设计价格趋势预测（分类任务，预测中价走势）、数据插补（回归任务，填补缺失快照）和重构任务（无监督学习，重构原始 LOB），以评估表示的充分性和必要性。
*   **评估方法创新**：提出综合损失函数 *L_All*，结合 MSE、加权 MSE 和结构正则化项（惩罚价格层级倒置），确保重构结果符合 LOB 经济意义和结构特性。
*   **实验流程**：所有模型统一编码到 256 维表示空间，使用 PyTorch Lightning 实现，Adam 优化器训练 100 轮，记录训练时间和多指标表现（如 MSE、MAE、wMSE）。

## Experiment

*   **重构任务效果**：大多数模型能有效将 LOB 数据映射到低维空间并重构，TimesNet 在 MSE 和 wMSE 上表现最佳（误差在 1e-3 量级），Transformer 架构整体优于 CNN 和 LSTM，但 TimesNet 训练时间较长（例如在 sz000001 数据集上耗时 9905 秒，远高于 CNN 的 2047 秒）。
*   **下游任务表现**：表示学习质量直接影响下游任务，TimesNet 和 iTransformer 在价格趋势预测和数据插补中表现优异（预测任务中交叉熵损失较低），而 DeepLOB 和 LSTM 表现较差，验证了表示学习对任务支持的必要性。
*   **转移性验证**：在 SZ000001 数据集训练的模型直接测试其他数据集，SimLOB-freeze（冻结编码器，仅微调解码器）平均精度达 0.7332，优于端到端模型，表明表示学习具有高转移性。
*   **实验设置评价**：实验覆盖五只股票、三个任务，评估指标全面（包括 MSE、MAE、wMSE、价格和体积分解损失等），并记录计算成本，设置合理且对比充分；但数据集规模较小（仅五只股票），可能限制多样性，未来可扩展更多数据。
*   **总体提升**：相比传统端到端模型，表示学习显著提升了模型泛化能力和开发效率，尤其在转移性和下游任务支持上效果明显。

## Further Thoughts

论文提出未来研究 LOB 表示的几何和语义一致性，这启发我思考是否可以通过嵌入空间可视化或聚类分析，验证相似市场状态在表示空间的接近性以提升可解释性；此外，是否可将 LOBench 扩展至其他市场（如美股）验证普适性，或引入在线学习机制适应市场动态，甚至结合新闻、宏观经济数据构建多模态表示学习框架，进一步提升预测能力。