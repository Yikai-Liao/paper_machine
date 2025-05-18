---
title: "Does Scaling Law Apply in Time Series Forecasting?"
pubDatetime: 2025-05-15T11:04:39+00:00
slug: "2025-05-alinear-forecasting-efficiency"
type: "arxiv"
id: "2505.10172"
score: 0.6838660662025718
author: "grok-3-latest"
authors: ["Zeyan Li", "Libing Chen", "Yin Tang"]
tags: ["Time Series Forecasting", "Adaptive Decomposition", "Parameter Efficiency", "Long-Term Prediction", "Lightweight Model"]
institution: ["Jinan University"]
description: "本文提出 ALinear 模型，通过自适应分解和频率衰减机制，以不到 1% 的参数量实现时间序列预测的顶级性能，尤其在超长期预测中显著优于现有模型，挑战了规模扩展定律的适用性。"
---

> **Summary:** 本文提出 ALinear 模型，通过自适应分解和频率衰减机制，以不到 1% 的参数量实现时间序列预测的顶级性能，尤其在超长期预测中显著优于现有模型，挑战了规模扩展定律的适用性。 

> **Keywords:** Time Series Forecasting, Adaptive Decomposition, Parameter Efficiency, Long-Term Prediction, Lightweight Model

**Authors:** Zeyan Li, Libing Chen, Yin Tang

**Institution(s):** Jinan University


## Problem Background

时间序列预测领域近年来模型规模急剧扩大，参数量从几十兆增长到几千兆，但性能提升与参数增长不成正比，且带来资源消耗、延迟和环境问题等挑战；此外，现有模型未能解决‘预测 horizon 困境’，即从短期到超长期预测时性能显著下降的问题，缺乏对不同预测长度下趋势和季节性成分重要性的动态调整。

## Method

* **核心思想**：提出 ALinear，一个超轻量级时间序列预测模型，通过自适应机制根据预测长度动态调整对趋势和季节性成分的处理策略，在极低参数量下实现高性能。
* **具体实现**：
  * **Horizon-Aware Adaptive Decomposition**：根据预测长度（horizon）动态调整分解策略，使用自适应窗口大小的移动平均方法分离趋势和季节性成分，短期预测关注季节性细节，长期预测强调趋势稳定性，窗口大小通过一个与 horizon 相关的可学习函数计算。
  * **Component-Specific Projections**：对趋势和季节性成分分别应用独立的线性投影矩阵和偏置向量，捕捉各自独特的数学特性（如趋势的长期依赖、季节性的周期行为），避免复杂架构的通用变换，保持参数效率。
  * **Horizon-Dependent Recombination**：通过一个与 horizon 相关的加权机制动态平衡趋势和季节性成分的贡献，使用 sigmoid 函数和可学习参数控制权重分配，同时引入渐进式频率衰减（progressive frequency decay），通过指数衰减函数在长期预测中逐步减少高频季节性成分的影响，解决预测 horizon 困境。
* **关键特点**：不依赖注意力机制等高计算成本组件，参数量仅为 k 级（不到同类模型的 1%），时间复杂度为 O(HT)，适合资源受限环境。

## Experiment

* **有效性**：在七个基准数据集（ETT 系列、Exchange Rate、Traffic、Weather）上，ALinear 在 71.4% 的场景中取得最佳 MSE 和 MAE 结果，尤其在超长期预测（960 步）中，MSE 平均提升 23.7%，如在 ETTm1 数据集上比 Informer 降低 78.3% 误差，验证了自适应设计的优势。
* **参数效率**：提出参数归一化性能（PNP）指标，ALinear 的 PNP-MSE 和 PNP-MAE 分别为 105 和 68.9，远超其他模型，参数量不到同类模型的 1%，展现了极高的效率。
* **实验设置**：数据集选择覆盖不同领域和时间粒度，预测长度从 48 到 960 步，指标（MSE、MAE）全面，实验重复 5 次确保统计可靠性；但仅限于单变量预测，未涉及多变量场景。
* **对比分析**：相比 Transformer 变体，ALinear 在长期预测中更稳定；相比轻量级模型如 DLinear，其自适应机制在长期预测中仍占优。

## Further Thoughts

论文挑战了‘更大模型更好’的假设，启发我们重新思考时间序列预测的本质，是否可以通过更简单、任务驱动的设计替代复杂架构；自适应分解和频率衰减机制提示模型应根据任务特性（如预测长度）定制，而非追求通用架构；此外，参数效率和 PNP 指标的提出，启发我们在模型评估中引入多维度视角（如计算成本、环境影响），这对资源受限场景尤为重要。