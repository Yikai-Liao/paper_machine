---
title: "Univariate to Multivariate: LLMs as Zero-Shot Predictors for Time-Series Forecasting"
pubDatetime: 2025-06-03T03:02:47+00:00
slug: "2025-06-llmpred-time-series"
type: "arxiv"
id: "2506.02389"
score: 0.5346714186685763
author: "grok-3-latest"
authors: ["Chamara Madarasingha", "Nasrin Sohrabi", "Zahir Tari"]
tags: ["LLM", "Time Series", "Zero-Shot Learning", "Decomposition", "Multivariate Analysis"]
institution: ["RMIT University", "Deakin University"]
description: "本文提出 LLMPred 框架，通过时间序列分解和提示工程，利用 LLMs 的零样本能力显著提升单变量和多变量时间序列预测性能，尤其在资源受限环境下表现优异。"
---

> **Summary:** 本文提出 LLMPred 框架，通过时间序列分解和提示工程，利用 LLMs 的零样本能力显著提升单变量和多变量时间序列预测性能，尤其在资源受限环境下表现优异。 

> **Keywords:** LLM, Time Series, Zero-Shot Learning, Decomposition, Multivariate Analysis

**Authors:** Chamara Madarasingha, Nasrin Sohrabi, Zahir Tari

**Institution(s):** RMIT University, Deakin University


## Problem Background

时间序列预测在金融、天气、交通等领域至关重要，但传统深度学习方法依赖大量数据和专家知识，在数据不足时性能下降且泛化能力有限。
大型语言模型（LLMs）因其强大的泛化能力和零样本学习能力被认为是解决这一问题的潜力工具，但其在处理复杂、噪声大的时间序列数据（尤其是多变量数据）时的表现仍未被充分探索。
论文的出发点是利用 LLMs 的零样本能力，通过创新的数据处理和提示策略，提升其在单变量和多变量时间序列预测中的表现，解决传统方法对数据和训练资源的依赖问题。

## Method

*   **核心思想:** 提出 LLMPred 框架，利用 LLMs 的零样本能力，通过时间序列分解和提示工程处理复杂时间序列数据，提升预测性能。
*   **时间序列分解:** 将单变量时间序列分解为低频（趋势）和高频（噪声）成分，分别处理以降低复杂性。使用基于余弦相似度和均方误差（MSE）的频率分解算法，在 2.5-15.0 Hz 范围内选择最佳截止频率，分离低频趋势和高频波动。
*   **文本转换与零样本预测:** 将数值序列转换为文本格式，通过规范化（如最大值归一化）和线性变换调整数值范围，结合指导性提示输入 LLMs 进行零样本预测，避免微调带来的计算成本。
*   **后处理优化:** 对低频成分使用多层感知机（MLP）进一步优化预测模式，提升模式保真度；对高频成分应用高斯变换，调整预测分布以匹配历史数据分布，增强一致性。
*   **多变量扩展:** 通过轻量级提示处理策略，将多变量数据组织为文本行（每行代表一个时间点多个特征），利用 LLMs 捕捉跨变量依赖关系，实现从单变量到多变量预测的扩展。
*   **关键点:** 方法不依赖模型微调，适用于资源受限环境，通过分解和提示设计使 LLMs 更好地理解时间序列结构。

## Experiment

*   **有效性:** 在单变量预测中，LLMPred（尤其是 GPT-4o-mini 版本）在 MSE 上实现了 26.8% 的显著降低，优于大多数基准模型（如 LLMTime、Autoformer）。在多变量预测中，LLMPred 的 MSE 与最佳基准模型（Autoformer 和 FEDformer）差距较小（仅 0.007-0.010），且多变量预测比单变量预测平均提升了 17.4%。
*   **稳定性:** LLMPred 在多变量预测中表现出更高的稳定性，MSE 和 MAE 的标准差低于基准模型，表明其对不同数据集的鲁棒性更强。
*   **消融研究:** 验证了各组件的重要性，例如频率分解平均降低 MSE 14.4%，后处理（如 MLP 和高斯变换）进一步提升了预测稳定性（MSE 标准差降低 82.0%）。
*   **实验设置:** 使用较小的 LLMs（如 Llama 2 7B、GPT-4o-mini）在多个数据集（ETTh1、ETTm1 等）上测试，预测长度为 48 和 96 步，覆盖单变量和多变量场景，设置合理。但上下文长度限制了多变量预测中特征数量（最多 9 个）和预测长度的扩展，部分模型（如 Llama 7B）在长预测时性能下降。
*   **计算开销:** 多变量预测对计算资源需求较高，但整体方法因零样本特性避免了训练成本，适用于低资源环境。

## Further Thoughts

时间序列分解为低频和高频成分的策略启发了我，可以将类似思路推广到其他领域（如图像或音频处理），利用 LLMs 处理多模态数据的结构化模式；此外，通过提示工程将数值数据转化为文本的做法，提示我们可以在其他非文本任务（如图结构预测）中探索 LLMs 的零样本潜力；最后，论文提到的动态频率选择和分布适应的未来方向，启发我们可以设计自适应 LLMs 框架，根据数据特性动态调整处理策略。