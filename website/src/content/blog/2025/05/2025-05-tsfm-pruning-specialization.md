---
title: "Less is More: Unlocking Specialization of Time Series Foundation Models via Structured Pruning"
pubDatetime: 2025-05-29T07:33:49+00:00
slug: "2025-05-tsfm-pruning-specialization"
type: "arxiv"
id: "2505.23195"
score: 0.6110102572712269
author: "grok-3-latest"
authors: ["Lifan Zhao", "Yanyan Shen", "Zhaoyang Liu", "Xue Wang", "Jiaji Deng"]
tags: ["Time Series", "Foundation Model", "Structured Pruning", "Fine-Tuning", "Sparsity"]
institution: ["Shanghai Jiao Tong University", "Alibaba Group"]
description: "本文提出‘先剪枝后微调’范式，通过结构化剪枝利用时间序列基础模型的稀疏性，显著提升下游预测任务性能并超越强基准。"
---

> **Summary:** 本文提出‘先剪枝后微调’范式，通过结构化剪枝利用时间序列基础模型的稀疏性，显著提升下游预测任务性能并超越强基准。 

> **Keywords:** Time Series, Foundation Model, Structured Pruning, Fine-Tuning, Sparsity

**Authors:** Lifan Zhao, Yanyan Shen, Zhaoyang Liu, Xue Wang, Jiaji Deng

**Institution(s):** Shanghai Jiao Tong University, Alibaba Group


## Problem Background

时间序列基础模型（TSFMs）在大规模数据集上预训练后，展现了强大的零样本预测能力，但即使经过微调，其在全样本下游任务上的表现仍经常不如专门训练的小型模型（如 PatchTST）。
这一性能差距源于预训练数据与下游任务数据之间的统计差异，核心问题是探索如何有效适配 TSFMs 以在特定时间序列预测任务中取得优异表现。

## Method

*   **核心思想:** 提出‘先剪枝后微调’（prune-then-finetune）范式，通过结构化剪枝（Structured Pruning）识别并保留任务相关的子网络，限制微调过程在更紧凑的参数空间内进行，以提升预测性能。
*   **剪枝单位:** 将线性层的输入和输出通道定义为剪枝单位，覆盖自注意力模块和前馈网络（FFN），通过二进制掩码控制通道的保留或移除。
*   **重要性评分:** 基于预测损失变化评估通道重要性，利用二阶泰勒展开和 Fisher 信息近似计算损失梯度，高效识别冗余通道。
*   **渐进式剪枝:** 采用批次级别的渐进式剪枝策略，通过指数移动平均（EMA）平滑重要性评分，避免一次性过度剪枝，每次基于全局排名移除最低重要性的通道。
*   **微调阶段:** 剪枝后对剩余参数在下游数据上进行微调，专注于任务相关的参数子空间，避免过拟合和对噪声特征的关注。
*   **创新点:** 剪枝不仅是模型压缩手段，更是正则化工具，利用预训练模型的稀疏性（Sparsity）作为先验知识，引导微调过程。

## Experiment

*   **性能提升:** 在七个 TSFMs 和六个基准数据集上的实验表明，剪枝后微调在 83% 的预测任务中优于直接微调，平均误差降低 1.0%-4.4%，最大误差降低达 22.8%（Time-MoE base 在 ETTm2 上）。
*   **胜率对比:** 与强基准 PatchTST 相比，剪枝后微调显著提高 TSFMs 的胜率，从 90% 提升至 100%（在某些数据集上）。
*   **推理效率:** 剪枝带来额外好处，推理速度提升最高达 7.4 倍（Moirai large）。
*   **迁移能力:** 剪枝后的模型在同一领域的其他数据集上表现出良好的零样本迁移能力，表明剪枝识别的子网络具有领域通用性。
*   **实验设置合理性:** 实验覆盖多种模型架构（编码器、解码器、混合专家等）和预测长度（96 至 720 步），数据集涵盖不同频率和领域（如 ETT, Weather, Electricity），设置全面且结果可信。

## Further Thoughts

论文揭示了 TSFMs 预训练过程中形成的稀疏性可以作为任务适配的先验知识，这启发我思考是否可以在其他领域的基础模型（如语言或图像模型）中挖掘类似稀疏模式用于特化；此外，剪枝作为正则化手段的思路是否能与 dropout 等技术结合，进一步提升适配效率；最后，剪枝子网络的迁移性提示在数据稀缺场景下，可以利用领域相关数据集指导剪枝，是否能扩展到跨领域适配，通过共享子网络实现更广义的迁移？