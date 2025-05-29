---
title: "Origin Tracer: A Method for Detecting LoRA Fine-Tuning Origins in LLMs"
pubDatetime: 2025-05-26T03:38:14+00:00
slug: "2025-05-lora-origin-detection"
type: "arxiv"
id: "2505.19466"
score: 0.6309719291623912
author: "grok-3-latest"
authors: ["Hongyu Liang", "Yuting Zheng", "Yihan Li", "Yiran Zhang"]
tags: ["LLM", "Fine-Tuning", "Model Verification", "Low-Rank Adaptation", "Transformer"]
institution: ["Shanghai Jiao Tong University", "National University of Defense Technology"]
description: "本文提出 Origin-Tracer 方法，通过中间状态重建和奇异值分析，鲁棒地检测大型语言模型是否基于特定基础模型通过 LoRA 微调，并提取微调秩，为开源社区透明度提供新工具。"
---

> **Summary:** 本文提出 Origin-Tracer 方法，通过中间状态重建和奇异值分析，鲁棒地检测大型语言模型是否基于特定基础模型通过 LoRA 微调，并提取微调秩，为开源社区透明度提供新工具。 

> **Keywords:** LLM, Fine-Tuning, Model Verification, Low-Rank Adaptation, Transformer

**Authors:** Hongyu Liang, Yuting Zheng, Yihan Li, Yiran Zhang

**Institution(s):** Shanghai Jiao Tong University, National University of Defense Technology


## Problem Background

随着大型语言模型（LLMs）的广泛应用，模型微调（fine-tuning）成为提升特定任务性能的重要手段，但部分模型提供商通过虚假声明夸大技术能力（如 Reflection-70B 的来源误导），引发了开源社区对透明度和信任的担忧。
现有模型溯源检测方法（如功能、表示和权重相似性）在面对混淆技术（如参数排列和缩放变换）时往往失效，因此亟需一种鲁棒方法来准确判断模型是否从特定基础模型微调而来，并揭示其微调细节。

## Method

*   **核心思想：** 提出 Origin-Tracer 方法，通过分析 Transformer 模型的中间状态（intermediate states）来推断是否基于特定基础模型通过 LoRA（Low-Rank Adaptation）微调，并提取微调过程中的 LoRA 秩，即使参数被混淆也能有效工作。
*   **具体步骤：**
    *   **LoRA 秩信息提取：** 聚焦自注意力模块的值（Value, W_V）和输出（Output, W_O）矩阵，利用输入-输出对的数学唯一性（通过定理证明），分析中间状态以推断参数矩阵的低秩差异。
    *   **等价中间状态重建：** 针对混淆后的候选模型，利用基础模型的多层感知机（MLP）模块和候选模型的输出，通过梯度下降迭代优化方法重建中间状态，解决非线性变换带来的逆向推导难题。
    *   **整体流程与算法：** 使用一维张量输入，结合随机采样和多次迭代策略，计算中间状态的奇异值分解（SVD），选择秩最小的结果作为最终输出，以提高估计准确性。
*   **创新点：** 不依赖直接参数比较，而是通过中间状态的数学性质间接推断微调来源，对混淆技术具有鲁棒性。

## Experiment

*   **有效性：** 在 31 个开源模型（包括 LLaMA2、LLaMA3、LLaMA3.1 和 Mistral 系列，规模从 7B 到 70B）上测试，Origin-Tracer 成功提取了 LoRA 秩，估计值与真实值高度一致（误差通常在 ±1 以内），如 LLaMA3.1-8B 模型的目标秩 8、16、32，估计值分别为 8、19、35。
*   **层级差异：** 中间层的重建效果显著优于首尾层（图 5），可能因中间层输出范数更高，信息表达能力更强（图 3），这为选择代表性层提供了依据。
*   **实验设置合理性：** 实验全面覆盖多种模型规模和架构，基于 NLTK 构建数据集，模拟真实世界混淆场景（灰盒模型，仅访问输入输出），并通过选择 top 10% 奇异值比的层确保结果代表性。
*   **局限性：** 方法目前仅适用于自注意力模块的低秩修改，不支持 MLP 模块变化，且对 V 和 O 矩阵修改有特定要求。

## Further Thoughts

1. 通过中间状态重建绕过混淆技术的思路令人启发，是否可扩展到其他模型架构（如 CNN）或检测其他微调方法（如全参数微调）？
2. 中间层输出范数更高的现象提示模型层级差异显著，是否可利用此特性优化模型设计，如在中间层引入更多任务特定调整？
3. 随机采样和多次迭代选择最小秩的策略在不确定性场景中表现良好，是否可应用于模型剪枝或量化等需要鲁棒估计的领域？