---
title: "Large Language Models as Computable Approximations to Solomonoff Induction"
pubDatetime: 2025-05-21T17:35:08+00:00
slug: "2025-05-llm-solomonoff-approximation"
type: "arxiv"
id: "2505.15784"
score: 0.6314912886650653
author: "grok-3-latest"
authors: ["Jun Wan", "Lingrui Mei"]
tags: ["LLM", "Solomonoff Induction", "Kolmogorov Complexity", "Few-Shot Learning", "In-Context Learning"]
institution: ["UBS AG", "State Key Lab of AI Safety"]
description: "本文通过证明大型语言模型的训练和推理过程是对 Solomonoff 先验和归纳的计算近似，统一解释了多种涌现现象，并提出了一种基于低置信度样本选择的少样本学习策略，显著提升模型性能。"
---

> **Summary:** 本文通过证明大型语言模型的训练和推理过程是对 Solomonoff 先验和归纳的计算近似，统一解释了多种涌现现象，并提出了一种基于低置信度样本选择的少样本学习策略，显著提升模型性能。 

> **Keywords:** LLM, Solomonoff Induction, Kolmogorov Complexity, Few-Shot Learning, In-Context Learning

**Authors:** Jun Wan, Lingrui Mei

**Institution(s):** UBS AG, State Key Lab of AI Safety


## Problem Background

大型语言模型（LLMs）在多个领域取得了显著成功，但缺乏统一的数学理论框架来系统解释其涌现现象（如上下文学习、少样本学习和规模法则）。
本文试图通过算法信息论（AIT）中的 Solomonoff 归纳理论，为 LLMs 提供一个统一的理论视角，解决现有理论碎片化、无法全面解释模型行为的难题。

## Method

*   **核心理论 1 - 训练过程与 Solomonoff 先验的联系**：
    *   作者通过数学构造证明，LLMs 的训练过程（通过最小化预测损失优化参数）是对 Solomonoff 先验的计算近似。
    *   具体而言，训练过程被重新解释为寻找能够生成训练数据的最短程序，这一过程与 Solomonoff 先验基于 Kolmogorov 复杂性（最短描述长度）的理念一致。
    *   实现上，作者构造了一个程序表示（包含模型参数、编码、解码迭代次数和随机种子），并通过 Elias gamma 编码确保前缀编码性质，最终推导出训练损失最小化等价于逼近 Solomonoff 先验。
*   **核心理论 2 - 推理过程与 Solomonoff 归纳的联系**：
    *   作者进一步证明，LLMs 的下一个 token 预测机制是对 Solomonoff 归纳的计算近似。
    *   通过分析程序编码长度和预测概率分布，论文展示了当序列较长时，LLM 的条件概率分布与 Solomonoff 归纳的预测概率高度相关（经过归一化处理）。
    *   这为 LLMs 的泛化能力和预测能力提供了理论依据。
*   **应用方法 - 少样本示例选择策略**：
    *   基于 Solomonoff 归纳的收敛性质，作者提出了一种少样本学习中的示例选择方法：优先选择模型预测置信度较低的样本（即暴露模型预测弱点的样本）。
    *   具体实现上，通过迭代遍历数据集，计算模型对每个样本正确标签的 softmax 概率，选择置信度最低的样本加入提示（prompt），以加速模型适应目标分布。
*   **关键特点**：理论证明严谨，结合了数学推导和计算近似；应用方法直接源于理论洞察，具有可操作性。

## Experiment

*   **有效性**：实验在三个文本分类数据集（SMS 垃圾邮件分类、情感识别、新闻分类）上验证了少样本示例选择策略，选择低置信度样本（low-confidence）在所有测试模型（Qwen2.5 3B/7B, Llama 3.1 8B/3.2 3B）和数据集上均显著优于高置信度样本（high-confidence），例如 Qwen2.5 3B 在 SMS 数据集上准确率从 76.62% 提升至 90.07%。
*   **规模效应**：随着模型规模增大，低置信度策略的性能提升幅度有所减小（例如 Qwen2.5 7B 在 SMS 上提升从 92.73% 到 94.60%），可能是因为大模型已有较强基线，但改进仍然存在。
*   **实验设置合理性**：实验覆盖了不同规模模型和任务类型，数据集具有代表性；温度参数设为 0 确保结果确定性和理论一致性；然而，实验局限于文本分类，未涉及其他任务类型（如生成任务），可能存在一定局限性。
*   **计算开销**：主要开销在于置信度计算和样本选择过程，但未显著增加推理负担，实验总计算时间约 1.5 天（基于 4 个 NVIDIA A100 GPU）。

## Further Thoughts

将 LLMs 视为 Solomonoff 归纳的计算近似，这一视角不仅统一解释了上下文学习和规模法则等现象，还启发我们思考是否可以通过更精确模拟 Solomonoff 先验（例如量化 Kolmogorov 复杂性）来优化模型训练；此外，低置信度样本选择策略或许可扩展至主动学习或强化学习场景，通过动态调整置信度阈值适应不同任务需求。