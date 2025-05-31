---
title: "Let's Predict Sentence by Sentence"
pubDatetime: 2025-05-28T10:28:35+00:00
slug: "2025-05-sentence-level-reasoning"
type: "arxiv"
id: "2505.22202"
score: 0.7051756801627689
author: "grok-3-latest"
authors: ["Hyeonbin Hwang", "Byeongguk Jeon", "Seungone Kim", "Jiyeon Kim", "Hoyeon Chang", "Sohee Yang", "Seungpil Won", "Dohaeng Lee", "Youbin Ahn", "Minjoon Seo"]
tags: ["LLM", "Sentence Embedding", "Reasoning", "Efficiency", "Latent Space"]
institution: ["KAIST", "Carnegie Mellon University", "University College London", "LG AI Research"]
description: "本文提出一种框架，将预训练语言模型提升到句子级推理，通过自回归预测连续句子嵌入，在保持与 Chain-of-Thought 相当性能的同时，显著降低推理计算成本。"
---

> **Summary:** 本文提出一种框架，将预训练语言模型提升到句子级推理，通过自回归预测连续句子嵌入，在保持与 Chain-of-Thought 相当性能的同时，显著降低推理计算成本。 

> **Keywords:** LLM, Sentence Embedding, Reasoning, Efficiency, Latent Space

**Authors:** Hyeonbin Hwang, Byeongguk Jeon, Seungone Kim, Jiyeon Kim, Hoyeon Chang, Sohee Yang, Seungpil Won, Dohaeng Lee, Youbin Ahn, Minjoon Seo

**Institution(s):** KAIST, Carnegie Mellon University, University College London, LG AI Research


## Problem Background

自回归语言模型（LMs）在复杂推理任务中依赖逐个 token 生成（如 Chain-of-Thought, CoT），导致计算效率低下，尤其在长推理链生成时。
人类推理通常基于更高层次的抽象单位（如句子或概念），而非逐词操作，因此论文探究是否能将预训练语言模型提升到句子级抽象推理空间，直接操作连续嵌入，而无需从头预训练，从而解决效率与推理能力之间的矛盾。

## Method

*   **核心思想**：将预训练的 token 级语言模型改造成在句子级嵌入空间中进行自回归推理，预测连续的下一句嵌入，而非逐 token 生成，以提升计算效率并探索更高层次的结构化推理。
*   **嵌入范式**：设计两种句子嵌入方式：
    *   **语义嵌入（Semantic Embeddings）**：通过自编码目标（reconstruction-based）训练，强调文本内容的保真度，确保嵌入捕捉单个推理步骤的完整语义。
    *   **上下文嵌入（Contextual Embeddings）**：通过下一句预测目标（prediction-based）训练，捕捉推理步骤间的预测性上下文结构，分为无正则化（CTX-B）和带对比损失正则化（CTX-C）两种变体。
*   **推理模式**：提出两种推理策略：
    *   **离散化推理（Discretized Inference）**：将预测的嵌入解码为自然语言文本，再重新编码为下一输入，旨在减少误差累积，但计算成本较高。
    *   **连续推理（Continuous Inference）**：直接在连续嵌入空间中传递预测嵌入，无需中间解码，显著降低计算复杂度。
*   **模型架构**：基于预训练的解码器型 Transformer（如 GPT-2），构建潜在模型（Latent Model），输入为问题和之前的句子嵌入，输出为下一句嵌入预测；通过冻结的解码器将嵌入映射回文本（可选）。
*   **训练目标**：结合交叉熵损失（Cross-Entropy Loss，确保预测嵌入与解码器生成的自然语言目标对齐）和对比损失（InfoNCE Loss，增强预测嵌入与真实嵌入的对齐），避免多解情况下的模糊表示。
*   **终止机制**：使用轻量级分类器判断推理是否结束，减少不必要的步骤。
*   **关键创新**：不修改预训练模型权重，仅通过适配实现抽象层次提升，同时探索嵌入类型和推理模式的性能-效率权衡。

## Experiment

*   **性能表现**：在四个推理领域（数学 GSM8K、常识 CSQA、逻辑 ProsQA、规划 Blocksworld）上，上下文嵌入（尤其是 CTX-C）在连续推理模式下表现最佳，与 token 级 CoT 相比，在逻辑（ProsQA: 92.6% vs 77.5%）和规划（Blocksworld: 80.8% vs 84.3%）任务上接近甚至超越，在数学（GSM8K: 38.3% vs 43.4%）和常识（CSQA: 37.0% vs 35.7%）任务上略逊，差距较小，表明句子级推理在特定领域具有竞争力。
*   **效率提升**：连续推理模式显著降低计算成本，平均推理 FLOPs 减少约一半（例如 CSQA 从 25.89 GFLOPs 降至 9.96 GFLOPs，Blocksworld 从 58.69 降至 28.57），即使离散化模式在长推理任务上也优于 CoT。
*   **实验设置**：实验基于 GPT-2 系列模型（参数小于 1B），数据集规模较大（如 GSM8K 扩展至 370k 训练样本），覆盖多种任务类型，并对比多种基线（CoT, No-CoT, Coconut），设置较为全面；但未测试更大规模模型，结论普适性有待验证。
*   **其他观察**：上下文嵌入优于语义嵌入，表明预测性结构对推理更关键；连续推理在逻辑和规划任务中表现更优，而离散化推理在常识和数学任务中略有优势，反映了两者适用场景的互补性。

## Further Thoughts

论文启发我们思考语言模型是否可以通过适配而非从头训练，逐步攀升到更高层次的抽象推理（如段落级或跨模态推理）；连续推理的高效性提示未来可通过强化学习或轨迹优化提升其稳定性；SentenceLens 工具为模型内部推理轨迹的可视化提供了新思路，是否能开发更多可解释性工具用于高风险应用？此外，模块化设计（独立编码器-解码器与潜在模型结合）是否意味着可以构建通用推理引擎，搭配不同领域嵌入模块？