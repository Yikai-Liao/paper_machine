---
title: "Prolonged Reasoning Is Not All You Need: Certainty-Based Adaptive Routing for Efficient LLM/MLLM Reasoning"
pubDatetime: 2025-05-21T06:20:17+00:00
slug: "2025-05-certainty-adaptive-reasoning"
type: "arxiv"
id: "2505.15154"
score: 0.8309432850854456
author: "grok-3-latest"
authors: ["Jinghui Lu", "Haiyang Yu", "Siliang Xu", "Shiwei Ran", "GuoZhi Tang", "Siqi Wang", "Bin Shan", "Teng Fu", "Hao Feng", "Jingqun Tang", "Han Wang", "Can Huang"]
tags: ["LLM", "MLLM", "Reasoning", "Adaptive Routing", "Efficiency"]
institution: ["ByteDance", "Fudan University, China"]
description: "本文提出基于确定性的自适应推理框架（CAR），通过困惑度动态切换短答案和长篇推理，在保持高准确率的同时显著降低计算成本，为 LLM 和 MLLM 高效推理提供了新思路。"
---

> **Summary:** 本文提出基于确定性的自适应推理框架（CAR），通过困惑度动态切换短答案和长篇推理，在保持高准确率的同时显著降低计算成本，为 LLM 和 MLLM 高效推理提供了新思路。 

> **Keywords:** LLM, MLLM, Reasoning, Adaptive Routing, Efficiency

**Authors:** Jinghui Lu, Haiyang Yu, Siliang Xu, Shiwei Ran, GuoZhi Tang, Siqi Wang, Bin Shan, Teng Fu, Hao Feng, Jingqun Tang, Han Wang, Can Huang

**Institution(s):** ByteDance, Fudan University, China


## Problem Background

大型语言模型（LLMs）和多模态大型语言模型（MLLMs）在复杂推理任务中通过链式思维（Chain-of-Thought, CoT）显著提升了性能，但过度依赖长篇推理会导致效率低下，尤其在简单任务上不仅未能提升准确率，反而可能引入噪声并增加计算成本（如 token 消耗）；论文旨在解决如何自适应地决定是否需要长篇推理，以在准确性和效率之间取得平衡。

## Method

* **核心思想：** 提出基于确定性的自适应推理框架（Certainty-Based Adaptive Routing, CAR），根据模型对短答案的置信度动态决定是否触发长篇推理，以优化准确性和效率的权衡。
* **具体实现：**
  - **短答案生成与置信度评估：** 模型首先生成短答案，并计算其困惑度（Perplexity, PPL），PPL 作为置信度指标，高 PPL 表示低置信度。
  - **高斯分布建模：** 利用训练数据中正确和错误短答案的 PPL 分布，分别拟合两个高斯分布，估计当前短答案的正确概率。
  - **决策路由：** 根据贝叶斯定理计算短答案正确的后验概率，若正确概率低于错误概率（即置信度不足），则触发长篇推理以提升准确性；否则直接输出短答案以节省计算资源。
* **训练与推理：** 模型通过混合短答案和长篇推理的训练数据进行指令微调，确保两种模式下的生成能力；推理时动态路由不引入额外模型，仅依赖 PPL 计算和预估分布。
* **关键优势：** 不需要修改模型架构，仅通过置信度评估实现自适应策略，避免了传统‘一刀切’推理方式的资源浪费。

## Experiment

* **有效性：** 在多模态任务（如 DocVQA, ChartQA, FUNSD）上，CAR 平均准确率（77.9%）显著优于短答案（75.1%）和长篇推理（72.4%），token 消耗仅为长篇推理的 15%（86.9 vs. 576.3 tokens）；在文本推理任务（如 GSM8K, MathQA）上，CAR 平均准确率（Qwen2.5 为 81.1%，Llama3.1 为 74.9%）优于短答案和长篇推理，同时 token 消耗减少约 45%。
* **优越性：** 相比最先进的 token 减少方法（如 TALE 和 COD），CAR 在准确率上提升明显（例如 Qwen2.5 上比 TALE 高 8.3%），且 token 消耗最低，展现出更好的性能-效率权衡。
* **任务适应性：** CAR 在简单任务（如 VQA/KIE）上提升显著，而在复杂推理任务（如 GSM8K）上提升较小，符合假设：复杂任务更依赖长篇推理，CAR 能正确识别并避免不必要的短答案。
* **实验设置：** 实验覆盖多模态和文本任务，测试了多种模型（Qwen2-VL, Qwen2.5, Llama3.1），数据集选择合理，评估指标包括准确率和 token 消耗，全面验证了方法的适用性。
* **开销：** CAR 引入的额外计算开销较小，仅限于 PPL 计算和概率估计，相比纯短答案推理略有增加，但远低于长篇推理的成本。

## Further Thoughts

CAR 基于困惑度（PPL）的自适应路由机制启发了我：置信度评估可以作为模型推理策略的动态调节器，不仅限于短答案与长篇推理的切换，或许还能用于多轮对话中是否调用外部工具或 API；此外，CAR 与其他 token 减少方法的结合（如 TALE）显示出互补性，未来是否可以设计通用框架整合多种推理优化策略？同时，PPL 作为置信度指标的局限性提示我们，是否可以引入熵或多模型一致性等指标以增强路由决策的鲁棒性？