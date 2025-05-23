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
description: "本文提出基于确定性的自适应推理框架 CAR，通过动态路由短答案和长篇推理，显著提升了 LLMs 和 MLLMs 在准确性和效率上的表现。"
---

> **Summary:** 本文提出基于确定性的自适应推理框架 CAR，通过动态路由短答案和长篇推理，显著提升了 LLMs 和 MLLMs 在准确性和效率上的表现。 

> **Keywords:** LLM, MLLM, Reasoning, Adaptive Routing, Efficiency

**Authors:** Jinghui Lu, Haiyang Yu, Siliang Xu, Shiwei Ran, GuoZhi Tang, Siqi Wang, Bin Shan, Teng Fu, Hao Feng, Jingqun Tang, Han Wang, Can Huang

**Institution(s):** ByteDance, Fudan University, China


## Problem Background

大型语言模型（LLMs）和多模态大型语言模型（MLLMs）在复杂推理任务中通过链式思维（Chain-of-Thought, CoT）显著提升了性能，但过度依赖长篇推理会导致效率低下，尤其在简单任务上不仅未能提升准确率，反而可能引入噪声并增加计算成本（如 token 消耗）；论文旨在解决如何自适应地决定是否需要长篇推理，以平衡准确性和效率。

## Method

* **核心思想**：提出基于确定性的自适应推理框架（Certainty-Based Adaptive Routing, CAR），根据模型对短答案的置信度动态决定是否触发长篇推理，以优化准确性和效率的权衡。
* **具体实现**：
  - **训练阶段**：使用包含短答案和长篇推理答案的混合数据集进行指令微调，使模型同时掌握两种输出模式；采用交叉熵损失优化模型对输入序列的预测概率。
  - **置信度评估**：通过困惑度（Perplexity, PPL）衡量短答案的置信度，PPL 定义为模型对生成序列概率的对数平均值的指数形式，反映模型对答案的不确定性。
  - **高斯分布建模**：基于训练数据中短答案的 PPL 分布，分别对正确和错误答案的 PPL 拟合两个高斯分布，估计其均值和方差，作为置信度判断的依据。
  - **推理阶段**：模型首先生成短答案并计算其 PPL；利用贝叶斯定理计算该答案正确的概率（基于高斯分布的似然和先验概率）；若正确概率低于错误概率（即置信度低），则触发长篇推理以提升准确性，否则直接输出短答案。
* **关键创新**：CAR 不依赖固定的推理策略，而是通过实时置信度评估实现动态路由，避免不必要的长篇推理；此外，PPL 作为置信度指标的有效性通过试点研究得到了验证。

## Experiment

* **有效性**：在多模态任务（如 DocVQA, ChartQA, FUNSD）中，CAR 平均准确率（77.9%）显著高于短答案基线（75.1%）和长篇推理基线（72.4%）；在文本推理任务（如 GSM8K, MathQA）中，CAR 准确率（Qwen2.5 上为 81.1%）也优于短答案（55.8%）和长篇推理（75.0%）。
* **效率提升**：CAR 在 token 消耗上远低于长篇推理，例如多模态任务中平均 token 数为 86.9，仅为长篇推理（576.3）的 15%；文本任务中 token 减少约 45%。
* **对比优势**：与最新 token 减少方法（如 TALE 和 COD）相比，CAR 在准确性和效率上均占优，例如在 Qwen2.5 上，CAR 准确率比 TALE 高 8.3%，比 COD 高 6.9%，且 token 消耗最低（69.2）。
* **实验设置**：实验覆盖多模态和纯文本任务，数据集从简单提取到复杂推理均有涉及，模型包括 Qwen2-VL-7B, Qwen2.5-7B 和 Llama3.1-8B，设置全面合理；但在复杂推理任务（如 GSM8K）上提升幅度较小，符合任务特性。
* **结论**：CAR 在简单任务上效果显著，在复杂任务上保持竞争力，实验数据充分支持方法的有效性。

## Further Thoughts

CAR 的自适应路由思想启发我们探索更多置信度指标（如基于熵或隐藏状态的度量）以提升决策精度；此外，CAR 与其他推理优化方法（如 TALE）的结合显示出潜力，未来可尝试将其与分层推理或工具调用集成；还可以考虑根据任务类型设计特异性路由规则，或引入在线学习机制动态调整置信度阈值以适应不同场景。