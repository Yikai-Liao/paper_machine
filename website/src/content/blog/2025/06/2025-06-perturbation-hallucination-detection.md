---
title: "Shaking to Reveal: Perturbation-Based Detection of LLM Hallucinations"
pubDatetime: 2025-06-03T09:44:28+00:00
slug: "2025-06-perturbation-hallucination-detection"
type: "arxiv"
id: "2506.02696"
score: 0.5616339888513616
author: "grok-3-latest"
authors: ["Jinyuan Luo", "Zhen Fang", "Yixuan Li", "Seongheon Park", "Ling Chen"]
tags: ["LLM", "Hallucination Detection", "Perturbation", "Intermediate Representation", "Contrastive Learning"]
institution: ["Australian Artificial Intelligence Institute, University of Technology Sydney", "Department of Computer Sciences, University of Wisconsin-Madison"]
description: "本文提出样本特定提示（SSP）框架，通过动态生成噪声提示诱导中间层表示变化，显著提升大型语言模型幻觉检测的精度和泛化能力。"
---

> **Summary:** 本文提出样本特定提示（SSP）框架，通过动态生成噪声提示诱导中间层表示变化，显著提升大型语言模型幻觉检测的精度和泛化能力。 

> **Keywords:** LLM, Hallucination Detection, Perturbation, Intermediate Representation, Contrastive Learning

**Authors:** Jinyuan Luo, Zhen Fang, Yixuan Li, Seongheon Park, Ling Chen

**Institution(s):** Australian Artificial Intelligence Institute, University of Technology Sydney, Department of Computer Sciences, University of Wisconsin-Madison


## Problem Background

大型语言模型（LLMs）在问答任务中常生成缺乏事实依据的‘幻觉’内容，这在高精度领域（如医疗、法律）限制了其可靠性。
传统的自评估方法依赖输出层置信度来检测幻觉，但由于模型预测分布与真实数据分布的偏差，输出置信度往往不可靠。
论文提出，中间层表示可能更接近模型内部推理过程，较少受输出层偏差影响，因此探索利用中间层信息进行更可靠的幻觉检测。

## Method

*   **核心思想:** 通过引入样本特定的扰动（Perturbation），揭示中间层表示对输入变化的敏感性差异，作为区分真实和幻觉响应的信号。
*   **具体实现:**
    *   **样本特定提示生成（Sample-Specific Prompting, SSP）:** 为每个问答对动态生成噪声提示（Noise Prompt），初始通过种子提示（SeedPrompt）引导模型生成语义中立的风格化句子，随后使用轻量级提示生成器（两层 MLP）优化噪声提示的嵌入，确保扰动与输入内容相关。
    *   **中间层表示提取与编码:** 提取原始输入和扰动输入在中间层的表示，通过一个可学习的轻量级编码器（三层 MLP）将这些表示映射到一个区分空间，放大真实和幻觉响应的表示差异。
    *   **差异量化:** 使用余弦相似度（Cosine Similarity）计算扰动前后表示的变化，预期真实响应的变化更大，幻觉响应的变化较小。
    *   **训练与优化:** 设计对比损失（Contrastive Loss），鼓励真实响应的表示变化更大（通过设置上界阈值），幻觉响应的变化更小（通过设置下界阈值），优化提示生成器和编码器的参数。
*   **关键优势:** 不依赖输出层置信度，避免输出偏差；不修改基础模型，仅在推理时引入扰动，计算开销低；动态生成噪声提示，适应不同输入特性。

## Experiment

*   **有效性:** SSP 在四个问答数据集（TruthfulQA, TriviaQA, CoQA, TydiQA-GP）上，使用 LLaMA-3-8B-Instruct 和 Qwen-2.5-7B-Instruct 模型，显著优于基线方法，例如在 TruthfulQA 上 AUROC 提升 4.78%，平均 AUROC 达 75.38%（LLaMA-3）和 72.72%（Qwen-2.5）。
*   **泛化性:** 跨数据集迁移实验显示 SSP 具有较强泛化能力，例如从 TriviaQA 迁移到 TydiQA-GP 时 AUROC 仍达 73.89%，优于其他方法如 EGH 和 Linear Probe。
*   **效率:** 相比需要多次采样的基线（如 Semantic Entropy），SSP 仅需计算扰动前后的表示变化，推理时间更短，计算成本更低。
*   **合理性:** 消融研究表明编码器和种子提示对性能至关重要；不同层表示提取实验验证了中间层选择的合理性；数据集覆盖开放式和闭合式问答任务，基线方法多样，实验设置全面。

## Further Thoughts

扰动作为揭示模型内部行为的工具，不仅限于幻觉检测，或许可用于模型解释性分析或鲁棒性测试；样本特定的动态提示生成策略可能适用于个性化生成任务，通过调整扰动控制输出风格；中间层表示的敏感性差异可能反映事实知识编码方式，未来可探索定位幻觉具体位置或原因。