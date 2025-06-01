---
title: "Let's Reason Formally: Natural-Formal Hybrid Reasoning Enhances LLM's Math Capability"
pubDatetime: 2025-05-29T17:39:30+00:00
slug: "2025-05-hybrid-reasoning-math"
type: "arxiv"
id: "2505.23703"
score: 0.6482121972470604
author: "grok-3-latest"
authors: ["Ruida Wang", "Yuxin Li", "Yi R. Fung", "Tong Zhang"]
tags: ["LLM", "Natural Language", "Formal Language", "Hybrid Reasoning", "Mathematical Reasoning"]
institution: ["University of Illinois Urbana-Champaign", "Hong Kong University of Science and Technology"]
description: "本文提出 **HybridReasoning** 框架，通过问题对齐、混合输入和答案提取，将形式语言推理的严谨性整合到自然语言数学问题求解中，显著提升了大型语言模型的数学推理能力。"
---

> **Summary:** 本文提出 **HybridReasoning** 框架，通过问题对齐、混合输入和答案提取，将形式语言推理的严谨性整合到自然语言数学问题求解中，显著提升了大型语言模型的数学推理能力。 

> **Keywords:** LLM, Natural Language, Formal Language, Hybrid Reasoning, Mathematical Reasoning

**Authors:** Ruida Wang, Yuxin Li, Yi R. Fung, Tong Zhang

**Institution(s):** University of Illinois Urbana-Champaign, Hong Kong University of Science and Technology


## Problem Background

大型语言模型（LLMs）在数学推理方面的能力提升是当前研究热点，但自然语言（NL）推理受限于基础模型能力，强化学习（RL）难以引入新知识，而形式语言（FL）推理虽具严谨性，却因问题结构和格式差异难以直接应用于 NL 数学问题。
论文旨在解决如何将 FL 推理的逻辑严谨性有效整合到 NL 数学问题求解中，以突破 LLMs 的能力瓶颈。

## Method

*   **核心思想:** 提出 **NL-FL HybridReasoning** 框架，通过端到端流程将 FL 推理能力融入 NL 数学问题求解，弥合两者在输入和输出格式上的差异。
*   **具体实现:** 框架分为三个阶段：
    *   **NL-FL Problem Alignment（问题对齐）:** 首先使用通用 LLM 将 NL 中的问答（QA）问题转化为 NL 存在性问题（Existence Problem），避免直接猜测答案带来的歧义；随后通过自动形式化工具（Autoformalizer，如基于 Lean4 的 KiminaAutoformalizer）将 NL 存在性问题转化为 FL 存在性定理，确保 FL 推理器能够处理。
    *   **FL Reasoner Solving with Mixed Problem Input（混合问题输入求解）:** 采用‘混合问题输入’技术，让 FL 推理器（如 KiminaProver-Preview-7B）同时接收 NL QA 问题和 FL 存在性定理作为输入，在长链式思维（Long CoT）过程中首先解决 NL 问题（提供初步答案），然后生成 FL 证明。这种方式充分利用 FL 推理器的逻辑严谨性，同时适应 NL 问题的灵活性。
    *   **Answer Extraction（答案提取）:** 由于 FL 推理器输出偏向形式化代码，难以直接用于 NL 问答，因此使用通用 LLM（如 Qwen3-8B）从 FL 推理器的 Long CoT 输出中提取隐含的 NL 答案，并格式化为可验证形式（如 \boxed{} 格式），从而统一输出格式。
*   **关键点:** 该方法不修改基础模型，仅通过后训练（Post-Training）方法扩展能力，且通过问题对齐和混合输入确保 FL 知识的有效迁移，同时控制推理过程以避免对 NL 性能的负面影响。

## Experiment

*   **有效性:** **HybridReasoning** 框架在 MATH-500 数据集上准确率达到 89.80%，在 AMC 数据集上达到 84.34%，分别比 NL 基线（Qwen3-8B）提升了 4.60% 和 4.82%，尤其在几何（Geometry，提升 14.63%）和预微积分（Precalculus，提升 7.14%）等需要严谨逻辑的领域表现突出。
*   **独特性:** 针对 NL 基线无法解决的问题子集（MATH-500 的 4.60%，AMC 的 8.43%），即使增加尝试次数到 pass@64，NL 基线仍为 0% 准确率，而 **HybridReasoning** 在 pass@16 下全部解决，证明 FL 推理带来了独特能力。
*   **消融研究:** 去掉存在性对齐步骤后准确率下降 4.42%，在 AMC 上甚至低于 NL 基线；去掉专业 FL 推理器后性能下降 2.55%，验证了框架各组件的必要性。
*   **实验设置合理性:** 实验覆盖 MATH-500（500 题）和 AMC（83 题）两个数据集，包含从预代数到预微积分的多个领域，使用 pass@16 指标评估能力极限，数据分析细致（如按学科拆分结果），增强了结果可信度。
*   **开销:** 主要增加了自动形式化和 FL 推理的计算成本，评估过程耗费约 200 A6000 GPU 小时，但未显著影响推理效率。

## Further Thoughts

1. **跨领域知识整合的通用性:** **HybridReasoning** 框架提供了一个将专家领域知识（如 FL 推理）融入另一领域（如 NL 推理）的通用管道，这种思想可扩展到其他领域，例如将医学规则系统嵌入 NL 对话，或将法律逻辑推理整合到文本分析中，启发跨领域任务的解决方案设计。
2. **问题对齐与格式转换策略:** 通过将 NL 问题转化为 FL 兼容格式，论文展示了如何通过中间步骤弥合不同推理范式的差距，这种策略可启发其他跨模态任务，如将图像描述问题转化为文本推理问题，或将语音输入转化为结构化数据分析。
3. **混合输入与多阶段推理潜力:** 混合问题输入技术允许模型在同一推理过程中处理多种格式输入，这种方法对多任务学习或多模态推理有借鉴意义，例如在视觉-语言任务中同时处理图像和文本输入，或在复杂决策系统中整合多种数据源。