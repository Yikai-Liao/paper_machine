---
title: "CAMA: Enhancing Mathematical Reasoning in Large Language Models with Causal Knowledge"
pubDatetime: 2025-08-04T16:39:24+00:00
slug: "2025-08-cama-math-reasoning"
type: "arxiv"
id: "2508.02583"
score: 0.6081283599354602
author: "grok-3-latest"
authors: ["Ruichu Cai", "Lei Zan", "Keli Zhang", "Lujia Pan"]
tags: ["LLM", "Causal Graph", "Mathematical Reasoning", "Structured Guidance", "Prompt Engineering"]
institution: ["Guangdong University of Technology", "Peng Cheng Laboratory", "Huawei Noah's Ark Paris Lab", "Huawei Noah's Ark"]
description: "本文提出 CAMA 框架，通过构建和利用数学因果图（MCG）以结构化提示指导大型语言模型，显著提升了复杂数学推理任务的性能。"
---

> **Summary:** 本文提出 CAMA 框架，通过构建和利用数学因果图（MCG）以结构化提示指导大型语言模型，显著提升了复杂数学推理任务的性能。 

> **Keywords:** LLM, Causal Graph, Mathematical Reasoning, Structured Guidance, Prompt Engineering

**Authors:** Ruichu Cai, Lei Zan, Keli Zhang, Lujia Pan

**Institution(s):** Guangdong University of Technology, Peng Cheng Laboratory, Huawei Noah's Ark Paris Lab, Huawei Noah's Ark


## Problem Background

大型语言模型（LLMs）在复杂数学推理任务中表现不佳，主要由于其架构限制了深层逻辑推理能力，以及对问题表述变化的敏感性导致输出不稳定。
作者提出通过引入显式的、可重用的数学因果结构，解决 LLMs 在多步骤、依赖性强的数学推理中的局限性。

## Method

*   **核心思想:** 提出 CAMA（Causal Mathematician）框架，通过构建数学因果图（MCG）并将其注入 LLM 提示中，增强模型的数学推理能力，而无需参数更新。
*   **学习阶段:** 
    *   从问题-解决方案对中，利用 LLM 提取关键数学知识点，并结合因果发现算法（如 PC 算法）构建 MCG，这是一个有向无环图，节点为知识点，边为因果依赖关系。
    *   通过 LLM 在下游任务上的回答准确率反馈，迭代优化 MCG，确保其与推理需求对齐，具体包括知识点去重、构建二进制矩阵表示知识点与问题的关联性，并推断因果结构。
*   **推理阶段:** 
    *   针对新问题，首先生成初步推理轨迹（Chain-of-Thought）。
    *   根据问题内容和推理轨迹，从 MCG 中动态提取相关子图，子图包含最相关的知识点及其因果依赖。
    *   将子图以自然语言形式编码为提示，注入 LLM，指导其完成最终推理。
*   **关键特点:** CAMA 是一种即插即用的轻量化框架，不依赖模型微调，通过结构化因果指导减少 LLM 对隐式推理的依赖，并提升中间步骤的准确性。

## Experiment

*   **有效性:** CAMA 在 AIME2024 (50.0%), AIME2025 (38.9%), Omni-MATH-200 (45.0%) 上显著优于基线方法（如 CoT-ZeroShot, CoT-FewShot），显示出在数学推理任务中的提升效果。
*   **局限性:** 在 OlympiadBench-674 上，CAMA (66.0%) 略低于 CoT-ZeroShot (67.2%)，因 MCG 知识点覆盖率不足（仅 3% 问题匹配），但调整知识点粒度参数 λ=2 后性能提升至 67.0%，接近基线。
*   **消融实验:** 去掉对齐步骤或有向边的变体性能下降，证明了反馈对齐和因果方向性对推理指导的重要性。
*   **实验设置合理性:** 实验覆盖多种数学基准数据集（AIME, Omni-MATH, OlympiadBench），类别分布均衡（代数、几何、数论、组合学），评价指标 Pass@1 直接反映首次回答准确率，多次重复实验增强结果可靠性。
*   **不足:** MCG 构建依赖训练数据，对新数据集泛化性受限，参数 λ 选择对性能影响较大，缺乏自适应机制。

## Further Thoughts

CAMA 的因果结构（MCG）作为显式知识表示，能否扩展到其他推理领域（如科学或法律推理）？是否可以通过动态更新 MCG 或结合 RLHF 进一步提升适应性？此外，能否设计自适应机制根据任务复杂度自动调整知识点粒度，而非手动设置参数 λ？