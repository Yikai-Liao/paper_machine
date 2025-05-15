---
title: "DeepMath-Creative: A Benchmark for Evaluating Mathematical Creativity of Large Language Models"
pubDatetime: 2025-05-13T16:58:05+00:00
slug: "2025-05-deepmath-creative-benchmark"
type: "arxiv"
id: "2505.08744"
score: 0.6917769964554747
author: "grok-3-latest"
authors: ["Xiaoyang Chen", "Yuting Gao", "Xiang Jiang", "Xiangnan Li"]
tags: ["LLM", "Mathematical Creativity", "Benchmark", "Evaluation", "Constructive Reasoning"]
institution: ["Tongji University", "Fudan University", "Tsinghua University", "Tianjin University", "University of Hong Kong", "University of Warwick", "MBZUAI", "LibrAI", "The Chinese University of Hong Kong", "The University of Texas at Austin", "Kyushu University", "University of Science and Technology of China", "Jilin University"]
description: "本文提出 DeepMath-Creative 基准，专注于评估大型语言模型的数学创造力，通过系统实验揭示了当前模型在构造性任务上的局限性，为未来研究提供了重要方向。"
---

> **Summary:** 本文提出 DeepMath-Creative 基准，专注于评估大型语言模型的数学创造力，通过系统实验揭示了当前模型在构造性任务上的局限性，为未来研究提供了重要方向。 

> **Keywords:** LLM, Mathematical Creativity, Benchmark, Evaluation, Constructive Reasoning

**Authors:** Xiaoyang Chen, Yuting Gao, Xiang Jiang, Xiangnan Li

**Institution(s):** Tongji University, Fudan University, Tsinghua University, Tianjin University, University of Hong Kong, University of Warwick, MBZUAI, LibrAI, The Chinese University of Hong Kong, The University of Texas at Austin, Kyushu University, University of Science and Technology of China, Jilin University


## Problem Background

大型语言模型（LLMs）在数学推理任务上表现出较强的能力，尤其是在基础到本科水平的数学问题上，但现有数据集和基准主要关注推理技能，对模型的数学创造力（Mathematical Creativity）评估不足，相关数据集稀缺。
论文提出，数学创造力是数学能力的重要维度，涉及新概念的生成、新方法的发明以及新例子的构建（如反例），而当前模型在这方面的表现尚未被充分探索。
因此，核心问题是设计一个系统性的评估框架和基准，揭示 LLMs 在创造性数学任务上的真实能力。

## Method

*   **基准设计原则**：论文提出 DeepMath-Creative 基准，强调创新性和构造性，覆盖代数、拓扑、分析、几何等核心数学分支，旨在评估模型是否能超越记忆模式，展现独立探索和创新解决能力。问题采用统一格式：'命题描述 + 若成立则证明，若不成立则提供反例'，鼓励模型进行多维度逻辑分析和综合思考。
*   **数据收集与构建**：数据集由数学领域专家（教授和研究生）设计和标注，确保逻辑严谨性和数学正确性。包含 179 个问题，其中 60% 为本科水平，40% 为硕士水平；40% 为证明题，60% 为反例题，覆盖多个难度和类型，力求全面评估创造力。
*   **评估框架与流程**：设计了定量和定性结合的评估体系。定量指标包括'方向准确性'（判断命题是否成立）和'过程准确性'（解题过程的正确性）；定性评估由专家手动评分，关注逻辑严谨性、表达清晰度和解决方案的原创性。实验通过统一 API 接口和标准化提示格式，确保评估的公平性和可重复性。

## Experiment

*   **效果表现**：实验评估了五个主流 LLMs（GPT O3-mini, Claude-3-7-Sonnet, Gemini-2.0-Flash, DeepSeek R1, Qwen QwQ-32B），结果显示即使在宽松评分标准下（仅关注核心解题要素，忽略小错误），最佳模型 O3 Mini 仅在本科水平的基本构造性任务上达到 70% 准确率。随着任务难度增加，模型性能显著下降，尤其在开放性问题上，未能提供有意义的策略。
*   **实验设置合理性**：实验涵盖不同规模和架构的模型，测试环境统一，确保公平性。数据集分布合理（本科 60%，硕士 40%；证明题 40%，反例题 60%），覆盖多个数学领域，评估全面。
*   **局限性分析**：实验揭示模型在创造性任务上的表现更多基于记忆模式的重组，而非真正的创造性洞察，特别是在高难度和开放性问题上表现不佳。

## Further Thoughts

论文提出的数学创造力评估框架（新概念、新方法、新例子）启发了我：是否可以通过强化学习或专门设计的创造性训练数据（如模拟人类数学家的非结构化探索过程）进一步提升模型在开放性问题上的表现？此外，构造性问题作为评估切入点的思路是否可以扩展到其他领域，如物理学中的模型构建或计算机科学中的算法创新？另外，结合人类思维轨迹数据（如草稿、探索记录）作为训练输入，或许能更好地模拟创造性思维过程。