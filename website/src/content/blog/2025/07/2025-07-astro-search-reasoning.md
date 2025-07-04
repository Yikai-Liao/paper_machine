---
title: "ASTRO: Teaching Language Models to Reason by Reflecting and Backtracking In-Context"
pubDatetime: 2025-07-01T04:10:15+00:00
slug: "2025-07-astro-search-reasoning"
type: "arxiv"
id: "2507.00417"
score: 0.7918247772530027
author: "grok-3-latest"
authors: ["Anonymous Authors"]
tags: ["LLM", "Search Behavior", "Reasoning", "Reinforcement Learning", "Supervised Fine-Tuning"]
institution: ["Unknown Institutions"]
description: "本文提出Astro框架，通过搜索轨迹生成、监督微调和强化学习将搜索行为内化到大型语言模型中，显著提升了其数学推理能力。"
---

> **Summary:** 本文提出Astro框架，通过搜索轨迹生成、监督微调和强化学习将搜索行为内化到大型语言模型中，显著提升了其数学推理能力。 

> **Keywords:** LLM, Search Behavior, Reasoning, Reinforcement Learning, Supervised Fine-Tuning

**Authors:** Anonymous Authors

**Institution(s):** Unknown Institutions


## Problem Background

大型语言模型（LLM）在推理能力上的提升通常依赖于从已有强大推理模型中蒸馏知识或直接应用强化学习（RL），但这些方法需要模型具备一定的推理基础。
本文提出了一种新框架'Astro'，旨在解决如何从头开始（ab initio）将搜索行为注入到不具备强大推理能力的模型中，从而提升其数学推理能力，特别是在面对复杂问题时能够自主探索、反思和回溯。

## Method

*   **核心思想:** 通过系统化的框架'Astro'（Autoregressive Search-Taught Reasoner），将搜索行为内化到语言模型中，使其能够在单次推理中生成包含自我反思和回溯的完整搜索轨迹，从而提升推理能力。
*   **具体实现:** 框架分为三个阶段：
    *   **搜索轨迹生成:** 使用蒙特卡洛树搜索（MCTS）探索数学问题的解空间，生成包含推理步骤的搜索树；将搜索树线性化为节点序列，包含正确和错误答案的路径；最终将节点序列转化为自然语言的思维链（Chain-of-Thought, CoT），注入自我反思和回溯模式。
    *   **监督微调（SFT）:** 基于生成的搜索轨迹数据，对模型进行微调，使其学习搜索行为，包括如何在推理过程中进行反思和回溯；这一阶段使用高质量的搜索轨迹数据集，为后续强化学习提供有益的先验。
    *   **强化学习（RL）:** 使用改进的Group Relative Policy Optimization（GRPO）算法进一步优化模型，增强其推理和搜索能力；通过对中等难度的数学问题进行训练，促使模型在搜索过程中生成正确的答案。
*   **关键点:** 不依赖外部搜索框架（如束搜索），而是将搜索过程完全内化到模型中；通过程序克隆（Procedure Cloning）技术，模型能够学习到探索和修正的模式，同时保持推理轨迹的结构化（可映射为有向图）。

## Experiment

*   **有效性:** 基于Llama-3.1-70B-Instruct模型，Astro框架在监督微调（SFT）阶段将MATH-500准确率从65.8%提升至69.6%，在强化学习（RL）阶段进一步提升至81.8%；在AMC 2023和AIME 2024上也有显著提升（分别达到64.4%和30.0%），优于多个基线模型。
*   **对比分析:** 与不含搜索先验的模型（Direct-SFT和Direct-RL）相比，Astro模型在所有基准测试中表现更优，证明搜索先验对推理能力提升的关键作用；同时，模型生成的思维链（CoT）长度增加，回溯频率与性能呈正相关（Pearson系数0.816-0.854）。
*   **实验设置合理性:** 实验覆盖多个数学数据集（MATH-500, AMC 2023, AIME 2024），评估指标包括pass@1和maj@8，考虑了多轮生成平均值（16次），数据集选择聚焦中等难度问题，设置较为全面合理。
*   **局限性:** 在Llama-3.3-70B-Instruct模型上观察到不稳定性（CoT长度爆炸），可能限制了框架的通用性，需进一步研究。

## Further Thoughts

Astro框架中搜索行为的内化是一个重要的启发，通过程序克隆和搜索轨迹生成，模型能够在推理时自主进行探索和修正，这种方法不仅适用于数学推理，还可能推广到其他复杂决策任务（如代码生成或逻辑推理）；此外，搜索先验与强化学习的结合提供了一种新的训练范式，即先通过SFT注入结构化行为，再通过RL优化性能，这种两阶段方法可能对未来的模型训练有广泛启发；回溯行为与性能的正相关性也提示我们可以通过设计奖励机制，鼓励模型在推理中进行更多自我修正。