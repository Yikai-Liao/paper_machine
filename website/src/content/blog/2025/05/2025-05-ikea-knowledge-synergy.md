---
title: "Reinforced Internal-External Knowledge Synergistic Reasoning for Efficient Adaptive Search Agent"
pubDatetime: 2025-05-12T14:21:57+00:00
slug: "2025-05-ikea-knowledge-synergy"
type: "arxiv"
id: "2505.07596"
score: 0.4912903153913076
author: "grok-3-latest"
authors: ["Ziyang Huang", "Xiaowei Yuan", "Yiming Ju", "Jun Zhao", "Kang Liu"]
tags: ["LLM", "Retrieval Augmented Generation", "Reinforcement Learning", "Knowledge Boundary", "Synergistic Reasoning"]
institution: ["Institute of Automation, Chinese Academy of Sciences", "University of Chinese Academy of Sciences", "Beijing Academy of Artificial Intelligence"]
description: "本文提出 IKEA 框架，通过强化学习和知识边界感知的奖励设计与数据集构建，使大型语言模型在知识密集型任务中高效协同内部与外部知识，显著提升准确性并减少不必要检索。"
---

> **Summary:** 本文提出 IKEA 框架，通过强化学习和知识边界感知的奖励设计与数据集构建，使大型语言模型在知识密集型任务中高效协同内部与外部知识，显著提升准确性并减少不必要检索。 

> **Keywords:** LLM, Retrieval Augmented Generation, Reinforcement Learning, Knowledge Boundary, Synergistic Reasoning

**Authors:** Ziyang Huang, Xiaowei Yuan, Yiming Ju, Jun Zhao, Kang Liu

**Institution(s):** Institute of Automation, Chinese Academy of Sciences, University of Chinese Academy of Sciences, Beijing Academy of Artificial Intelligence


## Problem Background

大型语言模型（LLMs）在知识密集型任务中常作为搜索代理（Search Agent）使用，但现有方法过度依赖外部检索（Retrieval），未充分利用内部参数知识（Parametric Knowledge），导致冗余检索、知识冲突（外部检索结果与内部知识矛盾）以及推理延迟增加的问题。
作者提出，关键挑战在于训练一个高效的自适应搜索代理，使其能够识别自身知识边界（Knowledge Boundary），优先使用内部知识，仅在内部知识不足或不确定时才进行外部检索，以提升效率和准确性。

## Method

*   **核心思想**：通过强化学习（Reinforcement Learning, RL）构建一个高效自适应搜索代理框架——IKEA（Reinforced Internal-External Knowledge Synergistic Reasoning Agent），实现内部参数知识与外部检索知识的协同推理，减少不必要的检索。
*   **框架设计**：
    *   **IKEA 代理提示模板**：设计特定的提示模板，引导模型自主判断其知识边界，优先利用内部知识，仅在知识不确定或不足时调用外部搜索引擎，通过结构化的交互格式（如 <THINK>, <SEARCH>, <ANSWER> 标签）规范推理过程。
    *   **知识边界感知奖励函数**：设计多组件奖励函数，包括答案正确性奖励（Answer Reward，若答案正确为 1，否则为 0）、知识边界奖励（Knowledge Boundary Reward，根据检索次数调整，鼓励减少不必要检索）和格式奖励（Format Reward，确保输出格式正确）。奖励函数旨在平衡内部与外部知识使用：当答案正确时，减少检索次数可获得更高奖励；当答案错误时，鼓励外部检索。
    *   **知识边界感知训练数据集**：构建平衡数据集，包含 1:1 比例的‘易答问题’（内部知识可解决）和‘难答问题’（需外部知识），通过在上下文学习（In-context Learning）中多次采样判断问题难度，确保模型自适应选择知识来源。
    *   **强化学习算法**：采用 Group Relative Policy Optimization (GRPO) 算法，通过多轮 rollout 计算组内相对奖励作为优势估计，避免传统 Proximal Policy Optimization (PPO) 算法中额外值模型的内存开销，提升训练效率。
*   **关键创新**：通过奖励设计和数据构建，动态调整检索时机（Retrieval Timing），使模型在内部知识足够时避免冗余检索，在知识不足时主动获取外部信息，从而实现高效协同推理。

## Experiment

*   **性能提升**：IKEA 在多个知识密集型推理任务数据集（如 NQ, PopQA, HotpotQA, 2Wiki）上显著优于基线方法（如 Search-R1, R1, RAG 等），在 Qwen2.5-3B 模型上平均准确率（Exact Match, EM）提升 5.47%，在 Qwen2.5-7B 模型上提升 5.05%，尤其在‘难答问题’子集上通过外部检索弥补内部知识不足，表现突出。
*   **效率提升**：IKEA 显著减少了检索次数（Retrieval Times, RT），例如在 Qwen2.5-7B 模型上相比 Search-R1 降低了约 50.81%，表明模型学会了在内部知识足够时避免不必要的检索，减少推理延迟。
*   **泛化能力**：在分布外数据集（Out-of-Distribution Datasets）上，IKEA 表现出较强的泛化能力，表明其通过自探索获得的知识寻求行为具有普适性。
*   **实验设置合理性**：实验覆盖不同模型规模（3B 和 7B）和类型（Base 和 Instruct），对比多种基线方法（包括无参数更新、SFT/DPO 和 RL 方法），并通过消融实验验证奖励设计和数据集难度的关键作用，设置较为全面。
*   **不足之处**：实验未报告误差条（Error Bars）或统计显著性检验，可能是由于计算资源限制，这在论文中也有提及。

## Further Thoughts

知识边界感知的概念非常具有启发性，通过强化学习让模型自主学习区分内部与外部知识的需求，这一思想可以扩展到模型可解释性或自适应学习领域；此外，奖励函数的多组件设计（平衡答案正确性与检索效率）为多目标优化任务提供了新思路，未来可以探索更动态的奖励调整机制；最后，平衡数据集的构建方法（1:1 比例的易答与难答问题）对解决数据偏见问题有借鉴意义，或许可以进一步研究如何根据模型学习进度动态调整数据集难度。