---
title: "Generalizable Heuristic Generation Through Large Language Models with Meta-Optimization"
pubDatetime: 2025-05-27T08:26:27+00:00
slug: "2025-05-meta-optimization-heuristics"
type: "arxiv"
id: "2505.20881"
score: 0.6729859438351832
author: "grok-3-latest"
authors: ["Yiding Shi", "Wen Song", "Yaoxin Wu", "Jianan Zhou", "Jieyi Bi", "Jie Zhang"]
tags: ["LLM", "Meta-Optimization", "Heuristic Design", "Combinatorial Optimization", "Generalization"]
institution: ["Nanyang Technological University", "Shandong University", "Eindhoven University of Technology"]
description: "本文提出 MoH 框架，通过元优化和多任务训练利用大型语言模型自动设计优化器，生成高效启发式算法，显著提升组合优化问题的解决效果和泛化能力。"
---

> **Summary:** 本文提出 MoH 框架，通过元优化和多任务训练利用大型语言模型自动设计优化器，生成高效启发式算法，显著提升组合优化问题的解决效果和泛化能力。 

> **Keywords:** LLM, Meta-Optimization, Heuristic Design, Combinatorial Optimization, Generalization

**Authors:** Yiding Shi, Wen Song, Yaoxin Wu, Jianan Zhou, Jieyi Bi, Jie Zhang

**Institution(s):** Nanyang Technological University, Shandong University, Eindhoven University of Technology


## Problem Background

组合优化问题（COPs）中的启发式算法设计传统上依赖专家知识，耗时长且针对特定任务，难以泛化到不同规模或类型的任务。
现有基于大型语言模型（LLM）的启发式设计方法受限于手动预定义的进化计算（EC）优化器，限制了搜索空间的探索，并且单一任务训练导致泛化能力不足。
本文旨在通过自动设计优化器，生成更有效的启发式算法，并提升跨任务和跨规模的泛化能力。

## Method

*   **核心思想:** 提出 Meta-Optimization of Heuristics (MoH) 框架，通过元优化（Meta-Optimization）利用大型语言模型（LLM）自动设计优化器（Optimizer），以生成针对组合优化问题（COPs）的启发式算法，而非依赖固定的进化计算（EC）框架。
*   **双层优化结构:** 
    *   外层循环（Outer Loop）：负责优化器设计，元优化器（Meta-Optimizer）通过自调用（Self-Invocation）生成一组候选优化器，并基于下游任务的效用评估选择最优优化器作为下一轮的元优化器。
    *   内层循环（Inner Loop）：每个候选优化器用于生成针对特定下游任务的启发式算法，通过迭代优化提升启发式性能。
*   **多任务训练:** 在多个任务（如不同规模的 TSP 问题）上同时训练，增强模型对未见任务的泛化能力，避免单一任务训练导致的过拟合。
*   **具体实现:** 
    *   利用 LLM 的上下文推理能力，通过提示（Prompting）生成优化器的自然语言描述和代码实现。
    *   优化器生成过程包括个体选择（Individual Selection，选择有潜力的候选）、想法生成（Idea Generation，利用自然语言描述提出改进思路）和实现生成（Implementation Generation，将思路转化为代码）。
    *   通过效用函数（Utility Function）评估优化器和启发式的性能，迭代更新种群（Population），保持多样性并平衡探索与利用。
*   **创新点:** MoH 突破了传统固定优化器的限制，允许动态探索更广泛的搜索空间，并通过多任务设置提升适应性。

## Experiment

*   **有效性:** 在 TSP 构造型启发式（Constructive Heuristic）中，MoH 取得平均最优性差距（Optimality Gap）11.792%，显著优于其他 LLM 基线（如 FunSearch, EoH）；在改进型启发式（Improvement Heuristic）中，最优性差距低至 0.391%，表现出色。
*   **泛化能力:** 在跨规模测试（如 TSP500 和 TSP1000）中，MoH 性能优于基线，特别是在大问题规模上，验证了多任务训练策略的有效性。
*   **实验设置合理性:** 实验覆盖多种问题（TSP, Online BPP, CVRP 等）和规模，训练和测试数据集设计合理（如 TSP20-200 训练，TSP500-1000 测试）；消融研究验证了想法生成和种群大小的影响，增强了结果可信度。
*   **计算成本:** MoH 训练和推理的计算成本较高（如 LLM 请求次数和 token 使用量），但通过控制评估次数（1000 次）确保公平性，性能提升证明成本是值得的。

## Further Thoughts

MoH 的元优化思想为算法设计提供了新视角，是否可应用于强化学习策略或软件开发的自动优化？
多任务训练提升泛化性的策略可推广至跨域 AI 任务，如自然语言处理中的跨语言模型训练。
此外，利用 LLM 自然语言能力生成算法描述再转化为代码的方式，启发我们在复杂任务设计中如何结合语言与技术实现更高效的自动化。