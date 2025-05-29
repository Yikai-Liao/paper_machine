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
description: "本文提出 MoH 框架，通过大型语言模型构建元优化器自动生成优化器，用于设计高效的组合优化启发式算法，显著提升了性能和跨规模任务的泛化能力。"
---

> **Summary:** 本文提出 MoH 框架，通过大型语言模型构建元优化器自动生成优化器，用于设计高效的组合优化启发式算法，显著提升了性能和跨规模任务的泛化能力。 

> **Keywords:** LLM, Meta-Optimization, Heuristic Design, Combinatorial Optimization, Generalization

**Authors:** Yiding Shi, Wen Song, Yaoxin Wu, Jianan Zhou, Jieyi Bi, Jie Zhang

**Institution(s):** Nanyang Technological University, Shandong University, Eindhoven University of Technology


## Problem Background

组合优化问题（Combinatorial Optimization Problems, COPs）如旅行商问题（TSP）和装箱问题（BPP）是 NP 难问题，传统上依赖专家设计启发式算法（Heuristics），但人工设计耗时长且高度依赖领域知识，难以适应多样化任务和问题规模。
近年来，大型语言模型（LLMs）被用于生成启发式算法，然而现有方法（如 LLM-EC 框架）受限于手动预定义的进化计算优化器，探索空间不足，且多针对单一任务训练，泛化能力有限，尤其在跨规模任务上表现不佳。
本文旨在通过元优化（Meta-Optimization）自动发现有效的优化器，解决探索空间受限和泛化能力不足的关键问题。

## Method

*   **核心思想:** 提出 Meta-Optimization of Heuristics (MoH) 框架，利用大型语言模型（LLMs）构建一个元优化器（Meta-Optimizer），该元优化器能够自主生成多样化的优化器（Optimizers），进而用于下游组合优化任务的启发式算法设计，突破传统固定优化器的限制。
*   **双层优化结构:** 外层循环（Outer Loop）负责优化器设计，元优化器通过自调用（Self-Invocation）生成一组候选优化器；内层循环（Inner Loop）负责启发式算法设计，每个候选优化器用于生成针对特定任务的启发式算法，并通过效用函数（Utility Function）评估其性能，最终选择效用最高的优化器作为下一轮的元优化器。
*   **多任务训练框架:** MoH 在多个任务（如不同规模的 TSP 实例）上同时训练元优化器，通过任务多样性提升泛化能力，避免单一任务训练导致的过拟合问题。
*   **个体与种群管理:** 优化器和启发式算法被定义为包含代码实现、自然语言描述和效用分数的个体，种群通过效用排序动态更新，确保探索过程中的多样性和稳定性。
*   **生成过程:** 利用 LLMs 的上下文推理能力，通过精心设计的提示（Prompting）生成优化器和启发式的创意（Idea Generation）和代码实现，自然语言描述在探索多样性解决方案中起到关键作用。
*   **灵活性与创新性:** MoH 生成的优化器涵盖了遗传算法、蚁群优化、模拟退火等多种策略，甚至包括混合或非常规方法，显著扩展了搜索空间。

## Experiment

*   **有效性:** 在 TSP 构造型启发式（Constructive Heuristic）中，MoH 取得了最低的平均最优性差距（Optimality Gap）11.792%，显著优于其他 LLM 基线（如 FunSearch, EoH）；在改进型启发式（Improvement Heuristic）中，MoH 的最优性差距低至 0.391%，表现出色。
*   **泛化能力:** MoH 在跨规模任务（如从 TSP20 泛化到 TSP1000）上表现优异，尤其在大问题规模上优于其他方法，验证了多任务训练框架在提升泛化能力方面的有效性。
*   **实验设置合理性:** 实验覆盖了多种问题类型（TSP, BPP, CVRP）和规模，训练和测试数据集设计合理（如 TSP 训练于 20-200 规模，测试泛化至 500-1000 规模），并控制了计算预算（如启发式评估次数限制为 1000 次），确保了与其他方法的公平对比。
*   **计算开销:** MoH 的训练和推理时间较长（如 TSP-GLS 训练耗时 238.4 分钟），LLM 请求和 token 使用量较高，表明其计算成本较高，是未来优化的方向。
*   **对比分析:** 相较于传统方法（如 Nearest Neighbor）和现有 LLM 基线，MoH 在性能和泛化能力上均有显著提升，数据支持了其方法论的有效性。

## Further Thoughts

元优化思想（Meta-Optimization）是一个极具启发性的创新，通过 LLMs 自动设计优化器而非直接设计解决方案，展现了更高层次的抽象能力，这种方法可能不仅限于组合优化领域，还可以应用于超参数优化、神经网络架构搜索等领域；此外，将自然语言描述（Idea Generation）融入代码生成过程，利用 LLMs 的语言理解能力增强探索多样性，这一策略对其他生成任务（如程序合成或自动化设计）具有借鉴意义；最后，多任务训练框架提升了模型对不同规模和类型任务的适应性，这一思想可以扩展到其他需要跨域或跨规模适应的 AI 系统中。