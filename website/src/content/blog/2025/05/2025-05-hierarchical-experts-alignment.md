---
title: "Multi-objective Large Language Model Alignment with Hierarchical Experts"
pubDatetime: 2025-05-27T09:15:03+00:00
slug: "2025-05-hierarchical-experts-alignment"
type: "arxiv"
id: "2505.20925"
score: 0.8033755926678979
author: "grok-3-latest"
authors: ["Zhuo Li", "Guodong Du", "Weiyang Guo", "Yigeng Zhou", "Xiucheng Li", "Wenya Wang", "Fangming Liu", "Yequan Wang", "Deheng Ye", "Min Zhang", "Jing Li"]
tags: ["LLM", "Multi-Objective Alignment", "Mixture of Experts", "Parameter Efficiency", "Pareto Frontier"]
institution: ["Harbin Institute of Technology, Shenzhen, China", "Nanyang Technological University, Singapore", "Peng Cheng Laboratory, China", "Beijing Academy of Artificial Intelligence, China", "Tencent, China"]
description: "本文提出 `HoE` 框架，通过分层专家系统实现大型语言模型的多目标对齐，以轻量级和参数高效的方式在 Pareto 前沿上取得优越性能。"
---

> **Summary:** 本文提出 `HoE` 框架，通过分层专家系统实现大型语言模型的多目标对齐，以轻量级和参数高效的方式在 Pareto 前沿上取得优越性能。 

> **Keywords:** LLM, Multi-Objective Alignment, Mixture of Experts, Parameter Efficiency, Pareto Frontier

**Authors:** Zhuo Li, Guodong Du, Weiyang Guo, Yigeng Zhou, Xiucheng Li, Wenya Wang, Fangming Liu, Yequan Wang, Deheng Ye, Min Zhang, Jing Li

**Institution(s):** Harbin Institute of Technology, Shenzhen, China, Nanyang Technological University, Singapore, Peng Cheng Laboratory, China, Beijing Academy of Artificial Intelligence, China, Tencent, China


## Problem Background

大型语言模型（LLM）在同时满足多种人类偏好（如帮助性、无害性、幽默感等）时面临挑战，因为这些目标往往相互冲突，导致现有对齐方法难以在 Pareto 前沿上实现灵活权衡，同时训练和存储成本高昂。
论文旨在解决如何在不进行昂贵重训练的情况下，动态适应多样化的用户偏好，并在多目标对齐中取得最优性能。

## Method

*   **核心思想:** 提出 `HoE`（Hierarchical Mixture-of-Experts），一个轻量级、参数高效且即插即用的框架，通过分层专家系统分解多目标对齐问题，避免单一模型覆盖整个 Pareto 前沿的困境。
*   **具体实现:** 包含三个层次的组件：
    *   **LoRA Experts（LoRA 专家）:** 从现成的单目标模型中通过任务向量奇异值分解（task-SVD）提取紧凑的 LoRA 适配器，作为单目标专家，专注于特定目标（如帮助性）；同时通过模型融合技术（如参数级别的选择性增强或抑制），合成多目标 LoRA 专家，覆盖 Pareto 前沿的中间偏好点（如 [0.5, 0.5]）。
    *   **Router Experts（路由专家）:** 引入轻量级线性路由网络，作为次级专家，参数规模远小于 LoRA 专家；通过训练，基于输入隐藏状态和用户偏好动态选择和组合 LoRA 专家，实现模块级别的细粒度调整；路由专家针对特定偏好优化，避免了 LoRA 专家数量过多带来的参数开销。
    *   **Preference Routing（偏好路由）:** 一个无参数的几何邻近模块，将用户输入的连续偏好向量映射到离散的专家子集，通过欧几里得距离选择最近的 N 个专家，指导下游路由专家的激活，实现对 Pareto 前沿的精确导航。
*   **关键优势:** 不需要对基础模型或 LoRA 专家进行重训练，仅对少量路由专家进行轻量级优化；通过分层设计平衡了性能、参数成本和训练效率，支持任意偏好的动态适应。

## Experiment

*   **有效性:** 在 6 个基准数据集上测试了 14 个目标和 200 种偏好组合，`HoE` 在两目标、三目标和多目标对齐任务中均显著优于 15 个基线方法（如 RS、MOD、RiC），Pareto 前沿接近理论上限（如 MORLHF），例如在 'Summary & Deberta' 设置中比 RiC 提升了 (+2, +0.8) 的奖励分数。
*   **效率与优越性:** 相比需要多模型存储或多轮推理的基线（如 MetaAligner、PAD），`HoE` 的存储成本低（仅 7.64B 参数 vs. PAD 的 14B），推理成本小（1.23 倍 vs. PAD 的 2.98 倍），仅需激活少量专家即可完成任务；消融研究表明 LoRA 专家与路由专家的组合在性能和参数效率间达到最佳平衡。
*   **全面性与合理性:** 实验覆盖多种任务类型（帮助性助手、数学、总结等）和模型规模（LLaMA2-7B、LLaMA3.1-8B），验证了方法的泛化能力；同时在多任务学习场景中也表现出色；评价指标包括奖励模型分数、GPT-4 胜率和 PASS@1 准确率，设置全面且合理。
*   **局限性:** 论文提到依赖现成单目标模型，若无可用模型则需从头训练，可能限制应用场景；此外，模型融合和 SVD 压缩在某些目标上的效果可能不佳。

## Further Thoughts

论文中的分解策略启发了我：将复杂多目标问题分解为单偏好子问题并通过专家系统解决的思路，可以扩展到其他领域，如多任务学习或个性化推荐中，通过分层专家动态适配不同需求；此外，无训练模型融合和动态路由机制提示我们可以在推理时引入更多上下文自适应策略，例如基于用户历史交互调整专家选择权重，进一步提升模型对复杂场景的适应性。