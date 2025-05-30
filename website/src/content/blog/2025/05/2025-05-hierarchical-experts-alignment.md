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
institution: ["Harbin Institute of Technology (Shenzhen)", "Nanyang Technological University (Singapore)", "Peng Cheng Laboratory", "Beijing Academy of Artificial Intelligence", "Tencent"]
description: "本文提出 `HoE` 框架，通过分层专家系统实现大型语言模型的多目标对齐，在 Pareto 前沿上动态适应用户偏好，同时显著降低训练和推理成本。"
---

> **Summary:** 本文提出 `HoE` 框架，通过分层专家系统实现大型语言模型的多目标对齐，在 Pareto 前沿上动态适应用户偏好，同时显著降低训练和推理成本。 

> **Keywords:** LLM, Multi-Objective Alignment, Mixture of Experts, Parameter Efficiency, Pareto Frontier

**Authors:** Zhuo Li, Guodong Du, Weiyang Guo, Yigeng Zhou, Xiucheng Li, Wenya Wang, Fangming Liu, Yequan Wang, Deheng Ye, Min Zhang, Jing Li

**Institution(s):** Harbin Institute of Technology (Shenzhen), Nanyang Technological University (Singapore), Peng Cheng Laboratory, Beijing Academy of Artificial Intelligence, Tencent


## Problem Background

大型语言模型（LLM）在同时满足多维度人类偏好（如帮助性、无害性、幽默感等）时面临挑战，因为这些目标往往相互冲突。
现有对齐方法（如 RLHF 或 DPO）通过训练多个模型或线性组合奖励信号来处理多目标，但效率低下，难以在 Pareto 前沿上实现灵活的偏好权衡。
论文旨在解决如何在不增加大量训练成本和参数开销的情况下，让 LLM 动态适应多样化的用户偏好，并在整个 Pareto 前沿上实现最优对齐。

## Method

*   **核心思想:** 提出 `HoE`（Hierarchical Mixture-of-Experts），一个轻量级、参数高效且即插即用的框架，通过分层专家系统分解多目标对齐问题，避免单一模型覆盖整个 Pareto 前沿的瓶颈。
*   **具体实现:** 包含三个层次的组件：
    *   **LoRA Experts（第一层）**：从现成的单目标模型中通过任务向量奇异值分解（task-SVD）提取紧凑的 LoRA 适配器，作为单目标专家，专注于特定目标（如帮助性）；同时通过模型融合技术（如参数级别的非线性合并）合成多目标 LoRA 专家，覆盖 Pareto 前沿的中间偏好点（如 [0.5, 0.5]）。
    *   **Router Experts（第二层）**：引入轻量级路由专家，通过训练一个小型线性网络，基于输入隐藏状态和用户偏好动态选择和组合 LoRA 专家，实现模块级别的细粒度调整；路由专家参数极小，但能显著提升覆盖效果。
    *   **Preference Routing（第三层）**：一个无参数的几何邻近模块，将用户连续偏好向量映射到离散专家子集，通过欧几里得距离选择最近的 N 个专家，指导下游路由专家的激活。
*   **关键优势:** 不需要重新训练整个模型，仅对路由专家进行少量训练；通过专家的动态组合实现对 Pareto 前沿的精准控制，同时保持低存储和推理成本。

## Experiment

*   **有效性:** 在 6 个基准数据集、14 个目标和 200 种偏好组合的测试中，`HoE` 在两目标、三目标和多目标对齐任务中均表现出色，Pareto 前沿显著优于 RS、MOD 等 15 个基线，接近理论上限（如 MORLHF）；例如，在 HelpAssistant 任务中，`HoE` 的前沿全面主导其他方法。
*   **效率与优越性:** 相比需要多模型存储或多轮推理的基线（如 MetaAligner、PAD），`HoE` 的存储和推理成本极低，以 Llama2-7B 为例，推理时间仅增加约 23%，而其他方法可能翻倍甚至更多；训练成本也大幅降低，仅需少量路由专家参数训练。
*   **实验设置合理性:** 实验覆盖多任务学习（MTL）和多目标对齐（MOA）场景，测试了不同模型（如 Llama2-7B、Llama3.1-8B），通过 GPT-4 胜率和奖励模型分数双重验证结果，设置全面且合理；但对极端偏好的覆盖略显不足，部分权重下略逊于 RiC。

## Further Thoughts

论文的多目标分解策略和无训练模型融合方法极具启发性，是否可以将这种专家系统思想推广到其他领域，如图像生成模型的多风格对齐？
此外，Router Experts 的动态路由机制让我思考是否可以通过更复杂的路由策略（如基于强化学习的路由）进一步提升性能，尤其是在处理极端偏好时。
另一个想法是探索预训练模型的知识模块化，通过提取更多‘通用专家’来实现跨任务的高效复用，减少对单目标模型的依赖。