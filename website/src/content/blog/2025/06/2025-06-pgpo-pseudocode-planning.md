---
title: "PGPO: Enhancing Agent Reasoning via Pseudocode-style Planning Guided Preference Optimization"
pubDatetime: 2025-06-02T09:35:07+00:00
slug: "2025-06-pgpo-pseudocode-planning"
type: "arxiv"
id: "2506.01475"
score: 0.6514146433292717
author: "grok-3-latest"
authors: ["Zouying Cao", "Runze Wang", "Yifei Yang", "Xinbei Ma", "Xiaoyong Zhu", "Bo Zheng", "Hai Zhao"]
tags: ["LLM", "Agent Reasoning", "Planning", "Preference Optimization", "Generalization"]
institution: ["Shanghai Jiao Tong University", "Alibaba Group"]
description: "本文提出 PGPO 方法，通过伪代码风格计划（P-code Plan）和规划引导的偏好优化显著提升了大型语言模型智能体的推理能力和泛化性能，特别是在未见任务和复杂交互场景中。"
---

> **Summary:** 本文提出 PGPO 方法，通过伪代码风格计划（P-code Plan）和规划引导的偏好优化显著提升了大型语言模型智能体的推理能力和泛化性能，特别是在未见任务和复杂交互场景中。 

> **Keywords:** LLM, Agent Reasoning, Planning, Preference Optimization, Generalization

**Authors:** Zouying Cao, Runze Wang, Yifei Yang, Xinbei Ma, Xiaoyong Zhu, Bo Zheng, Hai Zhao

**Institution(s):** Shanghai Jiao Tong University, Alibaba Group


## Problem Background

大型语言模型（LLM）作为智能体在处理复杂交互任务时，规划能力是推理的关键组成部分，但现有自然语言（NL）计划存在语义模糊、冗长低效的问题，且难以泛化到未见过的相似任务。
论文旨在解决如何设计更结构化、更简洁的计划格式，以提升 LLM 智能体的泛化能力和推理效率。

## Method

*   **伪代码风格计划（P-code Plan）:** 提出使用伪代码格式来表示计划，捕捉推理的结构化逻辑，相比自然语言计划更简洁且易于泛化。
    *   **生成流程:** 从 ReAct 风格数据集中提取思想（Thought）部分，利用 GPT-4o 模型通过少样本提示（Few-shot Prompting）生成符合预定义格式的 P-code Plan，并通过人工验证确保质量。
    *   **优势:** P-code Plan 通过规划步骤（Planning Step）和规划实体（Planning Entity）定义任务逻辑，支持控制流（如 if-else, for），从而减少语义歧义，提升推理效率。
*   **规划引导的偏好优化（PGPO）:** 提出一种迭代优化方法，进一步增强智能体的规划和推理能力。
    *   **第一阶段 - 监督微调（SFT）:** 使用包含 P-code Plan 的专家数据集对基础模型进行微调，构建初始智能体，使其具备初步规划能力。
    *   **第二阶段 - 探索与对比轨迹构建:** 基于基础智能体在专家轨迹上的探索，设计两种规划导向奖励：计划驱动奖励（Plan-driven Reward）评估计划对整体轨迹的影响，计划跟随奖励（Plan-following Reward）通过 Monte Carlo 采样评估智能体对计划的执行一致性；基于奖励构建对比轨迹数据集。
    *   **第三阶段 - 偏好优化（DPO）:** 利用直接偏好优化方法，基于对比轨迹数据集优化智能体参数，提升生成高质量 P-code Plan 和后续推理的能力，同时加入 SFT 损失防止性能退化。
*   **关键点:** 方法不依赖闭源模型，适用于开源 LLM，且通过迭代优化平衡了规划质量与执行效率。

## Experiment

*   **数据集与模型:** 在 ALFWorld（家庭任务）、WebShop（在线购物）和 TextCraft（游戏任务）三个基准数据集上进行实验，使用 Llama-2-7B/13B、Llama-3-8B 和 Mistral-7B 作为基础模型。
*   **效果显著性:** PGPO 在所有数据集和模型上均显著优于基线方法（如 SFT, ETO, IPR），例如在 Llama-2-7B 上平均奖励提升 7.2% 超过最先进的 IPR，尤其在未见任务（ALFWorld-Unseen）和复杂任务（TextCraft）上表现突出，验证了泛化能力；与闭源模型（GPT-3.5, GPT-4）结合提示方法（ReAct, ADaPT）相比，PGPO + Llama-3-8B 表现相当甚至更优。
*   **效率提升:** P-code Plan 显著减少了交互轮次（Interaction Turns），如在 ALFWorld 上比无计划和 NL 计划设置减少了平均轮次，表明推理效率提高；同时降低了无效动作率（23.57% vs. ETO 的 34.28%）和动作遗漏（WebShop 成功率 41.0% vs. IPR 的 40.5%）。
*   **实验合理性:** 实验设置全面，涵盖不同任务类型（已见/未见）、多种模型规模和家族，以及与闭源模型的对比；消融研究验证了 P-code Plan 和两种奖励机制的必要性。
*   **不足与成本:** Monte Carlo 采样用于计算计划跟随奖励增加了推理成本，训练时间（每迭代 3.2 小时）比 ETO（1.5 小时）高，但仍低于 IPR（4.5 小时），整体效率合理。

## Further Thoughts

P-code Plan 的结构化表示形式启发我们思考，是否可以探索其他形式化表示（如形式逻辑或图形化计划）来进一步提升智能体推理的精确性和泛化能力？此外，规划导向奖励的设计是否可以在多任务场景下通过共享元知识（Meta-Knowledge）实现更通用的奖励机制？另外，P-code Plan 生成依赖外部模型和人工验证，是否可以通过自监督学习或规则驱动的自动化生成降低成本并扩展到更广泛的应用场景？