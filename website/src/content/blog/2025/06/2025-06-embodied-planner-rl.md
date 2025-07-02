---
title: "Unleashing Embodied Task Planning Ability in LLMs via Reinforcement Learning"
pubDatetime: 2025-06-29T07:31:24+00:00
slug: "2025-06-embodied-planner-rl"
type: "arxiv"
id: "2506.23127"
score: 0.504110268992433
author: "grok-3-latest"
authors: ["Zhaoye Fei", "Li Ji", "Siyin Wang", "Junhao Shi", "Jingjing Gong", "Xipeng Qiu"]
tags: ["LLM", "Reinforcement Learning", "Task Planning", "Interaction", "Generalization"]
institution: ["Fudan University", "Shanghai Innovation Institute"]
description: "本文提出 Embodied Planner-R1 框架，通过强化学习显著提升大型语言模型在具身任务规划中的交互能力和泛化性能，为交互规划研究奠定新基准。"
---

> **Summary:** 本文提出 Embodied Planner-R1 框架，通过强化学习显著提升大型语言模型在具身任务规划中的交互能力和泛化性能，为交互规划研究奠定新基准。 

> **Keywords:** LLM, Reinforcement Learning, Task Planning, Interaction, Generalization

**Authors:** Zhaoye Fei, Li Ji, Siyin Wang, Junhao Shi, Jingjing Gong, Xipeng Qiu

**Institution(s):** Fudan University, Shanghai Innovation Institute


## Problem Background

大型语言模型（LLMs）在静态知识生成和开放域对话中表现出色，但面对具身任务规划（Embodied Task Planning）场景时，由于缺乏持续环境感知和实时反馈处理能力，难以在部分可观测环境中建立动作与环境反馈之间的因果关系。传统方法依赖静态知识生成固定动作序列，忽略动态交互，导致在复杂或未知环境中表现不佳。本文旨在解决如何在不依赖大量人工标注或预训练数据的情况下，通过自主探索提升 LLMs 的交互规划能力和泛化性能。

## Method

*   **核心框架：Embodied Planner-R1**：一个基于强化学习的端到端训练框架，通过自主探索和在线交互，使 LLMs 能够自我进化，适应多轮交互规划任务，而无需大量人工监督。
*   **具体创新点：**
    *   **Group Rollout with In-Environment Interaction**：采用并行探索策略，为每个任务-环境对生成多个副本（Replicas），同时采样多条轨迹，形成轨迹组，增加交互数据的多样性。这种方法通过在环境中直接交互，积累丰富的经验，解决单一轨迹探索的局限性。
    *   **Completion-driven Sparse Reward**：设计基于任务完成与否的二元稀疏奖励机制（完成任务奖励为 1，未完成奖励为 0），避免人为设计的复杂奖励函数，减少奖励欺骗（Reward Hacking）风险，鼓励模型通过自主探索理解环境规律，而非依赖预设规则。
    *   **Interactive Policy Optimization (IPO)**：针对多轮交互（ReAct 风格轨迹）设计的优化算法，基于组内归一化的优势估计（Group-normalized Advantage Estimator），计算轨迹中每个时间步的相对奖励，同时引入 KL 散度正则化项，防止策略偏离参考模型过多。IPO 有效处理稀疏奖励和长序列训练中的概率退化问题，提升学习效率。
*   **实现细节**：方法基于 ReAct 范式，将轨迹生成分为‘思考-动作-观察’循环，模型在每个时间步根据历史轨迹生成思考和动作，并通过环境反馈更新策略。训练不依赖专家轨迹，而是通过在线交互动态生成数据。

## Experiment

*   **有效性**：在 ALFWorld 基准上，Embodied Planner-R1 取得 97.78% 的任务完成率，在 ScienceWorld 上达到 79.92%，显著优于现有方法（如 ETO 在 ALFWorld 上为 82.07%，在 ScienceWorld 上为 56.20%），相比基础模型 Qwen2.5-7B-Instruct（ALFWorld 31.05%，ScienceWorld 22.05%）提升明显。
*   **泛化能力**：在未见环境中的性能下降仅为 -3.33%，远低于其他方法（如 ETO 为 -7.34%），尤其在复杂任务（如 ALFWorld 的‘Pick Two’）中表现出色，显示出较强的零样本泛化能力。
*   **效率提升**：通过减少无效动作，模型规划效率显著提高，例如在 ScienceWorld 中，平均响应长度从 900 个 token 降至 500 个 token，总动作步数从 20 步降至 12.5 步，无效动作几乎消除。
*   **实验设置**：实验在 ALFWorld 和 ScienceWorld 两个具身规划基准上进行，涵盖已见和未见场景，任务难度从简单（如‘Pick’）到复杂（如‘Pick Two’），并对比了多种基线模型（包括 GPT-4o、DeepSeek-V3 等），设置全面合理。
*   **局限性**：实验主要在纯文本环境中进行，未涉及多模态场景，且训练对计算资源需求较高。

## Further Thoughts

论文通过稀疏奖励和在线交互实现自主探索的思路令人启发，是否可以在其他动态决策领域（如机器人控制或多模态任务）中应用类似机制，减少对标注数据的依赖？此外，组内并行探索（Group Rollout）提供了一种高效数据收集方式，是否可以结合基于不确定性的探索策略进一步优化采样效率？另一个值得思考的点是，论文提到多轮交互任务中较短上下文长度可能带来更好性能，这提示我们在设计 LLM 应用时需重新审视上下文长度与效率的平衡，探索是否存在一个通用的最优上下文长度范围。