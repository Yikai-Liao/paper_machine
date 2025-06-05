---
title: "Self-Challenging Language Model Agents"
pubDatetime: 2025-06-02T14:23:33+00:00
slug: "2025-06-self-challenging-agents"
type: "arxiv"
id: "2506.01716"
score: 0.6685923842559047
author: "grok-3-latest"
authors: ["Yifei Zhou", "Sergey Levine", "Jason Weston", "Xian Li", "Sainbayar Sukhbaatar"]
tags: ["LLM", "Tool Use", "Reinforcement Learning", "Task Generation", "Self-Improvement"]
institution: ["UC Berkeley", "FAIR at Meta"]
description: "本文提出 Self-Challenging Agent 框架，通过 Code-as-Task 格式自动化生成高质量任务并结合强化学习，显著提升大型语言模型代理在多轮工具使用任务中的性能。"
---

> **Summary:** 本文提出 Self-Challenging Agent 框架，通过 Code-as-Task 格式自动化生成高质量任务并结合强化学习，显著提升大型语言模型代理在多轮工具使用任务中的性能。 

> **Keywords:** LLM, Tool Use, Reinforcement Learning, Task Generation, Self-Improvement

**Authors:** Yifei Zhou, Sergey Levine, Jason Weston, Xian Li, Sainbayar Sukhbaatar

**Institution(s):** UC Berkeley, FAIR at Meta


## Problem Background

大型语言模型（LLM）作为智能代理在多轮工具使用任务中展现出巨大潜力，但训练此类代理需要大量高质量任务数据、工具和评估标准，而这些通常依赖人工创建，成本高且难以扩展。
论文旨在解决如何自动化生成高质量合成任务，以减少人工标注需求，并在开放式、工具丰富的环境中提升 LLM 代理性能。

## Method

*   **核心框架：Self-Challenging Agent (SCA)**：提出了一种自挑战框架，让 LLM 代理通过两个角色（任务挑战者和任务执行者）实现任务生成与学习。
*   **任务挑战者（Challenger）**：代理通过与环境交互，探索工具功能和环境状态，生成合成任务。任务采用‘Code-as-Task’（CaT）格式，包括以下组件：
    *   **指令（Instruction）**：描述任务目标，模拟用户请求。
    *   **验证函数（Verification Function）**：以代码形式定义任务成功标准，可自动评估结果。
    *   **示例解决方案（Example Solution）**：提供一种可行的解决方案，确保任务可行性。
    *   **失败案例（Failure Cases）**：列出不正确的解决方案，确保任务具有挑战性并避免验证函数过于宽松。
    通过自动过滤机制（验证解决方案通过、失败案例不通过），确保任务质量。
*   **任务执行者（Executor）**：在自生成任务上通过强化学习（RL）训练，使用验证函数的反馈作为奖励信号。支持两种训练模式：
    *   **自改进（Self-Improvement）**：代理通过自身生成的轨迹学习，使用简单 RL 方法（如 Rejection Fine-Tuning）优化。
    *   **蒸馏（Distillation）**：利用强大模型生成示范轨迹，通过监督微调（SFT）将知识转移到较弱模型。
*   **关键创新**：CaT 格式利用代码的通用性和可执行性，确保任务可验证性和质量；任务挑战者通过环境交互生成多样化任务，避免了依赖静态初始观察的局限。

## Experiment

*   **有效性**：在 M[3] ToolEval 和 Tau-Bench 两个基准数据集（涵盖计算、网页浏览、零售、航空四个环境）上，SCA 显著提升了代理性能。自改进场景下，Llama-3.1-8B 的平均成功率从 12.0% 提升至 23.5%（几乎翻倍）；蒸馏场景下，成功率提升 20.2%，达到 32.2%。
*   **对比分析**：与基线方法 PAE 相比，SCA 在部分可观察环境（如零售和航空）表现更优，自改进场景中平均成功率高出 10.6%，因其任务挑战者通过环境交互生成更精确、多样的任务。
*   **实验设置合理性**：实验覆盖多种环境和训练场景（自改进与蒸馏），测试了多种 RL 算法（如 Rejection Fine-Tuning、DPO、PPO、GRPO），设置全面。数据表明任务数量增加（如 800 个任务）时泛化能力更强，但也指出较弱模型任务分布可能因过滤机制而单一，跨环境泛化有限。
*   **额外分析**：消融实验验证了 CaT 各组件（如验证函数、失败案例）对任务质量的提升作用；在线 RL 算法（如 PPO）可进一步提升性能，但稳定性较低。

## Further Thoughts

Code-as-Task（CaT）格式利用代码作为任务定义的通用接口，通过自动过滤机制大幅提升任务质量，这启发我思考是否可以将代码化思路扩展到环境建模或工具 API 生成，动态创建开放式训练环境。此外，SCA 的双角色设计（挑战者与执行者）让人联想到人类‘教学相长’模式，是否可以通过多代理协作（如多个挑战者生成任务，多个执行者竞争解决）进一步提升任务多样性和学习效率？