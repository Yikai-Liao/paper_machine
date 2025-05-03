---
title: "Reinforced MLLM: A Survey on RL-Based Reasoning in Multimodal Large Language Models"
pubDatetime: 2025-04-30T03:14:28+00:00
slug: "2025-04-rl-mllm-reasoning"
type: "arxiv"
id: "2504.21277"
score: 0.6170014601192129
author: "grok-3-latest"
authors: ["Guanghao Zhou", "Panjia Qiu", "Cen Chen", "Jie Wang", "Zheming Yang", "Jian Xu", "Minghui Qiu"]
tags: ["MLLM", "Reinforcement Learning", "Reasoning", "Cross-Modal Alignment", "Reward Design"]
institution: ["East China Normal University", "ByteDance"]
description: "本文系统综述了强化学习（RL）在多模态大语言模型（MLLMs）推理中的应用，分析了算法设计、奖励机制和应用场景，提出了当前局限和未来方向，为多模态推理研究提供了结构化指南。"
---

> **Summary:** 本文系统综述了强化学习（RL）在多模态大语言模型（MLLMs）推理中的应用，分析了算法设计、奖励机制和应用场景，提出了当前局限和未来方向，为多模态推理研究提供了结构化指南。 

> **Keywords:** MLLM, Reinforcement Learning, Reasoning, Cross-Modal Alignment, Reward Design

**Authors:** Guanghao Zhou, Panjia Qiu, Cen Chen, Jie Wang, Zheming Yang, Jian Xu, Minghui Qiu

**Institution(s):** East China Normal University, ByteDance


## Problem Background

多模态大语言模型（MLLMs）在扩展大语言模型（LLMs）能力以处理视觉、音频、视频等多种模态输入方面取得了显著进展，但如何在多模态场景下实现稳健的推理仍是一个关键挑战。
传统的监督微调（SFT）方法面临标注成本高和灾难性遗忘等问题，而强化学习（RL）通过优化推理路径和对齐多模态信息，提供了一种提升 MLLMs 推理能力的有效途径。
本文旨在探索 RL 如何解决跨模态对齐、推理路径优化和泛化能力不足等问题。

## Method

*   **核心范式:** 论文系统回顾了基于强化学习的推理方法，分为无价值方法（Value-Free，如 GRPO）和有价值方法（Value-Based，如 PPO），通过将推理过程建模为马尔可夫决策过程（MDP），优化推理路径的预期回报。
*   **奖励机制设计:** 包括结果导向奖励机制（ORM），关注最终输出的正确性；以及过程导向奖励机制（PRM），强调中间推理步骤的质量。奖励设计还考虑了跨模态交互（如 UI-R1 的多模态奖励框架）和结构化推理路径（如 R1-VL 的 StepGRPO）。
*   **训练范式迁移:** 从 LLM 到 MLLM 的 RL 训练范式（如 R1 范式）迁移，涉及冷启动策略（如 Vision-R1 的 PTST）和多阶段训练（如 LMM-R1 的文本到多模态训练）。
*   **效率与稳定性优化:** 采用课程学习（如 Curr-ReFT 的分阶段任务难度递增）、数据高效策略（如 Reason-RFT 的高质量样本筛选）和 KL 正则化缓解灾难性遗忘。
*   **关键创新:** 强调多模态特异性奖励设计和轻量化训练（如 Skywork R1V 的视觉投影模块），以适应多模态任务的复杂性和资源限制。

## Experiment

*   **有效性:** RL 基 MLLMs 在多个基准测试（如 MathVista、MMMU）上表现出显著提升，例如 Vision-R1 和 R1-Onevision 在数学和科学推理任务中相较非 RL 方法有明显优势，尤其在跨模态推理和泛化能力方面。
*   **实验设置:** 实验覆盖数学推理、图表分析、科学推理、空间推理等多个领域，数据集设计较为全面，包含结构化知识和现实场景任务（如 Table 5 所示）。
*   **局限性:** 当前基准测试偏重于数学和科学领域，缺乏对动态环境交互和社会文化推理的充分评估，可能限制结果的全面性；此外，RL 方法在训练效率和稳定性上仍面临稀疏奖励和计算开销问题。
*   **对比分析:** 相较于 SFT 方法，RL 基方法在泛化能力和避免灾难性遗忘方面表现更优，但训练复杂度和资源需求较高。

## Further Thoughts

论文中的跨模态奖励设计（如 MetaSpatial 的多层次奖励）启发了我思考如何构建自适应奖励机制，利用生成式模型动态调整奖励分布以适应不同任务和模态特性；此外，冷启动和课程学习策略（如 Vision-R1 的 PTST）可能适用于其他领域，如机器人学习或资源受限环境下的模型部署；另一个值得探索的方向是将 RL 的优化能力与神经符号方法结合，提升多模态推理的解释性和泛化能力。