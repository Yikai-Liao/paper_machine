---
title: "SFR-DeepResearch: Towards Effective Reinforcement Learning for Autonomously Reasoning Single Agents"
pubDatetime: 2025-09-08T02:07:09+00:00
slug: "2025-09-deepresearch-single-agent"
type: "arxiv"
id: "2509.06283"
score: 0.6386767072672638
author: "grok-3-latest"
authors: ["Xuan-Phi Nguyen", "Shrey Pandit", "Revanth Gangi Reddy", "Austin Xu", "Silvio Savarese", "Caiming Xiong", "Shafiq Joty"]
tags: ["LLM", "Reinforcement Learning", "Single Agent", "Tool Use", "Reasoning"]
institution: ["Salesforce AI Research"]
description: "本文提出了一种基于强化学习的训练框架，将推理优化的 LLMs 转化为自主单智能体系统，通过智能体推理管道和长度归一化 RL 策略，在 Deep Research 任务上取得显著性能提升，SFR-DR-20B 模型在 HLE 基准上达到 28.7%。"
---

> **Summary:** 本文提出了一种基于强化学习的训练框架，将推理优化的 LLMs 转化为自主单智能体系统，通过智能体推理管道和长度归一化 RL 策略，在 Deep Research 任务上取得显著性能提升，SFR-DR-20B 模型在 HLE 基准上达到 28.7%。 

> **Keywords:** LLM, Reinforcement Learning, Single Agent, Tool Use, Reasoning

**Authors:** Xuan-Phi Nguyen, Shrey Pandit, Revanth Gangi Reddy, Austin Xu, Silvio Savarese, Caiming Xiong, Shafiq Joty

**Institution(s):** Salesforce AI Research


## Problem Background

本文聚焦于 Deep Research (DR) 领域，旨在开发自主单智能体（Autonomous Single-Agent）系统，以处理需要广泛搜索和复杂推理的任务。
当前大型语言模型（LLMs）在工具使用和长距离任务规划方面的能力不足，尤其是在需要灵活自主决策的场景下，相比多智能体系统，单智能体因其泛化能力和灵活性被认为更具潜力，但如何在不依赖预定义工作流的情况下提升其智能体能力是一个关键挑战。
论文的目标是通过强化学习（RL）进一步增强推理优化模型的自主决策能力，同时保留其原有的推理能力。

## Method

*   **核心思想:** 提出一种基于强化学习的训练框架，针对推理优化的 LLMs，增强其作为自主单智能体的能力，用于 Deep Research 任务。
*   **智能体推理管道 (Agentic Inference Pipeline):** 设计了一个模仿多轮对话的推理框架，将多轮工具调用交互重构为单轮上下文问答任务（针对某些模型家族如 QwQ 和 Qwen），以适应其单步推理的训练分布；同时引入内存自管理机制，通过一个内存清理工具（clean_memory）让智能体自主管理上下文窗口，避免信息冗余或丢失，支持几乎无限的上下文长度。
*   **工具集设计:** 提供最小的工具集，包括网络搜索（search_internet）、网页浏览（browse_page）和 Python 解释器（code_interpreter），工具功能被刻意限制（如静态网页浏览、无状态代码执行），以增加训练挑战性，促使智能体更高效地探索和使用工具。
*   **RL 训练配方:** 基于 REINFORCE 算法的改进版本，引入时间优势归一化（Temporal Advantage Normalization）以平衡不同长度轨迹对训练损失的影响，避免长轨迹的过度强化；采用轨迹过滤（Trajectory Filtering）策略，剔除无效轨迹并控制正负样本比例，稳定训练过程；此外，使用部分轨迹重用（Partial Rollouts）和定制奖励模型（Reward Modeling）以提升长距离任务的训练效果。
*   **数据合成:** 构建复杂的合成数据集，涵盖短篇问答（多跳推理）和长篇报告任务，挑战性高于现有开源数据集，确保训练数据对当前最先进的 DR 智能体也具有难度。
*   **关键点:** 从推理优化的模型出发，通过 RL 训练增强其自主决策能力，同时通过上下文管理和训练稳定策略避免模型退化。

## Experiment

*   **有效性:** 提出的 SFR-DR-20B 模型在 Humanity’s Last Exam (HLE) 基准上达到 28.7% 的性能，相比基线 gpt-oss-20b 提升约 65%，在 FRAMES 和 GAIA 基准上也显著优于同规模的开源单智能体和多智能体系统，显示出 RL 训练在工具使用和复杂推理任务上的显著提升。
*   **实验设置合理性:** 实验覆盖了多种模型规模（8B、32B、20B）和任务类型（多跳问答、通用助手任务、推理任务），在多个公开基准上进行测试；采用污染防控措施（如屏蔽敏感域名），增强结果可信度；同时通过消融研究分析了智能体工作流和长度归一化的影响，设置较为全面。
*   **局限性与分析:** 尽管性能提升明显，但与一些专有系统（如 OpenAI 的 Deep Research）相比仍有差距；工具调用次数和响应长度分析显示不同模型家族（如 gpt-oss 和 Qwen 系列）在训练后行为差异较大，可能影响一致性；此外，过多的工具调用并不总是带来性能提升，需依赖策略性执行。

## Further Thoughts

单智能体系统的泛化潜力为未来智能体设计提供了新思路，其不依赖预定义角色和流程的灵活性可能适用于更多未知任务场景；内存自管理机制通过让模型自主清理上下文窗口，解决了长距离任务中的上下文限制问题，这一方法或可推广至其他需要长上下文处理的领域；此外，长度归一化在 RL 训练中的应用有效避免了长轨迹对训练的过度影响，这一技术可能对其他多步决策任务（如游戏 AI 或机器人控制）具有借鉴意义。