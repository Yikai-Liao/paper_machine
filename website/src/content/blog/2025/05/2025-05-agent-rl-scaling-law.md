---
title: "Agent RL Scaling Law: Agent RL with Spontaneous Code Execution for Mathematical Problem Solving"
pubDatetime: 2025-05-12T17:23:34+00:00
slug: "2025-05-agent-rl-scaling-law"
type: "arxiv"
id: "2505.07773"
score: 0.6948438999502478
author: "grok-3-latest"
authors: ["Xinji Mai", "Haotian Xu", "Xing W", "Weinong Wang", "Yingying Zhang", "Wenqiang Zhang"]
tags: ["LLM", "Reinforcement Learning", "Tool Integration", "Mathematical Reasoning", "Scaling Law"]
institution: ["Fudan University", "Xiaohongshu", "East China Normal University"]
description: "本文通过ZeroTIR框架，揭示了Agent RL Scaling Law，验证了基础LLM可通过强化学习自发学习代码执行工具，显著提升数学推理能力。"
---

> **Summary:** 本文通过ZeroTIR框架，揭示了Agent RL Scaling Law，验证了基础LLM可通过强化学习自发学习代码执行工具，显著提升数学推理能力。 

> **Keywords:** LLM, Reinforcement Learning, Tool Integration, Mathematical Reasoning, Scaling Law

**Authors:** Xinji Mai, Haotian Xu, Xing W, Weinong Wang, Yingying Zhang, Wenqiang Zhang

**Institution(s):** Fudan University, Xiaohongshu, East China Normal University


## Problem Background

大型语言模型（LLMs）在数学推理任务中面临挑战，尤其是在需要精确计算和多步推理的场景下，模型常因依赖概率预测而非计算正确性导致结果不可靠。
现有方法如监督微调（SFT）依赖高质量数据，易过拟合且缺乏泛化能力；工具集成推理（TIR）通常需要预定义提示或触发机制，缺乏自主性。
本文旨在解决如何通过强化学习（RL），让基础LLM在无监督工具使用示例的情况下，自发学习利用外部工具（如Python代码执行）来提升数学问题解决能力。

## Method

*   **核心框架：ZeroTIR（Zero Tool-Integrated Reasoning）**：通过基于结果的强化学习（ZeroRL），从基础LLM开始训练，使其自发生成并执行Python代码以解决数学问题。
*   **强化学习算法**：采用策略梯度方法，包括PPO（Proximal Policy Optimization）和Reinforce++变体。PPO通过优化裁剪代理目标函数并结合价值网络减少方差，Reinforce++则直接基于采样轨迹估计策略梯度，两种方法均通过最大化任务奖励并用KL散度正则化参考策略来确保训练稳定性。
*   **交互机制**：设计高效的代码执行环境交互方式，利用动态停止标记（如‘python）实现推理、代码生成、执行和反馈的迭代循环，避免生成完整序列后再解析的低效方式，同时通过最大调用次数（N_max）控制计算资源消耗。
*   **训练优化技术**：引入回放缓冲区过滤机制，筛选准确率在中间范围的样本以增强学习稳定性；采用异步流水线交互框架，提升训练吞吐量（比同步交互快4倍）。
*   **关键创新**：完全从基础模型开始训练，强调自发工具使用，而非依赖预训练工具轨迹或特定提示结构，探索自主工具学习的Scaling Law。

## Experiment

*   **有效性**：ZeroTIR训练的模型（ZTRL）在数学推理基准（如AIME24/25, MATH500）上表现优异，7B参数模型平均准确率达52.3%，显著优于无工具ZeroRL基线（39.1%）、SFT-TIR方法（41.6%）以及类似工作TORL（51.8%）。
*   **Scaling Law验证**：实验发现Agent RL Scaling Law，随着训练步数增加，代码使用频率、响应长度和任务准确率呈正相关；模型规模（1.5B到32B）和工具交互上限（N_max）提升进一步增强性能。
*   **实验设置合理性**：实验覆盖多种模型规模（1.5B, 7B, 32B）、RL算法（PPO vs Reinforce++）、数据集（ORZ-57k vs DeepMath）及参数调节（如交互上限、解码熵），设置全面且对比充分；训练动态分析揭示工具使用策略的演变（如倾向于单次高效代码调用）。
*   **局限性**：由于资源限制，未能对Scaling Law进行精确数学形式量化；部分实验（如高交互上限）受限于计算资源，未完全探索。

## Further Thoughts

Agent RL Scaling Law的发现启发了我思考工具使用学习是否具有通用规律，是否可以扩展到其他工具（如搜索、计算器）或多任务场景，让模型自发选择最合适的工具组合。
此外，模型倾向于少量高效代码调用的现象提示我们可以在训练初期限制交互次数以提升效率，后期动态调整上限探索复杂策略；还可以探索如何通过多任务RL训练，让模型在不同领域（如数学、编程、常识推理）中自适应工具选择。