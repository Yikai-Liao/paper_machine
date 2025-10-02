---
title: "Hybrid Reward Normalization for Process-supervised Non-verifiable Agentic Tasks"
pubDatetime: 2025-09-29T23:44:55+00:00
slug: "2025-09-ppr-reward-normalization"
type: "arxiv"
id: "2509.25598"
score: 0.6431114322855073
author: "grok-3-latest"
authors: ["Peiran Xu", "Zhuohao Li", "Xiaoying Xing", "Guannan Zhang", "Debiao Li", "Kunyu Shi"]
tags: ["LLM", "Process Reward", "Outcome Reward", "Reinforcement Learning", "Agentic Tasks"]
institution: ["Alibaba Group (Accio Team)", "University of California, Los Angeles (UCLA)"]
description: "本文提出 Principle Process Reward (PPR) 框架，通过原则驱动的过程奖励模型和奖励归一化策略，显著提升了大型语言模型在非可验证代理任务中的性能和训练稳定性。"
---

> **Summary:** 本文提出 Principle Process Reward (PPR) 框架，通过原则驱动的过程奖励模型和奖励归一化策略，显著提升了大型语言模型在非可验证代理任务中的性能和训练稳定性。 

> **Keywords:** LLM, Process Reward, Outcome Reward, Reinforcement Learning, Agentic Tasks

**Authors:** Peiran Xu, Zhuohao Li, Xiaoying Xing, Guannan Zhang, Debiao Li, Kunyu Shi

**Institution(s):** Alibaba Group (Accio Team), University of California, Los Angeles (UCLA)


## Problem Background

大型语言模型（LLMs）在复杂代理任务中依赖外部工具（如搜索引擎）进行推理和知识获取，但现有强化学习方法主要基于结果奖励（Outcome Rewards），这种奖励信号稀疏且延迟，尤其在长轨迹任务中难以有效分配信用；过程奖励（Process Rewards）虽能提供细粒度监督，但在非可验证任务中因缺乏‘黄金答案’而难以标注和应用，且优化过程奖励可能与最终结果不一致，导致训练不稳定。

## Method

* **核心思想：** 提出 Principle Process Reward (PPR) 框架，通过结合原则驱动的过程奖励和结果验证，解决非可验证代理任务中的奖励设计难题。
* **具体实现：**
  - **Principle-based Process Reward Model (PPRM)：** 基于一组预定义原则（如正确性、相关性、一致性），对中间步骤进行上下文自适应的评估，生成可解释的评分；PPRM 在 Qwen3-8B 模型上进行监督微调，使用约 2k 条多轮搜索轨迹数据，确保评估的透明性和可靠性。
  - **Reward Normalization (ReNorm)：** 提出一种奖励归一化策略，通过广播结果奖励到每个步骤并中心化奖励信号，统一过程奖励和结果奖励的量级，保持符号一致性（即最终答案错误时过程奖励为非正值，正确时为非负值），从而稳定训练并防止奖励欺骗（Reward Hacking）。
  - **优化方法：** 使用 Proximal Policy Optimization (PPO) 算法和 Generalized Advantage Estimation (GAE) 进行优化，在 token 级别上应用剪切代理目标，确保模型在长轨迹任务中的稳定性。
* **关键点：** PPR 不依赖大规模人工标注，而是通过原则驱动和自动评估实现非可验证步骤的监督，同时通过 ReNorm 平衡局部步骤质量与全局任务成功的关系。

## Experiment

* **有效性：** PPR 在 General QA 和 Multi-Hop QA 基准数据集（如 NQ, HotpotQA, TriviaQA 等）上显著优于基线方法，基于 3B 和 7B 模型分别实现 28% 和 24% 的平均相对提升（对比非 RL 基线），以及 15% 和 5% 的提升（对比结果奖励 RL 基线）。
* **全面性：** 实验设置覆盖不同模型规模（Qwen2.5-3B 和 7B）、初始化类型（Base 和 Instruct）以及领域内（ID）和领域外（OOD）数据，验证了方法的鲁棒性和泛化能力；此外，构建了 NVProcessBench 基准，用于评估非可验证过程奖励模型的性能，PPRM 达到 0.613 的准确率，优于其他基线。
* **稳定性：** 训练奖励曲线显示 PPR 在长轨迹优化中保持稳定增长，而其他基线（如 Search-R1, Qwen3-8B）出现性能下降或崩溃；ReNorm 策略在奖励设计消融实验中表现最佳，显著优于无归一化或简单缩放方法。
* **开销：** 主要额外开销在于 PPRM 的训练和推理，但由于其基于较小模型（8B），对整体计算成本影响有限。

## Further Thoughts

PPR 的原则驱动设计启发我们是否可以通过元学习或自适应方法自动生成任务特定的评估原则，而不仅仅依赖预定义原则；ReNorm 的奖励平衡策略可能扩展到多目标优化场景，用于协调多个冲突奖励信号；此外，是否可以引入人类反馈（Human-in-the-Loop）机制，动态调整原则和奖励权重，以适应更复杂的非可验证任务？