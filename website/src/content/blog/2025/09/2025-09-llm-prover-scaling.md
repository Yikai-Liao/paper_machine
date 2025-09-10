---
title: "Scaling up Multi-Turn Off-Policy RL and Multi-Agent Tree Search for LLM Step-Provers"
pubDatetime: 2025-09-08T09:54:18+00:00
slug: "2025-09-llm-prover-scaling"
type: "arxiv"
id: "2509.06493"
score: 0.7131449427105805
author: "grok-3-latest"
authors: ["Ran Xin", "Zeyu Zheng", "Yanchen Nie", "Kun Yuan", "Xia Xiao"]
tags: ["LLM", "Reinforcement Learning", "Tree Search", "Reasoning", "Multi-Agent"]
institution: ["ByteDance Seed", "Carnegie Mellon University", "Peking University"]
description: "本文提出 `BFS-Prover-V2` 系统，通过训练时的多阶段专家迭代和推理时的规划者增强多智能体树搜索，显著提升了大型语言模型在形式化数学定理证明中的性能，并在 MiniF2F 和 ProofNet 基准上取得最先进的步级证明结果。"
---

> **Summary:** 本文提出 `BFS-Prover-V2` 系统，通过训练时的多阶段专家迭代和推理时的规划者增强多智能体树搜索，显著提升了大型语言模型在形式化数学定理证明中的性能，并在 MiniF2F 和 ProofNet 基准上取得最先进的步级证明结果。 

> **Keywords:** LLM, Reinforcement Learning, Tree Search, Reasoning, Multi-Agent

**Authors:** Ran Xin, Zeyu Zheng, Yanchen Nie, Kun Yuan, Xia Xiao

**Institution(s):** ByteDance Seed, Carnegie Mellon University, Peking University


## Problem Background

大型语言模型（LLMs）在自动化定理证明（Automated Theorem Proving, ATP）中展现出巨大潜力，但面临训练时性能提升瓶颈（performance plateaus）和推理时搜索空间复杂性（combinatorial search complexity）两大挑战。
本文旨在解决如何通过强化学习持续提升模型能力，以及如何设计高效推理架构以处理复杂多步推理任务的问题，尤其是在形式化数学领域。

## Method

*   **训练时方法：多阶段专家迭代（Multi-Stage Expert Iteration）**
    *   **核心思想**：基于离策强化学习（Off-Policy RL），通过专家迭代框架（inspired by AlphaZero）交替进行证明生成和模型精炼，持续提升模型性能。
    *   **自适应策略过滤（Adaptive Tactic Filtering）**：根据策略的困惑度（perplexity）分布，动态筛选训练数据，过滤掉过于简单（低困惑度）和过于复杂（高困惑度）的策略，仅保留适中数据，形成自动化课程学习，确保模型在能力边界上学习。
    *   **周期性重新训练（Periodic Retraining）**：当性能出现平台期时，通过数据重新合成（re-synthesis）和严格筛选（aggressive curation），从基础检查点重新训练模型，作为‘软重置’跳出局部最优，增强探索能力。
*   **推理时方法：规划者增强的多智能体树搜索（Planner-Enhanced Multi-Agent Tree Search）**
    *   **核心思想**：通过分层推理架构，将复杂定理分解为可管理的子目标（subgoals），减少搜索空间，提升推理效率。
    *   **规划者-证明者范式（Planner-Prover Paradigm）**：高层次规划者（Planner）负责分解定理为子目标序列，低层次证明者（Prover）逐一解决子目标，模拟人类数学家的高层次规划与低层次执行分离。
    *   **多智能体协作与共享缓存**：多个证明者并行处理同一子目标，通过共享子目标缓存（Shared Subgoal Cache）记录状态和结果，避免重复计算，提升效率。
    *   **动态重新规划（Dynamic Replanning）**：当证明者在子目标上卡住时，规划者根据当前上下文重新生成更细化的计划，增强系统鲁棒性。

## Experiment

*   **有效性**：`BFS-Prover-V2` 在 MiniF2F 测试集上达到 95.08% 准确率，在 ProofNet 测试集上达到 41.4%，显著优于前代步级证明者（如 InternLM2.5-StepProver-7B 的 65.9% 和 BFS-Prover-V1 的 70.8% 在 MiniF2F 上），尤其在 ProofNet 上展现了从高中竞赛到本科级问题的泛化能力。
*   **训练时提升**：多阶段专家迭代和周期性重新训练有效突破性能平台期，长期训练中性能持续提升（如图3所示，每次重新训练后性能显著跳跃）。
*   **推理时效率**：规划者-证明者架构显著减少搜索复杂度，动态重新规划机制在卡住时有效调整策略（如附录中 IMO 问题案例，从 7200 次尝试失败到 800 次成功）。
*   **实验设置合理性**：基于 Qwen2.5-Math-7B 和 Qwen2.5-32B 模型，训练数据规模达 300 万条形式化语句，涵盖高中到本科数学问题，数据和模型选择具有代表性；推理配置（如 BFS 参数和规划者提示）详细说明，结果可信。
*   **局限性**：步级证明生成的证明简洁但可读性较差，相比整体证明方法对人类理解不够友好。

## Further Thoughts

自适应数据过滤机制可推广至其他需要课程学习的领域，如自然语言推理或代码生成，通过动态调整数据难度提升学习效率；
分层推理架构（规划者-证明者范式）模拟人类高层次规划与低层次执行分离的工作方式，可应用于多步决策或长程规划任务；
动态重新规划机制展示了根据上下文调整策略的重要性，启发在其他 AI 系统中引入类似反馈循环以提升复杂任务适应性。