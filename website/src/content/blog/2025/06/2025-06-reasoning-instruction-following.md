---
title: "Incentivizing Reasoning for Advanced Instruction-Following of Large Language Models"
pubDatetime: 2025-06-02T08:11:44+00:00
slug: "2025-06-reasoning-instruction-following"
type: "arxiv"
id: "2506.01413"
score: 0.8400826997995752
author: "grok-3-latest"
authors: ["Yulei Qin", "Gang Li", "Zongyi Li", "Zihan Xu", "Yuchen Shi", "Zhekai Lin", "Xiao Cui", "Ke Li", "Xing Sun"]
tags: ["LLM", "Reasoning", "Reinforcement Learning", "Instruction Following", "Test Time Scaling"]
institution: ["Tencent YouTu Lab", "Xiamen University", "The Chinese University of Hong Kong"]
description: "本文提出了一种通过自进化指令合成和强化学习驱动推理激励的系统性方法，显著提升了大型语言模型在复杂指令跟随任务上的表现，尤其在小模型上实现了性能飞跃。"
---

> **Summary:** 本文提出了一种通过自进化指令合成和强化学习驱动推理激励的系统性方法，显著提升了大型语言模型在复杂指令跟随任务上的表现，尤其在小模型上实现了性能飞跃。 

> **Keywords:** LLM, Reasoning, Reinforcement Learning, Instruction Following, Test Time Scaling

**Authors:** Yulei Qin, Gang Li, Zongyi Li, Zihan Xu, Yuchen Shi, Zhekai Lin, Xiao Cui, Ke Li, Xing Sun

**Institution(s):** Tencent YouTu Lab, Xiamen University, The Chinese University of Hong Kong


## Problem Background

大型语言模型（LLMs）在处理复杂指令时面临挑战，尤其是在指令包含多重约束（如并行、链式、分支结构）时表现不佳。
传统链式思维（Chain-of-Thought, CoT）方法由于推理过程流于表面，仅对指令进行简单复述，而未能深入分析约束之间的层次关系，导致性能提升有限甚至下降。
因此，研究的核心问题是激励 LLMs 进行深层推理，以提升其复杂指令跟随能力。

## Method

*   **核心思想:** 通过测试时计算扩展（test-time compute scaling），激励 LLMs 进行深层推理，解决复杂指令跟随中的浅层推理问题。
*   **自进化指令合成:** 基于现有约束分类法，设计可重复的数据获取方法，从种子指令（如 WildChat, Alpaca）出发，生成多样化的复杂指令数据集，涵盖不同类型（如 And, Chain, Selection）和组合方式，并通过代码执行和 LLM 评判进行验证。
*   **强化学习（RL）驱动推理:** 采用 Group Relative Policy Optimization (GRPO) 算法，通过规则中心奖励（rule-centric reward）优化 CoT 推理过程。奖励设计包括格式奖励（format reward，检查推理和答案标签）和准确性奖励（accuracy reward，验证约束满足情况），以指导模型生成结构化、深层推理。
*   **样本级对比与行为克隆:** 通过经验回放缓冲区（experience replay buffer）实现样本级对比（superior CoT enforcement），过滤掉浅层推理样本，确保推理对最终答案有益；同时通过行为克隆（behavior cloning）模仿专家响应，控制策略分布偏移，防止语义退化或奖励黑客问题。
*   **关键特点:** 不依赖任务特定微调或模板，注重通用推理能力培养，同时结合数学任务数据（如 DeepScaleR）补充推理训练。

## Experiment

*   **有效性:** 实验在七个综合性基准（如 IFEval, ComplexBench）上验证了方法的有效性。以 1.5B 参数的 Qwen2.5 模型为例，性能提升了 11.74%，接近 8B 模型表现，表明小模型通过推理激励可显著提升复杂指令跟随能力。
*   **对比优势:** 相较于传统 CoT 提示、监督微调（SFT）等基线，提出的方法在复杂指令（如 Chain, Selection 类型）上表现更优，尤其在需要深层结构分析的场景中，推理深度和广度得到显著改善。
*   **实验设置合理性:** 实验覆盖多种模型家族（Qwen, DeepSeek, LLaMA, Ministral）和规模（1.5B 到 8B），包括冷启动和热启动场景，设置全面且具代表性。但某些模型（如 LLaMA3.1-8B）在训练中出现崩溃，可能与预训练知识和 RL 目标冲突有关。
*   **局限性与开销:** 奖励模型（Qwen2.5-7B-Instruct）准确性略低于更大模型，可能引入噪声；数学任务与复杂指令任务的训练步长分配未完全优化，可能影响收敛；训练和推理时计算开销增加，主要来自 RL 优化和多样本生成。

## Further Thoughts

规则中心奖励设计是一个亮点，通过将复杂任务分解为可验证的原子约束，并设计分段奖励函数，可以有效引导模型关注关键目标，这种思路可扩展到其他领域（如代码生成或多模态任务），以确保输出符合多维度约束；此外，测试时计算扩展的理念启发我们思考如何在资源受限场景下，通过动态调整计算分配（如推理时增加 CoT 长度）来提升小模型性能，探索小模型与大模型性能差距的根本原因。