---
title: "DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition"
pubDatetime: 2025-05-01T15:49:05Z
slug: "2025-04-deepseek-prover-v2"
type: "arxiv"
id: "2504.21801"
score: 0.6595004968925717
author: "grok-3-latest"
authors: ["Z.Z. Ren", "Zhihong Shao", "Junxiao Song", "Huajian Xin", "Haocheng Wang", "Wanjia Zhao", "Liyue Zhang", "Zhe Fu", "Qihao Zhu", "Dejian Yang", "Z.F. Wu", "Zhibin Gou", "Shirong Ma", "Hongxuan Tang", "Yuxuan Liu", "Wenjun Gao", "Daya Guo", "Chong Ruan"]
tags: ["LLM", "Formal Reasoning", "Subgoal Decomposition", "Reinforcement Learning", "Curriculum Learning"]
institution: ["DeepSeek-AI"]
description: "本文通过子目标分解和强化学习，开发了 DeepSeek-Prover-V2 模型，显著提升了大型语言模型在形式化数学推理中的性能，并在多个基准测试中达到最先进水平。"
---

> **Summary:** 本文通过子目标分解和强化学习，开发了 DeepSeek-Prover-V2 模型，显著提升了大型语言模型在形式化数学推理中的性能，并在多个基准测试中达到最先进水平。 

> **Keywords:** LLM, Formal Reasoning, Subgoal Decomposition, Reinforcement Learning, Curriculum Learning
> **Recommendation Score:** 0.6595004968925717

**Authors:** Z.Z. Ren, Zhihong Shao, Junxiao Song, Huajian Xin, Haocheng Wang, Wanjia Zhao, Liyue Zhang, Zhe Fu, Qihao Zhu, Dejian Yang, Z.F. Wu, Zhibin Gou, Shirong Ma, Hongxuan Tang, Yuxuan Liu, Wenjun Gao, Daya Guo, Chong Ruan
**Institution(s):** DeepSeek-AI

## Problem Background

大型语言模型（LLMs）在自然语言推理中表现出强大的能力，但由于形式化数学推理需要严格的逻辑结构和无歧义的步骤，LLMs 的非形式化推理方式难以直接应用于形式化定理证明，尤其是在 Lean 4 等证明助手中处理复杂定理时。本文旨在解决非形式化推理与形式化证明之间的差距，关键问题是如何有效分解复杂数学问题并生成可验证的形式化证明。

## Method

* **核心思想：** 通过子目标分解（Subgoal Decomposition）和强化学习（Reinforcement Learning, RL），将非形式化数学推理与形式化证明能力统一到单个模型中，提升复杂定理证明的效率和准确性。
* **具体实现步骤：**
  * **递归定理证明管道：** 利用强大的通用模型 DeepSeek-V3 进行初始问题分解，生成自然语言证明草图（Proof Sketches），并将其形式化为 Lean 4 中的定理语句，包含一系列子目标（使用 'sorry' 占位符表示未完成证明的部分）。
  * **子目标解决：** 使用参数较小的 7B 模型针对分解后的子目标进行证明搜索，降低计算成本。成功解决的子目标被整合为完整的形式化证明。
  * **冷启动数据合成（Cold-Start Data）：** 将 DeepSeek-V3 的链式思维（Chain-of-Thought, CoT）推理与完整形式化证明结合，生成高质量的合成训练数据，为后续训练提供起点。
  * **课程学习与专家迭代（Curriculum Learning & Expert Iteration）：** 通过子目标分解生成不同难度的训练任务，逐步提升模型能力；同时在每次迭代中使用当前最佳策略生成证明尝试，成功证明加入训练集，逐步优化模型。
  * **强化学习优化：** 在冷启动数据基础上，采用 Group Relative Policy Optimization (GRPO) 算法进行强化学习，使用二元正确/错误反馈作为奖励信号，并引入一致性奖励（Consistency Reward）确保证明结构与子目标分解对齐。
  * **两阶段训练模式：** 设计了高效非链式思维（Non-CoT）模式用于快速生成简洁证明，以及高精度链式思维（CoT）模式用于复杂推理任务，两种模式通过不同提示（Prompts）引导。
* **关键特点：** 不依赖单一模型完成所有任务，而是通过大模型（DeepSeek-V3）与小模型（7B）的协作降低计算开销；子目标分解模仿人类分步解决复杂问题的方式，提高可解释性和证明效率。

## Experiment

* **有效性：** DeepSeek-Prover-V2-671B 在多个基准测试中取得最先进性能，例如在 MiniF2F-test 上 Pass@8192 准确率达 88.9%，相比其他模型有显著提升；在 PutnamBench 上解决了 49/658 个问题，展现了大学水平数学问题的泛化能力；在新引入的 ProverBench 上，解决了 AIME 24&25 子集中的 6/15 个问题。
* **优越性：** 链式思维（CoT）模式相比非 CoT 模式在准确率上有明显提升，例如在 MiniF2F-test 上 CoT 模式 Pass@32 准确率为 82.4%，而非 CoT 模式为 75.6%；同时，子目标引导的课程学习框架使模型在验证集上的表现接近最终模型，证明了分解策略的有效性。
* **实验设置合理性：** 实验覆盖了从高中竞赛（MiniF2F, AIME）到大学水平（ProofNet, PutnamBench）的广泛数学问题，测试了模型在不同难度和领域上的泛化能力；同时对比了不同参数规模（7B vs 671B）和不同生成模式（CoT vs 非 CoT），设计全面。
* **开销：** CoT 模式生成 token 数量显著高于非 CoT 模式（例如 671B 模型在 MiniF2F-test 上 CoT 模式平均生成 6751.9 个 token，非 CoT 模式仅 761.8 个），表明性能提升伴随着更高的计算成本。

## Further Thoughts

子目标分解结合强化学习的策略非常具有启发性，不仅适用于形式化数学推理，还可能推广到其他复杂推理任务（如代码生成或逻辑推理），通过分层分解降低问题难度；此外，冷启动数据的合成方式（大模型提供非形式化推理，小模型完成形式化验证）提供了一种高效利用异构模型资源的方法，未来可以探索是否能结合不同领域模型（如自然语言与符号推理模型）进一步提升性能。