---
title: "Chain-in-Tree: Back to Sequential Reasoning in LLM Tree Search"
pubDatetime: 2025-09-30T06:18:44+00:00
slug: "2025-09-chain-in-tree-search"
type: "arxiv"
id: "2509.25835"
score: 0.613703059332078
author: "grok-3-latest"
authors: ["Xinzhe Li"]
tags: ["LLM", "Tree Search", "Test Time Scaling", "Reasoning", "Efficiency"]
institution: ["Independent Researcher (partially conducted at Deakin University)"]
description: "本文提出 Chain-in-Tree (CiT) 框架，通过自适应分支减少 LLM 树搜索中的不必要计算开销，在保持性能的同时显著降低推理成本。"
---

> **Summary:** 本文提出 Chain-in-Tree (CiT) 框架，通过自适应分支减少 LLM 树搜索中的不必要计算开销，在保持性能的同时显著降低推理成本。 

> **Keywords:** LLM, Tree Search, Test Time Scaling, Reasoning, Efficiency

**Authors:** Xinzhe Li

**Institution(s):** Independent Researcher (partially conducted at Deakin University)


## Problem Background

大型语言模型（LLMs）在测试时扩展计算（test-time scaling）可以提升长距离推理任务的性能，树搜索方法在此场景下取得了最先进的结果，但其效率极低，通常比简单迭代方法慢10-20倍。
主要问题在于现有树搜索框架在每个推理步骤都强制分支（branching），即使某些步骤显而易见或不需要探索，导致不必要的计算开销（如过多的模型调用）。
论文旨在通过自适应地决定何时分支，减少不必要的扩展，从而降低推理成本，同时保持搜索框架的性能。

## Method

*   **核心思想：** 提出 Chain-in-Tree (CiT)，一个插件框架，用于 LLM-in-the-loop 树搜索（LITS），通过自适应地决定何时分支来减少不必要的计算开销。
*   **链式阶段（Chaining Phase）：** 在树扩展前插入链式阶段，若当前步骤被认为自信或常规，则不分支，而是将多个节点线性连接成链，直到遇到不确定点才触发分支，减少不必要的模型调用。
*   **分支必要性评估（Branching Necessity, BN）：** 使用两种轻量级方法判断是否需要分支：
    *   **BN-DP（Direct Prompting）：** 利用辅助 LLM 直接评估当前状态和动作是否需要分支，输出评分（1-4）决定是否继续链式连接，评分高表示步骤逻辑上不可避免，应继续链式。
    *   **BN-SC（Self-Consistency）：** 通过政策模型生成多个候选动作，然后聚类这些动作以评估一致性；若大多数动作属于最大簇（一致性高），则继续链式，否则触发分支。BN-SC 有两种实现：
        - **BN-SC [1]（Aggregator-based）：** 使用辅助 LLM 聚合器将候选动作聚类为等价类，计算最大簇占比作为 BN 评分。
        - **BN-SC [2]（Pairwise-Equivalence）：** 使用辅助 LLM 作为二元判定器，成对比较候选动作的语义等价性，通过并查集式合并形成簇，计算最大簇占比。
*   **兼容性与复用：** CiT 设计为插件，可集成到现有 LITS 框架（如 ToT-BS、ReST-MCTS、RAP）中；链式阶段生成的子节点在扩展时可复用，避免重复调用政策模型。
*   **理论保证：** 提供理论分析，证明 BN-DP 在最坏情况下不会增加政策模型调用次数，且在易节点（easy nodes）存在时严格减少调用。

## Experiment

*   **有效性：** 在 GSM8K 和 Math500 数据集上，BN-DP 一致地将运行时间、令牌生成和模型调用次数减少了75-85%，准确率损失极小（有时甚至略有提升），验证了理论效率保证。
*   **BN-SC 的表现：** BN-SC 在大多数设置下节省高达80%的成本，但在14个设置中有1-4个表现出不稳定性，尤其在使用较弱 LLM（如 LLaMA3-8B）作为 BN 评估器时，失败由少数极端困难实例驱动，导致过长推理步骤。
*   **评估器质量影响：** BN 评估器质量对 CiT 效果至关重要；使用强模型（如 Qwen3-32B）时，性能接近或达到基线 LITS 框架；使用弱模型时，准确率显著下降，接近简单 CoT 基线。
*   **实验设置：** 实验覆盖了两种数据集（GSM8K、Math500）、三种 LITS 框架（ToT-BS、ReST-MCTS、RAP）、两种基础 LLM（Qwen3-32B、LLaMA3-8B）及14种配置，评估指标包括令牌数、调用次数、运行时间和准确率，设置全面合理。
*   **局限性：** 实验聚焦数学推理，未涉及确定性动作空间（如棋盘游戏），对准确率提升原因的分析也待深入。

## Further Thoughts

CiT 的自适应分支思想可推广至其他搜索范式（如 A* 或启发式搜索）及非 LLM 规划问题；BN-SC 在确定性动作空间中可完全依赖程序化规则而非 LLM，为降低成本和提高稳定性提供新思路；BN 评估器质量的影响提示未来可探索混合评估机制（如规则与模型结合）；CiT 在某些设置下的准确率提升值得研究，或揭示链式推理对推理质量的潜在正向作用。