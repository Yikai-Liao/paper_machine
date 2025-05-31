---
title: "3DLLM-Mem: Long-Term Spatial-Temporal Memory for Embodied 3D Large Language Model"
pubDatetime: 2025-05-28T17:59:13+00:00
slug: "2025-05-3dllm-memory-fusion"
type: "arxiv"
id: "2505.22657"
score: 0.6761114340124864
author: "grok-3-latest"
authors: ["未明确列出，推测与 Yining Hong, Chuang Gan 等相关团队"]
tags: ["LLM", "3D Representation", "Memory System", "Spatial Reasoning", "Embodied AI"]
institution: ["未明确列出，推测与 University of California, Los Angeles (UCLA) 相关"]
description: "本文提出 3DLLM-Mem 模型，通过双重记忆系统和记忆融合机制显著提升 3D 具身语言模型的长期空间-时间推理能力，并在新基准 3DMem-Bench 上验证了其优越性。"
---

> **Summary:** 本文提出 3DLLM-Mem 模型，通过双重记忆系统和记忆融合机制显著提升 3D 具身语言模型的长期空间-时间推理能力，并在新基准 3DMem-Bench 上验证了其优越性。 

> **Keywords:** LLM, 3D Representation, Memory System, Spatial Reasoning, Embodied AI

**Authors:** 未明确列出，推测与 Yining Hong, Chuang Gan 等相关团队

**Institution(s):** 未明确列出，推测与 University of California, Los Angeles (UCLA) 相关


## Problem Background

当前 3D 大型语言模型（3DLLMs）在复杂长时任务中面临挑战，尤其是在需要跨多房间、长时间跨度的空间-时间记忆能力方面表现不足。
关键问题包括：难以维持长期记忆链、无法有效存储和检索密集的 3D 空间信息，以及在动态环境中跟踪物体位置和状态变化的能力有限，导致模型在具身环境中执行需要长期规划和空间推理的任务时效果不佳。

## Method

*   **核心思想:** 提出 3DLLM-Mem 模型，通过模仿人类认知结构设计双重记忆系统（工作记忆与情景记忆），并引入记忆融合模块，提升模型在具身 3D 环境中的长期空间-时间推理、规划和行动能力。
*   **具体实现:** 
    *   **双重记忆系统:** 工作记忆（Working Memory）存储当前观察，容量有限；情景记忆（Episodic Memory）以密集 3D 表示形式存储过去的空间-时间信息，并通过时间位置编码增强时间理解。
    *   **记忆融合模块:** 使用查询-键-值机制，将工作记忆中的当前观察编码为查询特征，从情景记忆库中检索相关信息并融合，生成最终的记忆增强表示，确保任务相关的长期信息被有效整合。
    *   **动态记忆更新:** 工作记忆随环境交互实时更新，当代理进入新环境时，旧的工作记忆转移到情景记忆库，并根据环境变化动态调整记忆内容，保持记忆的一致性和最新性。
*   **关键优势:** 利用密集 3D 表示保留空间细节，通过选择性融合降低计算负担，同时支持长时任务中的动态记忆管理。

## Experiment

*   **有效性:** 在新基准 3DMem-Bench 上，3DLLM-Mem 在具身任务中平均成功率（Success Rate）为域内 37.6%、野外 32.1%，比最强基线高出约 16.5%；在空间-时间问答（EQA）和场景描述（Captioning）任务中也显著优于其他方法。
*   **优越性:** 相比其他记忆管理策略（如‘Everything in Context’、‘Most Recent Memory’和‘Retrieval-Augmented Memory’），3DLLM-Mem 在任务复杂度增加时表现更稳定，尤其在困难的野外任务中成功率仍达 27.8%，而基线仅约 5%。
*   **实验设置合理性:** 基准包含多房间 3D 场景和超过 26,000 个轨迹，任务难度从简单到困难，涵盖多种任务类型（具身任务、EQA、Captioning），并通过域内和野外任务评估泛化能力；数据质量通过自动验证和人工检查得到保障，但未涉及低级导航和控制策略的集成。
*   **计算开销:** 训练时间较短（约 1 天），能耗较低，但密集 3D 表示和记忆融合可能增加推理时计算负担，需进一步分析。

## Further Thoughts

双重记忆系统的仿生设计启发我们可以在其他领域（如视频理解或多轮对话）中应用分层记忆结构；记忆融合的查询机制提示任务驱动的记忆选择可能通过强化学习进一步优化；密集 3D 表示的应用为未来的机器人导航和虚拟现实任务提供了新思路。