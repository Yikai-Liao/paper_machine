---
title: "An Empirical Study on Strong-Weak Model Collaboration for Repo-level Code Generation"
pubDatetime: 2025-05-26T16:25:38+00:00
slug: "2025-05-strong-weak-collaboration"
type: "arxiv"
id: "2505.20182"
score: 0.649248249704722
author: "grok-3-latest"
authors: ["Shubham Gandhi", "Atharva Naik", "Yiqing Xie", "Carolyn Rose"]
tags: ["LLM", "Code Generation", "Cost Efficiency", "Model Collaboration", "Pipeline Strategy", "Dynamic Routing"]
institution: ["Carnegie Mellon University"]
description: "本文通过系统评估强弱模型协作策略，为仓库级代码生成任务提供了成本高效的解决方案，并在不同预算与性能约束下给出了实用指导原则。"
---

> **Summary:** 本文通过系统评估强弱模型协作策略，为仓库级代码生成任务提供了成本高效的解决方案，并在不同预算与性能约束下给出了实用指导原则。 

> **Keywords:** LLM, Code Generation, Cost Efficiency, Model Collaboration, Pipeline Strategy, Dynamic Routing

**Authors:** Shubham Gandhi, Atharva Naik, Yiqing Xie, Carolyn Rose

**Institution(s):** Carnegie Mellon University


## Problem Background

大型语言模型（LLMs）在复杂任务如仓库级代码生成中表现出色，但其高昂的调用成本限制了规模化应用，尤其是在需要多次调用强模型时（例如 SWE-Agent 每次运行成本高达 4 美元）。
论文聚焦于如何设计成本高效且有效的 LLM 系统，解决任务复杂性差异带来的挑战：弱模型难以处理复杂任务，而过度依赖强模型会抵消成本优势，特别是在 GitHub 问题解决等实际场景中。

## Method

*   **核心思想：** 通过强弱模型协作，在仓库级代码生成任务中平衡性能与成本，利用弱模型处理简单任务，将复杂任务委托给强模型。
*   **具体策略分类：**
    *   **静态上下文增强（Static Context Augmentation）：** 利用强模型生成背景信息或实例特定上下文，供弱模型使用以降低成本。包括仓库摘要（Repository Summary）、仓库级 FAQ、规划（Planning）和少样本示例（Few-Shot Examples）等方法，旨在通过强模型提供高质量上下文，减少弱模型的生成负担。
    *   **管道划分（Pipeline Division）：** 通过硬编码顺序调用强弱模型，分为‘强模型优先’（Strong LM First，强模型首先尝试，弱模型迭代优化）、‘弱模型优先’（Weak LM First，弱模型先尝试，失败后调用强模型）和‘提示减少’（Prompt Reduction，弱模型先过滤无关上下文，减少强模型输入规模）三种方法，旨在通过明确分工优化资源分配。
    *   **动态协作（Dynamic Collaboration）：** 在推理时通过路由器（Router）动态判断任务复杂性，决定调用强模型还是弱模型，分为弱路由器（Weak Router）和强路由器（Strong Router），以实现灵活的任务分配。
*   **实现框架：** 基于 Agentless-Lite 框架，结合检索增强生成（RAG），首先检索仓库中相关文档，然后应用上述协作策略进行代码生成。
*   **对比基线：** 包括仅使用弱模型的自一致性（Self-Consistency）方法，通过多次采样并选择最一致输出提升性能，作为成本等价的对比。

## Experiment

*   **有效性：** 最优策略‘强模型优先’（Strong LM First）在多个强弱模型对中表现出色，例如在 O3-mini 与 Qwen2.5-Coder-32B 组合中，分辨率（Resolution Rate）接近强模型水平，同时成本降低约 40%。弱模型性能通过协作提升显著（例如提升约 62%）。
*   **策略对比：** 管道划分和静态上下文增强策略在平均成本效率上优于动态协作和弱模型基线（如自一致性），其中‘弱模型优先’和‘弱路由器’在低预算下表现最佳，而‘强模型优先’在较高预算下接近强模型性能。
*   **实验设置：** 实验在 SWE-Bench Lite 数据集（包含 300 个 GitHub 问题）上进行，覆盖多种强弱模型对（如 O3-mini、O4-mini 与 GPT-4o-mini、Qwen2.5-Coder 系列），评估指标包括分辨率和成本，并通过性能-成本曲线分析不同预算下的最佳策略。设置合理，任务贴近实际应用。
*   **局限性：** 实验局限于 Agentless-Lite 框架，未涉及训练或微调方法，成本评估仅基于 API 调用，未考虑延迟或能耗，可能影响结果的全面性。

## Further Thoughts

强弱模型协作的灵活性启发我在其他领域（如文本生成或多模态任务）探索类似的多模型协作机制，通过动态路由或管道策略优化资源分配；‘弱路由器优于强路由器’的反直觉发现提示我们，强模型可能在元任务上‘过度思考’，未来可以设计专门的轻量级路由模型来提升效率；‘提示减少’策略通过预处理优化强模型注意力分配，启发我们可以在输入阶段引入类似过滤机制，进一步降低成本。