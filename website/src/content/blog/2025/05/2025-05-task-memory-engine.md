---
title: "Task Memory Engine: Spatial Memory for Robust Multi-Step LLM Agents"
pubDatetime: 2025-05-26T02:53:22+00:00
slug: "2025-05-task-memory-engine"
type: "arxiv"
id: "2505.19436"
score: 0.568393761826802
author: "grok-3-latest"
authors: ["Ye Ye"]
tags: ["LLM", "Spatial Memory", "Task Graph", "Intent Classification", "Reasoning"]
institution: ["New York University"]
description: "本文提出 Task Memory Engine (TME)，通过空间记忆框架和动态任务图（DAG）显著提升大型语言模型在多步骤交互任务中的鲁棒性，消除幻觉和误解，同时优化 token 效率。"
---

> **Summary:** 本文提出 Task Memory Engine (TME)，通过空间记忆框架和动态任务图（DAG）显著提升大型语言模型在多步骤交互任务中的鲁棒性，消除幻觉和误解，同时优化 token 效率。 

> **Keywords:** LLM, Spatial Memory, Task Graph, Intent Classification, Reasoning

**Authors:** Ye Ye

**Institution(s):** New York University


## Problem Background

大型语言模型（LLMs）在多步骤交互任务中表现不佳，经常出现幻觉、重复操作或误解用户修正的问题，原因在于其依赖线性、无结构的上下文，缺乏持久记忆机制来追踪动态目标和任务依赖，导致上下文过载和语义漂移。
论文的出发点是设计一种无需微调的解决方案，提升 LLMs 在复杂多轮交互中的鲁棒性和可靠性。

## Method

*   **核心思想:** 提出 Task Memory Engine (TME)，一个模块化的记忆控制器，将线性上下文替换为空间记忆框架，通过动态任务图（DAG）组织任务和依赖，提升多步骤推理能力，而无需对 LLM 进行微调。
*   **具体实现:**
    *   **Task Memory Structure (TMS):** 基于有向无环图（DAG）的任务记忆结构，初始为树形，后演变为 DAG，支持子任务共享和依赖追踪。每个节点存储任务状态、历史记录、父节点等信息，便于全局状态更新和修订管理。
    *   **Task Representation and Intent Management (TRIM):** 使用基于 LLM 的少样本提示（few-shot prompting）分解用户输入为子任务，分类意图为‘新任务’（new）、‘更新’（update）或‘查询’（check），并映射到 DAG 操作（如添加、替换节点）。TRIM 输出结构化 JSON 格式，确保意图准确对齐。
    *   **工作流程:** TME 分为五步操作：(1) 输入分解，将用户输入拆分为子任务；(2) 意图分类，确定每个子任务的意图类型；(3) TMS-DAG 更新，添加或修改节点并传播依赖；(4) 上下文检索，仅提取相关子图以减少 token 使用；(5) 响应生成，基于精简上下文生成一致性响应。
*   **关键创新:** TME 作为现有 LLMs 的轻量级层，通过空间记忆和动态任务图解决上下文过载和依赖追踪问题，同时支持模块化设计和任务特定定制。

## Experiment

*   **有效性:** TME 在四个多步骤场景（旅行计划、烹饪、会议安排、购物车编辑）中测试，涵盖 27 个用户轮次。结果显示，TME-DAG 在三个任务中完全消除幻觉和误解（100% 减少），整体上幻觉减少 66.7%，误解减少 83.3%，显著优于 ReAct 基线（ReAct 有 3 次幻觉和 5 次误解）。
*   **效率提升:** 在 token 使用上，TME-DAG 通过子图检索减少上下文冗余，在表单填写任务中减少 19.4% 的 token 使用（725 vs. 899），提升了交互的可扩展性。
*   **实验设置合理性:** 实验场景覆盖了从简单线性任务到复杂依赖任务的多种情况，使用 ChatGPT-4o 作为底层 LLM，确保方法通用性。消融研究验证了 TME-DAG 和 TRIM 模块的必要性，去掉任一组件均导致性能下降。
*   **局限性:** 在购物车编辑任务中，TME-DAG 因初始设计复杂性出现小错误（1 次幻觉和 1 次误解），需通过任务特定调整解决，表明方法在简单任务中的适应性有待优化。

## Further Thoughts

空间记忆框架（spatial memory framework）是一个极具启发性的概念，将任务组织为 DAG 结构不仅能有效追踪依赖，还能通过子图检索减少计算成本，这种思路可扩展到知识图谱与 LLM 结合或多用户协作任务图构建；此外，TRIM 模块的结构化意图分类设计提示我们，少样本提示生成的 JSON 输出可作为中间层，应用于其他需要结构化推理的场景，如代码生成或数据分析；未来引入图神经网络（GNN）增强依赖推理的方向也值得探索，尤其是在循环依赖或企业级任务中的应用潜力。