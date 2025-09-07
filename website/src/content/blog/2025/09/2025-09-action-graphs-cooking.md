---
title: "Towards an Action-Centric Ontology for Cooking Procedures Using Temporal Graphs"
pubDatetime: 2025-09-04T12:34:56+00:00
slug: "2025-09-action-graphs-cooking"
type: "arxiv"
id: "2509.04159"
score: 0.57894032202554
author: "grok-3-latest"
authors: ["Aarush Kumbhakern", "Saransh Kumar Gupta", "Lipika Dey", "Partha Pratim Das"]
tags: ["Semantic Modeling", "Action Graphs", "Recipe Formalization", "Temporal Constraints", "Concurrency"]
institution: ["Ashoka University"]
description: "本文提出一种基于动作图的领域特定语言（DSL），通过有向无环图结构化表示烹饪食谱，捕捉状态变化、空间转移、并发动作和环境上下文，为计算美食学和智能厨房技术奠定了基础。"
---

> **Summary:** 本文提出一种基于动作图的领域特定语言（DSL），通过有向无环图结构化表示烹饪食谱，捕捉状态变化、空间转移、并发动作和环境上下文，为计算美食学和智能厨房技术奠定了基础。 

> **Keywords:** Semantic Modeling, Action Graphs, Recipe Formalization, Temporal Constraints, Concurrency

**Authors:** Aarush Kumbhakern, Saransh Kumar Gupta, Lipika Dey, Partha Pratim Das

**Institution(s):** Ashoka University


## Problem Background

烹饪食谱作为程序性文本，因其语言模糊性、隐含上下文和输入目标的多样性，给计算建模带来了挑战。
现有形式化方法（如线性动作列表）无法有效捕捉状态转换、并发性、空间操作和环境影响等复杂动态，限制了自动化烹饪、机器人操作和营养分析等应用的发展。
作者旨在设计一种结构化表示框架，通过动作图（Action Graphs）实现食谱的精确语义建模和机器理解。

## Method

*   **核心思想:** 提出一种领域特定语言（DSL），将食谱表示为有向无环图（DAG），以动作节点和依赖边捕捉烹饪过程中的状态变化、空间转移和时间顺序。
*   **动作类型:** 定义三种基本动作节点：
    *   **Process（处理）:** 表示物理或化学状态变化，参数包括输入（食材或部分处理组件）、技术、工具、温度曲线、时间/持续时间、完成条件和修饰符，技术参数引用标准化烹饪技术词典。
    *   **Transfer（转移）:** 表示食材或部分处理组件在环境间的空间转移（如从盘子到锅中），更新后续动作的环境关联，捕捉上下文驱动的状态变化。
    *   **Plate（摆盘）:** 用于最终组装和呈现，目前尚未充分开发。
*   **实体建模:** 包括食材（Ingredients，含名称、数量、单位等）、环境（Environments，作为容器、位置和几何形状的元组）和部分处理组件（PPCs，作为中间状态）。
*   **中间状态与溯源:** PPCs 作为隐式中间状态，不显式创建节点以保持图的紧凑性，但通过子图回溯可恢复完整溯源信息。
*   **并发与相对时间:** 通过并行分支和合并点支持并发动作，通过相对时间字段和子过程建模交错步骤（如‘在炒菜中途加入大蒜’），在 DAG 中保持部分顺序约束。
*   **扩展性与模块化:** 动作和环境引用外部定义，支持新元素引入；食谱可作为插件节点嵌入其他食谱，支持复杂菜肴的模块化构建。

## Experiment

*   **评估方式:** 通过手动编码一个复杂的‘全英式早餐’食谱，定性评估 DSL 的表达能力和实用性，食谱因其多组件、异构工具和环境、并发流程和频繁 PPC 重新分配而具有高复杂性。
*   **结果展示:** 动作图成功编码了食谱的复杂工作流程，展示了 DSL 在处理并发活动、环境转移和精确参数（如温度范围、技术引用、工具声明、终止条件）方面的能力。
*   **比较分析:** 与三种基线方法（MILK、Corel、Culinary Grammar）进行定量和定性比较，Action-Graph DSL 覆盖率达 72.4%，显著高于 MILK（46.6%）、Corel（31.0%）和 Bagler（43.1%），尤其在并发性、环境谱系和相对时间建模上表现突出。
*   **实验局限:** 目前仅限于手动编码和定性评估，缺乏大规模自动化解析和执行验证，实验设置合理但广度和深度不足，未来需在多样化食谱上进一步测试。

## Further Thoughts

动作图的结构化表示思路不仅适用于烹饪食谱，还可推广至其他程序性任务领域（如工业流程或医疗操作），为复杂任务的语义建模和自动化执行提供新视角；
隐式状态与溯源平衡的设计启发我们在处理大规模复杂流程时，如何在紧凑性和信息完整性之间找到折衷；
并发与相对时间建模为多任务或多智能体协作场景（如机器人任务规划）提供了潜在的借鉴意义。