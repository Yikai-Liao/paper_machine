---
title: "MIRROR: Multi-agent Intra- and Inter-Reflection for Optimized Reasoning in Tool Learning"
pubDatetime: 2025-05-27T03:37:33+00:00
slug: "2025-05-mirror-reflection-toollearning"
type: "arxiv"
id: "2505.20670"
score: 0.5745743982476497
author: "grok-3-latest"
authors: ["Zikang Guo", "Benfeng Xu", "Xiaorui Wang", "Zhendong Mao"]
tags: ["LLM", "Multi-Agent System", "Reflection", "Tool Learning", "Reasoning"]
institution: ["University of Science and Technology of China", "Metastone Technology, Beijing, China"]
description: "本文提出 MIRROR 框架，通过多智能体系统中的内部反思和外部反思机制，显著提升工具学习任务中的决策质量和协作效率。"
---

> **Summary:** 本文提出 MIRROR 框架，通过多智能体系统中的内部反思和外部反思机制，显著提升工具学习任务中的决策质量和协作效率。 

> **Keywords:** LLM, Multi-Agent System, Reflection, Tool Learning, Reasoning

**Authors:** Zikang Guo, Benfeng Xu, Xiaorui Wang, Zhendong Mao

**Institution(s):** University of Science and Technology of China, Metastone Technology, Beijing, China


## Problem Background

大型语言模型（LLMs）在处理复杂任务时，尤其是在需要工具集成和动态适应的场景下，面临显著挑战，如实时信息获取、新数据模式处理和精确系统控制的局限性。
现有反思机制（如 Reflexion）仅在执行后进行分析，无法预防初始错误，可能导致不可逆的系统变化和高昂的学习成本。
论文旨在通过多智能体工作流和新型反思机制，解决工具学习任务中错误预防和轨迹优化的关键问题。

## Method

*   **核心思想:** 提出 MIRROR 框架，通过结合内部反思（Intra-Reflection）和外部反思（Inter-Reflection），在多智能体系统中优化工具学习任务的决策和协作效率。
*   **框架结构:** 包含三个专门智能体：
    *   **Planner Agent**：将复杂任务分解为子任务序列，考虑依赖关系和执行顺序。
    *   **Tool Agent**：为每个子任务选择合适的工具和参数。
    *   **Answer Agent**：综合所有子任务结果，生成最终答案。
*   **内部反思（Intra-Reflection）**：
    *   每个智能体在输出前进行自我评估，确保质量达到预设阈值（基于角色特定的标准，如完整性、效率等）。
    *   若未达标，则迭代修正输出，例如 Planner Agent 优化任务分解，Tool Agent 调整工具选择，Answer Agent 改进答案呈现。
    *   此机制作为预防性措施，嵌入智能体生成周期，无需额外模型或外部反馈。
*   **外部反思（Inter-Reflection）**：
    *   通过双重记忆架构实现：短期记忆（STM）记录子任务执行失败和反思输出，用于 Tool Agent 的即时调整；长期记忆（LTM）存储任务级轨迹，用于整体策略优化。
    *   支持系统从失败中学习，持续改进任务分解和工具选择策略。
*   **智能体协作**：
    *   智能体通过反思机制确保高质量输出传递，形成自优化系统，减少错误传播并提升整体性能。
*   **关键创新:** 内部反思在执行前预防错误，外部反思在执行后优化轨迹，二者结合形成全面的错误预防和修正系统。

## Experiment

*   **有效性:** MIRROR 在 StableToolBench 和 TravelPlanner 两个基准上显著优于基线方法（如 ReAct, Reflexion, DFSDT）。
    *   在 StableToolBench 上，平均通过率（Pass Rate）比次优方法高出 2.5% 至 7.0%，胜率（Win Rate）提升 5.2% 至 9.2%。
    *   在 TravelPlanner 上，交付率（Delivery Rate）提升 14.4% 至 21.1%，常识和硬性约束通过率也有显著改善。
*   **实验设置合理性:** 实验覆盖多种 LLM 核心（如 GPT-3.5 Turbo, GPT-4o, Qwen2.5-72B），并在不同复杂度的任务场景中测试，设置全面。
    *   消融研究验证了内部反思和外部反思的重要性，例如移除所有内部反思导致通过率下降 7.0%。
*   **局限性:** TravelPlanner 上的最终通过率（Final Pass Rate）仍较低，反映复杂规划任务中多约束管理的挑战；实验受预算限制，仅测试少量 LLM 核心，可能影响普适性。
*   **成本与效率:** MIRROR 虽增加 token 消耗（如 5 轮反思耗费 13.6k token/查询），但通过早期错误检测和路径优化实现较高性能，成本-收益权衡合理。

## Further Thoughts

内部反思的概念（执行前自我评估）可推广至其他领域，如自动驾驶或医疗诊断中的决策优化，预防错误并提升系统可靠性。
双重记忆架构（STM 和 LTM）启发多智能体系统中短期适应与长期学习的平衡设计，未来可通过跨任务记忆机制进一步提升泛化能力。
反思机制的计算成本与性能提升之间的权衡值得关注，可探索自适应反思策略，根据任务复杂性动态调整反思轮数或阈值，优化资源利用。