---
title: "Rethinking the Unsolvable: When In-Context Search Meets Test-Time Scaling"
pubDatetime: 2025-05-28T12:28:18+00:00
slug: "2025-05-incontext-testtime-scaling"
type: "arxiv"
id: "2505.22290"
score: 0.799958836279087
author: "grok-3-latest"
authors: ["Fanzeng Xia", "Yidong Luo", "Tinko Sebastian Bartels", "Yaqi Xu", "Tongxin Li"]
tags: ["LLM", "In-Context Learning", "Test-Time Scaling", "Reasoning", "Prompting"]
institution: ["The Chinese University of Hong Kong, Shenzhen", "Beijing University of Posts and Telecommunications"]
description: "本文通过结合上下文搜索提示和测试时扩展，显著提升大型语言模型在复杂推理任务上的成功率（最高30倍），并从理论和实证上挑战了现有评估方法的系统性低估。"
---

> **Summary:** 本文通过结合上下文搜索提示和测试时扩展，显著提升大型语言模型在复杂推理任务上的成功率（最高30倍），并从理论和实证上挑战了现有评估方法的系统性低估。 

> **Keywords:** LLM, In-Context Learning, Test-Time Scaling, Reasoning, Prompting

**Authors:** Fanzeng Xia, Yidong Luo, Tinko Sebastian Bartels, Yaqi Xu, Tongxin Li

**Institution(s):** The Chinese University of Hong Kong, Shenzhen, Beijing University of Posts and Telecommunications


## Problem Background

大型语言模型（LLMs）在复杂推理任务（如NP难问题和长距离规划）上表现不佳，成功率常低于5%，被认为是‘不可解’。
现有评估多依赖直接提示（Direct Prompting）或简单上下文学习，未能充分挖掘模型潜力，存在系统性低估，导致对LLMs推理边界的误判。

## Method

*   **核心思想:** 通过结合上下文搜索提示（In-Context Search Prompting）和测试时扩展（Test-Time Scaling），增强LLMs在复杂推理任务上的表现，突破现有性能天花板。
*   **上下文搜索提示的具体策略:**
    *   **直接提示（Direct Prompting）:** 提供少量问题-答案对作为示例，依赖模型内部推理能力，无需显式推理步骤指导。
    *   **思维链提示（Chain-of-Thought, CoT）:** 通过示例引导模型生成逐步推理步骤，模拟贪婪搜索过程，帮助模型分解问题并逐步推导出答案。
    *   **算法思维提示（Algorithm-of-Thought, AoT）:** 提供详细的算法搜索示例（如深度优先搜索），指导模型模拟结构化的算法操作，支持复杂多路径推理。
*   **测试时扩展的具体方式:**
    *   **并行扩展（Parallel Scaling）:** 在推理时生成多个输出并选择最佳结果（如Best-of-N方法，N=3），通过多样化尝试提升准确性。
    *   **顺序扩展（Sequential Scaling）:** 采用迭代方式优化输出（如Self-Refine技术），基于前一轮结果逐步改进答案。
    *   **内部扩展（Internal Scaling）:** 模型根据任务复杂性自主调整推理深度（如激活‘思考模式’），通过内部学习策略动态分配计算资源。
*   **实现细节与关键点:** 方法不依赖外部机制或额外微调，仅通过提示和推理时计算资源调整实现性能提升；重点在于高级提示（如AoT）与内部扩展的协同作用，以模拟更复杂的推理过程。

## Experiment

*   **有效性:** 实验在受控NP难任务（如Vertex Cover、3-Dimensional Matching）和复杂现实世界规划任务（如Trip Planning、Meeting Planning）上进行，难度设为最高级别（Level 10）。结果显示，单独使用上下文搜索或测试时扩展时，成功率提升有限（常低于5%）；但结合高级上下文搜索（如AoT）和内部扩展后，成功率显著提高，最高达30倍（如Claude 3.7在Trip Planning上从0%提升至40%）。
*   **合理性与全面性:** 实验设置覆盖多种提示策略和扩展方法的细粒度组合，任务选择具有代表性，难度控制合理，测试了两个模型（Qwen3和Claude 3.7），结果趋势一致，表明方法普适性强。
*   **模型差异:** Claude 3.7在大多数任务上表现优于Qwen3，尤其在数值推理任务（如Vertex Cover）中，Qwen3几乎无提升，可能是训练数据或内部机制差异所致。
*   **局限性:** 实验未涉及外部机制（如多轮交互或工具辅助），可能仍未完全挖掘模型潜力；此外，计算开销随扩展策略增加而上升，实际应用需权衡效率。

## Further Thoughts

论文揭示了推理步长和质量对LLMs性能的深远影响，启发我们探索如何优化推理路径以减少冗余计算，例如通过自动化提示设计或自适应策略生成高效推理示例；此外，不同模型对提示和扩展策略的响应差异显著，未来可以研究通用的推理增强框架，适应多种模型架构，甚至结合外部工具或多轮交互进一步突破性能边界。