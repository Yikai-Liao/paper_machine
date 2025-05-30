---
title: "Rethinking the Unsolvable: When In-Context Search Meets Test-Time Scaling"
pubDatetime: 2025-05-28T12:28:18+00:00
slug: "2025-05-incontext-testtime-scaling"
type: "arxiv"
id: "2505.22290"
score: 0.799958836279087
author: "grok-3-latest"
authors: ["Fanzeng Xia", "Yidong Luo", "Tinko Sebastian Bartels", "Yaqi Xu", "Tongxin Li"]
tags: ["LLM", "In-Context Search", "Test-Time Scaling", "Reasoning", "Scaling Methods"]
institution: ["The Chinese University of Hong Kong, Shenzhen", "Beijing University of Posts and Telecommunications"]
description: "本文通过结合上下文搜索提示和测试时扩展技术，显著提升了大型语言模型在超难推理任务上的成功率（最高 30 倍），并从理论上证明了其扩展可解问题复杂性类别的潜力。"
---

> **Summary:** 本文通过结合上下文搜索提示和测试时扩展技术，显著提升了大型语言模型在超难推理任务上的成功率（最高 30 倍），并从理论上证明了其扩展可解问题复杂性类别的潜力。 

> **Keywords:** LLM, In-Context Search, Test-Time Scaling, Reasoning, Scaling Methods

**Authors:** Fanzeng Xia, Yidong Luo, Tinko Sebastian Bartels, Yaqi Xu, Tongxin Li

**Institution(s):** The Chinese University of Hong Kong, Shenzhen, Beijing University of Posts and Telecommunications


## Problem Background

大型语言模型（LLMs）在复杂推理任务上常常面临性能瓶颈，尤其是在成功率低于5%的‘不可解’任务上。
现有研究多采用直接提示等简单评估方法，未能充分挖掘 LLMs 的推理潜力，导致对其真实能力的低估。
本文旨在探索通过先进的上下文搜索提示和测试时扩展技术，是否能突破这一性能上限，并重新审视当前评估范式。

## Method

*   **核心思想:** 通过结合上下文搜索提示（In-Context Search Prompting）和测试时扩展（Test-Time Scaling），增强 LLMs 在复杂推理任务上的表现，尤其是在被认为‘不可解’的任务上。
*   **具体策略:** 
    *   **上下文搜索提示:** 包括三种方法：
        *   **直接提示（Direct Prompting）:** 提供简单的输入-输出示例对，依赖模型内部推理能力，无显式中间步骤指导。
        *   **思维链提示（Chain-of-Thought, CoT）:** 通过示例引导模型生成逐步推理步骤，通常采用贪婪搜索方式，缺乏复杂的算法操作如回溯。
        *   **算法思维提示（Algorithm-of-Thought, AoT）:** 提供结构化的算法搜索示例，指导模型模拟算法操作（如初始化、扩展、评估和回溯），以深度优先搜索等方式探索解空间。
    *   **测试时扩展:** 包括三种方式：
        *   **并行扩展（Parallel Scaling）:** 通过生成多个输出并选择最佳结果（如 Best-of-N 策略）提升性能。
        *   **顺序扩展（Sequential Scaling）:** 通过迭代计算逐步改进结果（如 Self-Refine 技术），基于前一步结果指导后续推理。
        *   **内部扩展（Internal Scaling）:** 模型根据任务复杂性自主调整计算资源（如激活‘思考模式’），动态分配推理步数，可能从多项式级别扩展到指数级别。
*   **实现细节:** 实验中结合不同提示和扩展策略进行消融研究，特别是在内部扩展与 AoT 提示结合时，模型能够处理更复杂的推理任务。
*   **关键点:** 不依赖外部工具或额外训练，仅通过提示和推理时调整即可显著提升性能，方法具有通用性和可扩展性。

## Experiment

*   **有效性:** 实验在控制的 NP 难任务（如 Vertex Cover）和复杂现实世界规划任务（如 Trip Planning）上显示，结合算法思维提示（AoT）和内部扩展（Internal Scaling）后，成功率提升显著，最高达 30 倍（从 <5% 提升至 40%，如 Claude 3.7 在 Trip Planning 任务上）。
*   **模型差异:** Claude 3.7 在大多数配置下表现优于 Qwen3，尤其在数值抽象推理任务上，Qwen3 几乎无提升，表明方法效果可能依赖于模型的基础推理能力。
*   **实验设置:** 实验覆盖多种任务类型和难度级别（难度 10），包括 100 个实例的测试集，设计较为全面合理，消融研究细致探讨了不同提示和扩展策略的组合效果。
*   **局限性:** 实验未详细讨论计算成本，内部扩展可能带来额外开销；此外，测试模型数量有限，未涉及更多架构或训练范式的验证。
*   **结论:** 结果表明当前评估方法低估了 LLMs 的推理潜力，结合高级策略可突破传统性能上限。

## Further Thoughts

论文启发我思考是否可以设计一种自适应提示生成机制，根据任务复杂性动态选择合适的提示策略（如 CoT 或 AoT），并结合内部扩展与其他技术（如强化学习）进一步优化模型在超难任务上的表现；此外，是否可以通过分析推理轨迹的质量，开发一种预测模型潜力上限的评估框架，以指导更高效的策略设计？