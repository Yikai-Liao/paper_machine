---
title: "MCP-RADAR: A Multi-Dimensional Benchmark for Evaluating Tool Use Capabilities in Large Language Models"
pubDatetime: 2025-05-22T14:02:37+00:00
slug: "2025-05-mcp-radar-benchmark"
type: "arxiv"
id: "2505.16700"
score: 0.4847748651286215
author: "grok-3-latest"
authors: ["Xuanqi Gao", "Juan Zhai", "Siyi Xie", "Shiqing Ma", "Chao Shen"]
tags: ["LLM", "Tool Use", "Evaluation Framework", "Efficiency Tradeoff", "Reasoning"]
institution: ["Xi’an Jiaotong University", "University of Massachusetts at Amherst"]
description: "本文提出 MCP-R ADAR 基准，通过五维度客观量化指标全面评估大型语言模型在 MCP 框架下的工具使用能力，揭示性能差异和权衡，为模型与工具优化提供指导。"
---

> **Summary:** 本文提出 MCP-R ADAR 基准，通过五维度客观量化指标全面评估大型语言模型在 MCP 框架下的工具使用能力，揭示性能差异和权衡，为模型与工具优化提供指导。 

> **Keywords:** LLM, Tool Use, Evaluation Framework, Efficiency Tradeoff, Reasoning

**Authors:** Xuanqi Gao, Juan Zhai, Siyi Xie, Shiqing Ma, Chao Shen

**Institution(s):** Xi’an Jiaotong University, University of Massachusetts at Amherst


## Problem Background

随着大型语言模型（LLMs）从被动文本生成器转变为能够与外部工具交互的主动推理代理，工具使用能力成为评估模型性能的重要维度。
然而，现有的评估方法主要关注知识推理和文本生成，缺乏针对工具使用能力的系统化基准，尤其是在 Model Context Protocol (MCP) 这一标准化工具交互框架下的表现评估。
论文试图解决如何客观、全面地评估 LLMs 在 MCP 环境下的工具使用能力，并揭示模型在不同任务领域和性能维度上的优劣势。

## Method

*   **核心思想:** 提出 MCP-R ADAR 基准，通过多维度、客观量化的评估框架，全面衡量 LLMs 在 MCP 框架下的工具使用能力。
*   **评估维度:** 设计了五个互补的评估维度：
    *   结果准确性（Result Accuracy, RA）：衡量任务完成成功率，关注最终结果的正确性。
    *   工具选择效率（Dynamic Tool Selection Rate, DTSR）：评估工具调用过程中的每轮准确性，反映模型在多步任务中的行为质量。
    *   首次错误位置（First Error Position, FEP）：测量首次错误在工作流程中的位置，用于评估模型在长序列任务中的稳定性。
    *   计算资源效率（Computational Resource Efficiency, CRE）：基于 token 消耗量评估模型的计算资源利用效率。
    *   响应时间效率（Response Time Efficiency, RTE）：测量从用户输入到最终响应的时间效率，反映模型在实时交互中的表现。
*   **数据集构建:** 构建包含 300 个任务的基准数据集，覆盖软件工程、数学推理和通用问题解决三个领域，每个领域 100 个任务，任务按复杂度分级（1-3 级），并配备 42 个主流 MCP 工具。
*   **评估方式:** 采用自动化、客观的量化指标，避免主观人工评估，确保结果可重复性；通过雷达图可视化模型在五个维度上的表现，便于直观比较。
*   **适用性:** 虽然聚焦于 MCP 框架，但方法设计具有普适性，可扩展到其他工具集成框架。

## Experiment

*   **有效性:** 实验评估了 7 个主流 LLMs（包括 Claude 3.7、Gemini 2.5 Pro、GPT-4o 等闭源模型和 DeepSeek-V3、Llama 3.3 等开源模型），结果表明模型在数学推理领域的表现最佳（平均 RA 0.78），显著优于软件工程（0.42）和通用问题解决（0.28）；Gemini 2.5 Pro 在数学领域的 RA 达到 0.91，为最高。
*   **性能权衡:** 实验揭示了准确性与效率之间的明显权衡，例如 DeepSeek-V3 在计算资源效率上表现突出（CRE=1813），但准确性较低（RA=0.59）；而 Gemini 2.5 Pro 准确性高，但资源消耗大（CRE=6187）。
*   **实验设置合理性:** 实验覆盖多个模型、任务领域和工具配置，任务复杂度分级和多轮验证确保数据集质量；使用标准化 MCP 测试环境和 Openrouter 接口提供商，保证结果可比性；实验数据通过三轮运行取平均值，进一步提高可靠性。
*   **不足之处:** 实验未深入探讨工具文档质量对模型性能的影响，仅在讨论中提及，可能是未来改进方向。

## Further Thoughts

MCP-R ADAR 的多维度评估框架为工具使用能力的评估提供了一个通用思路，未来可以扩展到更多交互模式或领域；此外，论文强调错误恢复机制比单纯避免错误更关键，这启发我们可以通过强化学习或专门的错误恢复数据集提升模型鲁棒性；‘快优先于准’的策略也提示在实时交互场景中，工具设计应注重快速尝试和纠正，而模型训练应平衡局部准确性和全局效率。