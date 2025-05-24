---
title: "MCP-RADAR: A Multi-Dimensional Benchmark for Evaluating Tool Use Capabilities in Large Language Models"
pubDatetime: 2025-05-22T14:02:37+00:00
slug: "2025-05-mcp-radar-benchmark"
type: "arxiv"
id: "2505.16700"
score: 0.4847748651286215
author: "grok-3-latest"
authors: ["Xuanqi Gao", "Juan Zhai", "Siyi Xie", "Shiqing Ma", "Chao Shen"]
tags: ["LLM", "Tool Integration", "Benchmarking", "Efficiency", "Accuracy"]
institution: ["Xi’an Jiaotong University", "University of Massachusetts at Amherst"]
description: "本文提出 MCP-R ADAR，首个针对大型语言模型工具使用能力的多维度评估基准，通过五维度客观量化框架揭示模型性能权衡，为优化模型和工具设计提供指导。"
---

> **Summary:** 本文提出 MCP-R ADAR，首个针对大型语言模型工具使用能力的多维度评估基准，通过五维度客观量化框架揭示模型性能权衡，为优化模型和工具设计提供指导。 

> **Keywords:** LLM, Tool Integration, Benchmarking, Efficiency, Accuracy

**Authors:** Xuanqi Gao, Juan Zhai, Siyi Xie, Shiqing Ma, Chao Shen

**Institution(s):** Xi’an Jiaotong University, University of Massachusetts at Amherst


## Problem Background

随着大型语言模型（LLMs）从被动文本生成器向主动推理代理演变，模型上下文协议（MCP）作为工具交互的标准化框架被广泛采用，但现有评估方法未能充分衡量模型在这一范式下的工具使用能力。
关键问题在于传统评估多依赖主观判断或单一二元指标，忽视了工具利用的多维度表现（如准确性、效率、速度），从而无法为模型和工具开发者提供细致指导。

## Method

*   **核心框架：MCP-R ADAR 基准测试**：提出了一种基于客观量化、多维度互补和领域普适性原则的评估框架，专门用于评估 LLMs 在 MCP 环境下的工具使用能力。
*   **五维度评估体系**：
    *   **结果准确性（RA）**：衡量任务完成成功率，通过成功任务数与总任务数的比值计算，关注最终结果正确性。
    *   **工具调用过程准确性（TCPA/DTSR）**：评估工具调用每轮的准确性，通过错误调用比例计算，反映模型在多步任务中的行为质量。
    *   **首次错误位置（FEP）**：测量首次错误在工作流中的位置，通过错误发生深度比例计算，评估模型在长序列任务中的稳定性。
    *   **计算资源效率（CRE）**：基于 token 消耗量评估计算资源利用效率，与基准模型对比，反映模型运行成本。
    *   **响应时间效率（RTE）**：测量从输入到响应的时间效率，与基准模型对比，影响实时应用场景的用户体验。
*   **数据集设计**：构建包含 300 个任务的基准测试集，覆盖软件工程、数学推理和通用问题解决三个领域，每个领域 100 个任务，按复杂度分级（1-3 级），配备 42 个主流 MCP 工具，确保场景多样性。
*   **可视化与分析工具**：采用雷达图展示模型在五维度上的表现，直观揭示各模型的强弱项，而非单纯排名，辅助开发者针对性优化。
*   **扩展性**：框架设计注重普适性，不仅适用于 MCP，也可推广至其他工具集成环境，并支持新增领域和任务。

## Experiment

*   **有效性**：实验评估了 7 个主流 LLMs（包括 Claude 3.7、Gemini 2.5 Pro、GPT-4o 等闭源模型及 DeepSeek-V3、Llama 3.3 70B 等开源模型），结果显示模型在数学推理领域的表现最佳（平均 RA 0.78），远高于软件工程（0.42）和通用问题解决（0.28），表明当前 LLMs 在结构化任务上的工具使用能力更强；Gemini 2.5 Pro 在数学领域 RA 达 0.91，为最高。
*   **性能权衡**：实验揭示了准确性与效率的显著权衡，例如 DeepSeek-V3 在计算资源效率上占优（数学领域 CRE 为 1813，远低于 Gemini 2.5 Pro 的 6187），但准确性较低（RA 0.59）；GPT-4o-mini 在响应时间上表现突出（通用问题领域 RTE 为 1077），适合实时应用。
*   **设置合理性**：实验覆盖多个模型、任务领域和工具配置，通过三次独立运行取平均值增强可靠性；雷达图可视化帮助识别模型性能特征（如 GPT-4o 的平衡性）；但实验也显示所有模型在通用问题解决领域的中间过程表现与最终准确性差距较大，工具链整合能力需改进。
*   **全面性**：实验设置较为全面，数据呈现清晰，但对工具设计模式对模型性能的具体影响探讨不足，仅提供初步指导。

## Further Thoughts

MCP-R ADAR 的多维度评估框架启发我们可以在更多 AI 应用场景中设计类似体系，例如评估多轮对话一致性或情感交互细腻度；工具设计对模型性能的影响提示未来可探索自动化工具接口优化或自适应文档生成以降低调用难度；错误恢复机制的重要性表明可以在训练中加入错误解析任务，或设计动态调整策略应对复杂任务后期错误。