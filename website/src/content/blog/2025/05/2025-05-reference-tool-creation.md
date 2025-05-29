---
title: "RefTool: Enhancing Model Reasoning with Reference-Guided Tool Creation"
pubDatetime: 2025-05-27T16:41:19+00:00
slug: "2025-05-reference-tool-creation"
type: "arxiv"
id: "2505.21413"
score: 0.7210981007869995
author: "grok-3-latest"
authors: ["Xiao Liu", "Da Yin", "Zirui Wu", "Yansong Feng"]
tags: ["LLM", "Tool Creation", "Reasoning", "External Knowledge", "Hierarchical Selection"]
institution: ["Wangxuan Institute of Computer Technology, Peking University", "University of California, Los Angeles"]
description: "R EF T OOL 框架通过参考引导的工具创建和层次化选择机制，显著提升了大型语言模型在复杂问题解决中的推理能力，克服了内部知识局限。"
---

> **Summary:** R EF T OOL 框架通过参考引导的工具创建和层次化选择机制，显著提升了大型语言模型在复杂问题解决中的推理能力，克服了内部知识局限。 

> **Keywords:** LLM, Tool Creation, Reasoning, External Knowledge, Hierarchical Selection

**Authors:** Xiao Liu, Da Yin, Zirui Wu, Yansong Feng

**Institution(s):** Wangxuan Institute of Computer Technology, Peking University, University of California, Los Angeles


## Problem Background

大型语言模型（LLMs）在复杂问题解决任务（如数学推理、因果分析、物理和化学问题）中可以通过外部工具增强推理能力，但并非所有任务都有现成的工具。
现有方法依赖模型内部知识生成工具，然而当任务涉及模型知识范围之外的领域（如专业或新兴领域）时，这些方法往往失效。
论文提出通过外部结构化参考材料（如教科书）生成工具，解决 LLMs 在缺乏领域知识时无法有效生成和使用工具的关键问题。

## Method

*   **核心思想:** 提出 R EF T OOL 框架，通过外部参考材料（如教科书）引导工具创建，克服模型内部知识局限，并在推理时通过层次化选择机制高效使用工具。
*   **工具创建模块 (Tool Creation):** 
    *   **结构提取:** 从参考材料中提取层次结构（如章节和子章节），保留其系统性知识组织方式。
    *   **初始工具生成:** 针对每个章节内容，指导 LLM 生成可执行工具，每个工具包括自然语言描述、Python 函数实现（含参数和返回值的注释）以及示范示例（优先从参考材料中提取，否则由模型生成）。
    *   **工具过滤与优化:** 通过执行测试和输出验证，使用示范示例过滤无效工具；对失败工具提供反馈进行优化，最终将有效工具按参考材料结构组织成层次化工具箱。
*   **工具使用模块 (Tool Utilization):** 
    *   **层次化工具选择:** 在推理时，采用两阶段检索：首先基于参考材料目录选择相关章节（最多 nc 个），然后在选定章节内选择具体工具（最多 nt 个），若无合适工具则退化为标准推理。
    *   **解决方案生成:** 将选定工具集成到推理过程中，支持单轮 Program-of-Thoughts (PoT) 或多轮 ReAct 推理范式，模型根据工具描述和示例调用工具生成答案。
*   **关键创新:** 工具生成不依赖模型内部知识，而是基于外部参考材料，确保工具的准确性和领域适应性；层次化组织和选择机制提高了工具检索效率，避免了大规模工具库中的选择困难。

## Experiment

*   **有效性:** R EF T OOL 在因果分析（QRData）、物理（TheoremQA）和化学（SciBench）三个领域上，平均准确率比现有工具生成方法（如 Creator）和领域特定推理方法提高了 11.3%，在多个模型（Llama-3.1-70B, Gemini-1.5-Pro, GPT-4, GPT-4o）上均表现出色。
*   **对比分析:** 相比检索增强生成（RAG），R EF T OOL 通过工具形式和层次化选择更好地利用参考材料（平均准确率提升 1.9%）；相比其他工具生成方法，参考引导避免了内部知识局限（比 Creator 提升 12.3%）。
*   **成本效率:** 与领域特定方法（如 ChemAgent）相比，R EF T OOL 的工具构建和推理时间成本大幅降低（化学领域工具构建时间减少 99%，推理成本减少 57-97%）。
*   **泛化性与鲁棒性:** 工具在不同数据集（如 SciBench-fund）上表现出良好的可重用性，且使用不同模型（如 Gemini-1.5-Pro）创建工具仍保持优越性能，表明方法并非数据集特异性。
*   **实验设置合理性:** 实验覆盖多个领域、数据集和模型，验证了方法的广泛适用性；但也存在局限，如对模型基础领域知识的依赖，若模型缺乏基础理解可能导致工具生成错误。

## Further Thoughts

论文通过外部参考材料生成工具的思路非常具有启发性，未来可以扩展到更多类型的结构化知识源（如学术论文、行业报告）甚至动态更新的在线资源，以适应快速变化的领域；此外，层次化工具选择机制也启发我们可以在工具管理中引入更复杂的结构（如知识图谱），通过语义关联进一步提高工具检索的精度和效率，尤其是在大规模工具库中。