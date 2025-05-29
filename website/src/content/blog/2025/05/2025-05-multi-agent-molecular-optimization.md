---
title: "MT-Mol:Multi Agent System with Tool-based Reasoning for Molecular Optimization"
pubDatetime: 2025-05-27T07:27:30+00:00
slug: "2025-05-multi-agent-molecular-optimization"
type: "arxiv"
id: "2505.20820"
score: 0.64295751677468
author: "grok-3-latest"
authors: ["Hyomin Kim", "Yunhui Jang", "Sungsoo Ahn"]
tags: ["LLM", "Multi-Agent System", "Molecular Optimization", "Tool Integration", "Reasoning"]
institution: ["KAIST"]
description: "本文提出 MT-MOL 多智能体框架，通过工具引导的推理和角色分工的 LLM 智能体，在分子优化中实现高效且可解释的设计，在 PMO-1K 基准上显著超越基线。"
---

> **Summary:** 本文提出 MT-MOL 多智能体框架，通过工具引导的推理和角色分工的 LLM 智能体，在分子优化中实现高效且可解释的设计，在 PMO-1K 基准上显著超越基线。 

> **Keywords:** LLM, Multi-Agent System, Molecular Optimization, Tool Integration, Reasoning

**Authors:** Hyomin Kim, Yunhui Jang, Sungsoo Ahn

**Institution(s):** KAIST


## Problem Background

大型语言模型（LLMs）在分子优化领域（如药物设计）展现出巨大潜力，但现有方法缺乏结构化推理、解释性和全面的工具支持。
本文旨在通过多智能体协作和工具整合，解决分子优化中设计效率和透明度不足的问题。

## Method

*   **框架概述:** MT-MOL 是一个多智能体系统，将分子优化任务分解为四个角色：分析师、科学家、验证者和评审者，通过迭代协作实现分子设计。
*   **分析师智能体:** 包含五个专门的智能体，分别负责结构描述符、电子与拓扑特征、基于片段的功能基团、分子表示和其他化学性质，利用 RDKit 工具提取任务相关的化学特征，为设计提供数据支持。
*   **科学家智能体:** 基于分析师提供的信息和参考分子，生成新的分子（以 SMILES 格式表示），并通过分步推理解释设计逻辑，确保设计过程透明。
*   **验证者智能体:** 检查科学家智能体的推理步骤与生成的 SMILES 是否一致，识别逻辑与结构之间的不匹配，并要求重新生成直到一致或达到最大迭代次数。
*   **评审者智能体:** 基于工具分析结果，评估生成的分子和推理过程，提供化学依据充分的反馈，指导科学家智能体在下一轮迭代中改进设计。
*   **迭代过程:** 通过生成-验证-评审的循环，逐步优化分子设计，确保化学合理性和任务相关性，同时保持推理的可解释性。

## Experiment

*   **性能表现:** 在 PMO-1K 基准数据集（包含 23 个分子优化任务）上，MT-MOL 在 17 个任务中取得最先进性能，总 AUC 分数为 15.42，显著优于基线方法 MOLLEO（12.23）和 LICO（11.71）。
*   **具体提升:** 在复杂任务如 celecoxib_rediscovery 和 mestranol_similarity 上，AUC 提升幅度较大（分别从 0.512 到 0.867 和 0.630 到 0.996），显示出方法在化学多样性任务中的优势。
*   **效率分析:** AUC 曲线表明 MT-MOL 在早期阶段即可达到较高性能，适合预算受限的分子发现场景。
*   **消融研究:** 移除分析师、验证者或评审者智能体后性能下降明显，尤其是移除分析师（工具支持）导致 AUC 显著降低（如 albuterol_similarity 从 0.998 降至 0.750），验证了各组件的必要性。
*   **实验设置合理性:** 实验覆盖多种任务类型（药物再发现、相似性优化、多属性优化等），与多种基线方法对比全面，但多智能体结构增加了计算开销，可能影响大规模应用。

## Further Thoughts

多智能体协作框架通过角色分工模拟人类团队合作，这种方法不仅适用于分子优化，还可能推广到其他需要复杂推理的科学领域，如蛋白质设计或材料科学；此外，工具与 LLM 的深度整合为 AI 在领域特定任务中的应用提供了新思路，可以探索更多轻量级工具或跨领域工具的支持方式；迭代反馈机制也启发我们设计更健壮的 AI 系统，通过验证和评审减少错误，提升输出质量。