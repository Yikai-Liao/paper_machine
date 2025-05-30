---
title: "MT-Mol:Multi Agent System with Tool-based Reasoning for Molecular Optimization"
pubDatetime: 2025-05-27T07:27:30+00:00
slug: "2025-05-multi-agent-molecular-optimization"
type: "arxiv"
id: "2505.20820"
score: 0.64295751677468
author: "grok-3-latest"
authors: ["Hyomin Kim", "Yunhui Jang", "Sungsoo Ahn"]
tags: ["LLM", "Multi-Agent System", "Molecular Optimization", "Tool Integration", "Iterative Reasoning"]
institution: ["KAIST"]
description: "本文提出 MT-MOL，一个多智能体分子优化框架，通过工具引导的推理和角色协作，在 PMO-1K 基准上显著提升了分子设计效率和质量。"
---

> **Summary:** 本文提出 MT-MOL，一个多智能体分子优化框架，通过工具引导的推理和角色协作，在 PMO-1K 基准上显著提升了分子设计效率和质量。 

> **Keywords:** LLM, Multi-Agent System, Molecular Optimization, Tool Integration, Iterative Reasoning

**Authors:** Hyomin Kim, Yunhui Jang, Sungsoo Ahn

**Institution(s):** KAIST


## Problem Background

大型语言模型（LLMs）在分子优化（如药物设计）中展现出潜力，但现有方法在结构化推理、解释性和全面工具支持方面存在不足。
论文旨在解决如何通过多智能体协作、工具整合和迭代推理，在低预算设置下设计高质量分子候选物，同时提升过程的化学合理性和透明度。

## Method

*   **框架概述:** 提出 MT-MOL，一个多智能体系统，将分子优化任务分解为四个角色：分析师、科学家、验证者和评审者，通过协作实现可解释、化学合理的分子设计。
*   **分析师 (Analyst):** 包含五个子智能体，分别负责结构描述符、电子与拓扑特征、基于片段的功能基团、分子表示和其他化学属性，利用 154 个 RDKit 工具提取任务相关化学信息，为后续设计提供数据支持。
*   **科学家 (Scientist):** 基于分析师的工具分析和参考分子，生成新的分子（以 SMILES 格式表示），并通过分步推理解释设计逻辑，确保设计过程透明。
*   **验证者 (Verifier):** 检查科学家生成的推理步骤与分子结构是否一致，若发现逻辑与结果不匹配，则要求科学家重新生成，确保设计的一致性和准确性。
*   **评审者 (Reviewer):** 基于工具分析和任务目标，评估生成的分子和推理过程，提供详细的化学反馈，推动科学家在下一轮迭代中改进设计。
*   **迭代过程:** 通过科学家-验证者-评审者之间的多轮交互，实现分子设计的逐步优化，同时保持对任务目标的高对齐性。

## Experiment

*   **性能表现:** 在 PMO-1K 基准数据集（包含 23 个分子优化任务）上，MT-MOL 在 17 个任务中取得最优性能，总 AUC 分数达 15.42，显著优于基线方法（如 MOLLEO 的 12.23）。
*   **效率优势:** AUC 曲线显示 MT-MOL 在早期生成阶段即可达到较高性能，特别是在复杂任务（如 celecoxib_rediscovery）上提升明显，表明其在低预算设置下的高效性。
*   **消融验证:** 移除分析师、验证者或评审者均导致性能下降，其中分析师的工具支持影响最大，验证了多智能体协作和工具整合的必要性。
*   **实验设置合理性:** 实验涵盖多种任务类型（如药物再发现、结构相似性优化），并与多种基线（如 LICO、MOLLEO）对比，设置全面且结果可信。
*   **局限性:** 多智能体结构增加了计算开销，可能限制资源受限场景下的应用；依赖 RDKit 等规则工具可能影响对新型化学空间的泛化能力。

## Further Thoughts

MT-MOL 的多智能体协作结合领域工具的框架启发了我，是否可以将这种角色分解和迭代反馈机制推广到其他科学领域（如材料设计），并通过强化学习优化智能体协作策略？此外，工具整合的思路也让我思考，未来是否可以引入更复杂的化学模拟工具（如量子计算）或动态数据库，以提升分析精度和适应性。