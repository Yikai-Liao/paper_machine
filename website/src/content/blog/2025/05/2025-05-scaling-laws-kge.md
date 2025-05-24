---
title: "How do Scaling Laws Apply to Knowledge Graph Engineering Tasks? The Impact of Model Size on Large Language Model Performance"
pubDatetime: 2025-05-22T06:21:40+00:00
slug: "2025-05-scaling-laws-kge"
type: "arxiv"
id: "2505.16276"
score: 0.6193150029641706
author: "grok-3-latest"
authors: ["Desiree Heim", "Lars-Peter Meyer", "Markus Schröder", "Johannes Frey", "Andreas Dengel"]
tags: ["LLM", "Knowledge Graph", "Scaling Laws", "Benchmarking", "Cost Effectiveness"]
institution: ["DFKI, Kaiserslautern, Germany", "RPTU, Kaiserslautern, Germany", "InfAI, Leipzig, Germany", "TU Chemnitz, Germany", "Uni Leipzig, Germany"]
description: "本文通过 LLM-KG-Bench 框架分析了模型规模对大型语言模型在知识图谱工程任务中性能的影响，发现规模越大性能通常越好，但存在 plateau 和 ceiling 效应，为成本效益导向的模型选择提供了指导。"
---

> **Summary:** 本文通过 LLM-KG-Bench 框架分析了模型规模对大型语言模型在知识图谱工程任务中性能的影响，发现规模越大性能通常越好，但存在 plateau 和 ceiling 效应，为成本效益导向的模型选择提供了指导。 

> **Keywords:** LLM, Knowledge Graph, Scaling Laws, Benchmarking, Cost Effectiveness

**Authors:** Desiree Heim, Lars-Peter Meyer, Markus Schröder, Johannes Frey, Andreas Dengel

**Institution(s):** DFKI, Kaiserslautern, Germany, RPTU, Kaiserslautern, Germany, InfAI, Leipzig, Germany, TU Chemnitz, Germany, Uni Leipzig, Germany


## Problem Background

知识图谱工程（Knowledge Graph Engineering, KGE）任务，如创建和维护知识图谱（Knowledge Graphs, KGs），通常需要大量手动工作，而大型语言模型（LLMs）的出现为自动化提供了潜力。
然而，模型规模的增加往往伴随着更高的资源成本（如内存和计算需求），因此需要在性能和成本之间找到平衡。
本文研究了规模法则（Scaling Laws）在 KGE 任务中的适用性，探讨模型规模是否总是与性能正相关，以及是否存在成本效益更优的较小模型。

## Method

*   **基准测试框架:** 使用 LLM-KG-Bench 框架对 26 个开源 LLMs 进行性能评估，模型规模从 0.5B 到 72B 参数不等，涵盖多个模型家族（如 Qwen、Meta-LLama、Microsoft-Phi）。
*   **任务设计:** 测试任务包括 7 个 KGE 相关类别（如 RdfConnectionExplain、RdfSyntaxFixing、Text2Sparql），共 23 个任务变体，涉及 RDF 和 SPARQL 的理解与生成，任务输入格式包括 JSON-LD、Turtle 等多种序列化格式。
*   **评估指标:** 采用三种类型的性能指标：Central 指标（关注输出格式准确性，如 listTrimF1）、Fragment 指标（关注部分正确性，如 contentF1）和 Syntax 指标（关注语法正确性，如 parsableSyntax）。
*   **分析手段:** 将模型按规模分为四组（tiny: 0-3B, small: 3-8B, medium: 8-33B, large: 33-72B），使用 Kruskal-Wallis 测试和 Dunn 测试分析规模组间的性能差异，并通过可视化手段观察模型家族内部和跨家族的性能趋势。
*   **特殊考虑:** 特别分析了 Mixture-of-Experts (MoE) 模型和代码专项模型的表现，以探讨模型架构和领域微调对性能的影响。

## Experiment

*   **总体效果:** 实验结果表明，模型规模越大，性能通常越好，但存在明显的 plateau 和 ceiling 效应，尤其在 medium (8B-33B) 规模模型上，部分任务性能已接近上限（如 RdfConnectionExplain 任务在 large 组达到 0.93）。
*   **任务差异:** 某些任务（如 RdfFriendCount 和 Text2Sparql）即使是大模型也表现不佳，平均得分较低（例如 Text2Sparql turtle schema 仅 0.13），表明任务复杂性可能超出当前模型能力。
*   **统计显著性:** Kruskal-Wallis 测试显示所有任务变体在不同规模组间存在显著差异（p < 0.001），但 Dunn 测试表明 medium 和 large 组在多个任务上无显著差异，验证了 plateau 效应。
*   **家族内部表现:** 同一模型家族内，偶尔出现较大模型性能低于较小模型的情况，但这种下降通常是局部的，下一规模模型往往恢复或提升性能。
*   **特殊模型表现:** 代码专项模型（如 Deepseek-Coder 33B）在 RDF 和 SPARQL 输出任务上表现较优，而 MoE 模型（如 Qwen2-57B）性能与同规模常规模型相当，未显示出成本效益优势。
*   **实验设置合理性:** 实验覆盖了多种任务类型和模型规模，重复执行 50 次以减少随机性，设置较为全面；但任务提示未优化，且未包含超过 80B 的模型，可能限制了对更大规模模型表现的评估。

## Further Thoughts

论文中的 plateau 和 ceiling 效应启发了我，是否可以通过选择中等规模模型（8B-14B）来优化成本效益比，尤其是在性能接近上限的任务上？此外，代码专项模型在 RDF 和 SPARQL 任务上的优异表现提示，领域特定微调可能是提升 KGE 任务性能的关键方向。另一个想法是，是否可以通过任务分解或引入外部知识库（如预训练的 KG 嵌入），进一步提升小模型在复杂任务（如 Text2Sparql）上的表现？最后，MoE 模型未展现明显优势，是否可以通过结合 MoE 和常规模型的混合架构，在推理时动态调整活跃参数，以实现性能和成本的更好平衡？