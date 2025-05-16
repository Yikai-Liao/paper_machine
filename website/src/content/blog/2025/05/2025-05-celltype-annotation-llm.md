---
title: "CellTypeAgent: Trustworthy cell type annotation with Large Language Models"
pubDatetime: 2025-05-13T14:34:11+00:00
slug: "2025-05-celltype-annotation-llm"
type: "arxiv"
id: "2505.08844"
score: 0.4946182556606551
author: "grok-3-latest"
authors: ["Jiawen Chen", "Jianghao Zhang", "Huaxiu Yao", "Yun Li"]
tags: ["LLM", "Cell Type Annotation", "Trustworthy AI", "Database Integration", "Single-Cell Analysis"]
institution: ["University of North Carolina at Chapel Hill"]
description: "CellTypeAgent通过结合大型语言模型推理和数据库验证，显著提升了单细胞RNA测序中细胞类型注释的准确性和可信度。"
---

> **Summary:** CellTypeAgent通过结合大型语言模型推理和数据库验证，显著提升了单细胞RNA测序中细胞类型注释的准确性和可信度。 

> **Keywords:** LLM, Cell Type Annotation, Trustworthy AI, Database Integration, Single-Cell Analysis

**Authors:** Jiawen Chen, Jianghao Zhang, Huaxiu Yao, Yun Li

**Institution(s):** University of North Carolina at Chapel Hill


## Problem Background

单细胞RNA测序（scRNA-seq）分析中的细胞类型注释是关键步骤，但传统手动方法依赖标记基因与文献比对，耗时且劳动密集。
近年来，大型语言模型（LLMs）如GPT展现了自动化注释的潜力，然而其‘幻觉’问题（生成不准确或虚构信息）限制了在生物医学领域的可靠性。
本文旨在解决LLM在细胞类型注释中的准确性和可信度问题，通过结合LLM的推理能力与数据库验证，确保高效且可靠的注释结果。

## Method

*   **核心框架:** CellTypeAgent是一个可信的LLM代理工具，采用两阶段方法来实现细胞类型注释。
*   **第一阶段 - LLM候选预测:** 利用大型语言模型（如GPT-4或开源模型Deepseek-R1），根据输入的标记基因、组织类型和物种信息，通过精心设计的提示（如‘基于以下标记基因识别最可能的前3种细胞类型’），生成一组有序的候选细胞类型，通常取前3名。
*   **第二阶段 - 基因表达验证:** 利用CellxGene数据库中的基因表达数据（包括表达值和表达比例），对LLM生成的候选细胞类型进行评分。评分函数综合考虑LLM的初始排名、基因在特定细胞类型中的表达模式（针对特定组织或跨组织），最终选择得分最高的细胞类型作为注释结果。
*   **可选工具:** 尝试使用文献搜索（如LitSense）和基因总结（如NCBI Gene数据库）作为补充信息输入LLM，但效果不佳，可能是由于文献数据结构化不足或LLM对外部文本的过度依赖。
*   **隐私与灵活性:** 支持开源LLM以解决闭源模型（如ChatGPT）带来的数据隐私问题，同时评分函数设计考虑了组织类型已知和未知两种情况，增强了方法的适用性。

## Experiment

*   **数据集与评估:** 在9个真实数据集上进行测试，涵盖303种细胞类型和36种组织，使用一致性分数（Agreement Score）作为评估指标，与GPTCelltype、单独使用CellxGene和PanglaoDB等方法对比。
*   **性能表现:** CellTypeAgent在所有数据集上均显著优于其他方法，尤其在使用较弱LLM模型时，通过数据库验证带来的性能提升更为明显（如Deepseek-R1性能提升5.1%，接近GPT-4o在GPTCelltype上的表现）。
*   **全面性分析:** 实验考虑了多种变量的影响，包括不同基础LLM模型（o1-preview表现最佳）、候选细胞类型数量（3个候选时性能略优）、标记基因数量（更多基因提升性能）以及混合细胞类型的处理（仍能准确识别多种成分，尽管性能略低于纯细胞类型）。
*   **额外测试:** 文献搜索和基因总结作为补充信息的尝试未显著提升性能，甚至因LLM对输入文本的过度依赖而降低效果，同时增加了计算成本。
*   **合理性:** 实验设置全面，数据集覆盖广泛，指标清晰，验证了方法的有效性和数据库验证对缓解LLM幻觉的重要性。

## Further Thoughts

CellTypeAgent的‘LLM推理+数据库验证’混合策略可推广至其他生物信息学任务或医学诊断领域，以解决LLM幻觉问题；
使用开源LLM结合外部知识源（如数据库）不仅提升性能，还能应对数据隐私问题，值得在高敏感性领域进一步探索；
此外，候选细胞类型数量对性能有轻微影响，未来可研究基于输入复杂性或模型置信度的动态候选数量调整策略，以进一步优化注释精度。