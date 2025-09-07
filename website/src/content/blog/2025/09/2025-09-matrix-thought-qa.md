---
title: "MTQA:Matrix of Thought for Enhanced Reasoning in Complex Question Answering"
pubDatetime: 2025-09-04T06:13:28+00:00
slug: "2025-09-matrix-thought-qa"
type: "arxiv"
id: "2509.03918"
score: 0.7116284124304993
author: "grok-3-latest"
authors: ["Fengxiao Tang", "Yufeng Li", "Zongzong Wu", "Ming Zhao"]
tags: ["LLM", "Reasoning", "Retrieval-Augmented Generation", "Knowledge Graph", "Question Answering"]
institution: ["Central South University"]
description: "本文提出 Matrix of Thought (MoT) 推理范式，通过矩阵结构和列-单元通信机制，结合优化的知识单元，显著提升大型语言模型在复杂问答任务中的推理能力和效率。"
---

> **Summary:** 本文提出 Matrix of Thought (MoT) 推理范式，通过矩阵结构和列-单元通信机制，结合优化的知识单元，显著提升大型语言模型在复杂问答任务中的推理能力和效率。 

> **Keywords:** LLM, Reasoning, Retrieval-Augmented Generation, Knowledge Graph, Question Answering

**Authors:** Fengxiao Tang, Yufeng Li, Zongzong Wu, Ming Zhao

**Institution(s):** Central South University


## Problem Background

大型语言模型（LLM）在复杂问答（QA）任务中因推理能力不足而表现不佳，尤其是在涉及多实体和多跳信息的抽象问题上。
现有方法如 Chain-of-Thought (CoT) 受限于单一推理路径，Tree-of-Thought (ToT) 存在层内冗余和效率低下问题，而检索增强生成（RAG）方法在处理复杂知识时常引入无关或错误信息，误导推理。
本文旨在设计一种高效、低冗余的多分支推理结构，并优化外部知识的表示和利用方式，以提升 LLM 在复杂问答中的准确性和可靠性。

## Method

*   **核心思想**：提出 Matrix of Thought (MoT) 推理范式，通过矩阵结构组织推理过程，结合列-单元通信机制，实现横向多策略和纵向深层次推理，减少冗余，同时优化知识表示以辅助推理。
*   **知识库构建与检索增强**：从输入文档中提取实体和关系，构建知识图谱（KG），并结合原始文本形成知识单元（Knowledge Units），作为推理的辅助知识；知识单元同时支持多跳推理（通过结构化信息）和语义细节保留（通过原始文本），更符合 LLM 偏好。
*   **矩阵推理结构**：将推理过程组织为矩阵，行代表不同推理策略（广度探索），列代表推理深度（纵向挖掘）；每个推理节点（Thought Node）基于前一节点的部分内容生成新策略，避免重复。
*   **列-单元通信机制**：通过通信权重（Communication Weight）控制前一节点对后一节点的影响，早期抑制历史信息以激发多样性，后期增加影响以避免遗忘早期结论，促进多分支推理。
*   **总结与优化**：每列推理节点生成后，通过检索增强的知识单元进行事实校正和总结，生成总结节点（Summary Node），作为下一列推理基础，逐步逼近最优答案。
*   **框架实现**：基于 MoT 构建 MTQA 框架，集成上述模块，形成自动化问答流程，兼顾推理广度和深度。

## Experiment

*   **性能提升**：在多个数据集（NaturalQuestions, HotpotQA, 2WikiMultihopQA, UltraDomain 子集）上，MTQA 的 F1 和 EM 分数相较基线方法（包括 CoT, ToT, 多种 RAG 变体）提升 3%-9%，尤其在多跳问答任务中表现突出，表明其处理复杂推理任务的能力显著增强。
*   **效率优势**：推理时间仅为最佳基线 RATT 的 1/7（约 3.2 分钟 vs 22 分钟），理论时间复杂度分析支持这一结论，显示 MTQA 在计算开销上的优化效果。
*   **实验设置合理性**：实验涵盖单跳和多跳问答任务，采用多指标（F1, EM）和多维度胜率评估（Comprehensiveness, Accuracy, Empowerment, Overall），与多种基线对比，数据支持结论可靠。
*   **消融研究**：移除 RAG 校正、知识图谱增强、原始文本增强和通信机制等模块后，性能均下降，其中通信机制影响最大，验证了矩阵结构和通信策略的核心作用。
*   **参数分析**：通信权重矩阵和矩阵大小的调参实验表明，3x4 矩阵结合 Vert&Hor-0.1 权重配置是性能与效率的最佳平衡点，过大矩阵导致时间成本激增而性能提升有限。

## Further Thoughts

MoT 作为一种新的推理结构，不仅适用于问答任务，还可能扩展到其他需要多维度推理的领域，如多模态任务或模型对齐，启发我们探索更灵活的推理范式；
知识单元将结构化知识与非结构化文本结合的做法，提示在 RAG 系统中进一步研究知识表示的最优形式，以适配不同任务或模型偏好；
列-单元通信机制通过动态调整权重控制信息流动，这种思想可应用于其他需要平衡探索与利用的场景，如强化学习或自适应推理。