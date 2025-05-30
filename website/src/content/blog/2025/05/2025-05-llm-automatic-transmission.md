---
title: "Automatic Transmission for LLM Tiers: Optimizing Cost and Accuracy in Large Language Models"
pubDatetime: 2025-05-27T09:11:00+00:00
slug: "2025-05-llm-automatic-transmission"
type: "arxiv"
id: "2505.20921"
score: 0.6684869658338615
author: "grok-3-latest"
authors: ["Injae Na", "Keonwoong Noh", "Woohwan Jung"]
tags: ["LLM", "Cost Optimization", "Model Selection", "Accuracy Estimation", "Iterative Refinement"]
institution: ["Hanyang University"]
description: "本文提出 LLM-AT 框架，通过无需训练的动态模型层级选择机制，显著优化了大型语言模型在成本与准确性之间的权衡，为实际应用提供了高效解决方案。"
---

> **Summary:** 本文提出 LLM-AT 框架，通过无需训练的动态模型层级选择机制，显著优化了大型语言模型在成本与准确性之间的权衡，为实际应用提供了高效解决方案。 

> **Keywords:** LLM, Cost Optimization, Model Selection, Accuracy Estimation, Iterative Refinement

**Authors:** Injae Na, Keonwoong Noh, Woohwan Jung

**Institution(s):** Hanyang University


## Problem Background

大型语言模型（LLMs）服务提供商通常提供多个性能和价格不同的模型层级（tiers），而随着自然语言处理（NLP）任务日益复杂并模块化为多个子任务，如何为每个子任务选择合适的模型层级以平衡成本和准确性成为关键挑战。
现有基于训练的模型选择方法需要大量标注数据、面对新模型时需重新训练，且在分布外数据上泛化能力有限，导致成本高昂或性能不足。

## Method

*   **核心思想:** 提出 LLM-AT（LLM Automatic Transmission）框架，通过动态选择模型层级，在无需训练的情况下优化成本与性能的权衡。
*   **框架组成:** 包含三个主要模块：
    *   **Starter（启动器）:** 负责选择初始模型层级，利用‘accuracy estimator’（准确性估计器）基于历史推理记录（History）中相似问题的伪标签（pseudo-labeling），估计每个模型层级对当前问题的准确性，选择成本最低且预计能正确回答的层级。
    *   **Generator（生成器）:** 使用选定层级的模型生成回答，采用通用的提示策略如 Chain-of-Thought（CoT）和 Program-of-Thought（PoT）以减少模型依赖性。
    *   **Judge（评判器）:** 使用与 Generator 相同层级的模型评估回答的有效性，若回答无效则升级到更高层级模型，重复生成和评估过程，直到获得有效回答或达到最高层级。
*   **创新机制:** 
    *   准确性估计器通过历史记录中 top-k 相似问题的正确率（基于嵌入向量的余弦相似度）计算估计准确性，并结合基准性能作为平滑因子，避免训练开销。
    *   为最低层级模型提供‘abstain’（弃权）选项，若模型对问题无把握则直接升级到更高层级，减少幻觉风险和成本。
*   **特点:** 不依赖特定模型提示，模块化设计易于扩展，迭代升级机制确保性能，同时控制成本。

## Experiment

*   **有效性:** 在 MATH 数据集上，LLM-AT 相比单一 o1-mini 迭代方法，执行时间减少 28.24%（123.73 → 88.79 分钟），成本减少 59.37%（$41.56 → $16.89）；在 MCQA 数据集上，时间减少 59.34%（230.54 → 93.69 分钟），成本比单一 o1 模型低 88.01%（$59.52 → $7.14），同时保持了接近最高层级模型的准确性。
*   **全面性:** 实验覆盖从简单到研究生水平的多种难度问题（MATH 和 MCQA 数据集），验证了准确性估计器的估计值与实际准确性趋势一致，证明其可靠性。
*   **鲁棒性:** 系统对历史数据量和质量表现出鲁棒性，即使在冷启动场景下也能快速提升性能；对模型层级性能反转（低层级模型在某些问题上优于高层级）也表现出适应性。
*   **开销分析:** Starter 和 Judge 模块的开销远低于 Generator（Starter 比 Generator 快 44.6-125 倍，Judge 快约 2.5 倍），且大多数问题在初始层级即可解决，减少了层级过渡次数，整体系统效率高。
*   **合理性:** 实验设置考虑了不同任务难度、历史数据积累、以及模块间协作，相比单一模型和迭代基线方法，LLM-AT 在准确性-成本和准确性-时间权衡上均表现出显著优势。

## Further Thoughts

LLM-AT 框架无需训练的动态选择机制为其他 AI 资源分配问题提供了启发，例如可以探索更复杂的相似性度量或上下文信息来提升准确性估计；Abstain 选项的应用可以在多模型协作中进一步优化，避免低性能模型的无效推理；此外，模块化设计也启发我们思考如何将类似框架扩展到异构模型系统（如开源与专有模型混合），以实现更广泛的成本-性能优化。