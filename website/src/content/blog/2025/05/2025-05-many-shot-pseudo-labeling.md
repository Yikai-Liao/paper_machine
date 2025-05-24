---
title: "MAPLE: Many-Shot Adaptive Pseudo-Labeling for In-Context Learning"
pubDatetime: 2025-05-22T04:54:27+00:00
slug: "2025-05-many-shot-pseudo-labeling"
type: "arxiv"
id: "2505.16225"
score: 0.4885916346329382
author: "grok-3-latest"
authors: ["Cong Shen"]
tags: ["LLM", "In-Context Learning", "Pseudo-Labeling", "Sample Selection", "Reasoning"]
institution: ["University of Virginia", "Arizona State University"]
description: "本文提出 MAPLE 框架，通过基于影响力的伪标注和自适应示例选择，在资源受限场景下显著提升了许多示例上下文学习的性能。"
---

> **Summary:** 本文提出 MAPLE 框架，通过基于影响力的伪标注和自适应示例选择，在资源受限场景下显著提升了许多示例上下文学习的性能。 

> **Keywords:** LLM, In-Context Learning, Pseudo-Labeling, Sample Selection, Reasoning

**Authors:** Cong Shen

**Institution(s):** University of Virginia, Arizona State University


## Problem Background

大型语言模型（LLMs）的上下文学习（In-Context Learning, ICL）通过在输入中加入多个示例（demonstrations）来处理多样化任务，尤其是在上下文窗口扩展后，许多示例（many-shot ICL）显著提升了复杂任务的性能。
然而，获取大量标注示例的高成本限制了其应用，尤其在资源受限场景下手动标注昂贵且不可行，因此需要一种方法减少对标注数据的依赖，同时保持或提升模型性能。

## Method

*   **核心思想:** 提出 MAPLE（Many-Shot Adaptive Pseudo-Labeling）框架，通过伪标注未标注样本并自适应选择示例，在标注数据有限的情况下增强许多示例ICL的性能。
*   **具体实现:** 
    *   **影响力的样本选择与伪标注:** 构建一个包含标注和未标注样本的图（labeled-unlabeled graph），利用节点影响力（node influence）概念，通过计算最短路径距离和路径数量的几何平均值，识别对标注样本最具影响的未标注样本。随后，使用语言模型对这些样本进行伪标注，形成候选示例池（candidate demonstration set）。
    *   **自适应示例选择:** 针对每个测试查询，构建一个伪标注图（pseudo-labeled graph），再次基于影响力计算，从候选示例池中选择与测试查询最相关的标注和伪标注示例作为输入，避免无关示例引入噪声。
*   **关键创新:** 利用图结构和影响力机制，确保伪标注样本和示例选择的高质量，同时控制计算成本（如图构建和路径计算仅需执行一次），适用于资源受限场景。

## Experiment

*   **有效性:** MAPLE 在八个真实世界数据集（涵盖总结、推理、分类、问答任务）上均优于基线方法（如随机选择、RAG），尤其在复杂任务（如 Banking77、Date、GPQA）上表现突出，准确率提升显著（如 Banking77 任务平均提升 1.5%-1.7%）。
*   **优越性与可扩展性:** 随着伪标注样本数量从 20 增加到 250，MAPLE 性能持续提升，展现出在许多示例场景下的可扩展性；同时在不同模型（如 Gemini 1.5 Flash 和 Pro）上均表现稳健，表明方法适应性强。
*   **实验设置合理性:** 实验设计全面，涵盖多种任务、模型和伪标注样本数量，对比多种基线，并通过消融实验验证了方法组件（如影响力分数、编码器选择）的有效性；数据集选择具有代表性，任务类型多样。
*   **局限性:** 在某些数据集（如 Tracking7、XSum）上，增加伪标注样本未带来持续提升，可能是伪标注质量或噪声影响，提示方法在特定任务上的适用性需进一步优化。

## Further Thoughts

基于影响力的样本选择机制启发了我，可以将图结构和节点影响力的概念扩展到其他领域，如主动学习或数据增强中选择最具代表性的样本；此外，自适应示例选择的思路提示我们可以在推理时根据查询特性动态调整输入，未来可以探索结合多模型或迭代优化的伪标注策略，进一步提升伪标注质量和ICL性能。