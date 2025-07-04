---
title: "From Sentences to Sequences: Rethinking Languages in Biological System"
pubDatetime: 2025-07-01T16:57:39+00:00
slug: "2025-07-biological-language-modeling"
type: "arxiv"
id: "2507.00953"
score: 0.5567391898864802
author: "grok-3-latest"
authors: ["Ke Liu", "Shuaike Shen", "Hao Chen"]
tags: ["LLM", "Biological Sequences", "Inverse Folding", "Stochastic Generation", "Structural Semantics"]
institution: ["Zhejiang University"]
description: "本文提出随机顺序生成范式，通过动态调整生物序列生成顺序，显著提升了 RNA 和蛋白质逆折叠任务中的序列与结构恢复效果，并倡导基于结构的评估体系，为生物语言建模提供了新思路。"
---

> **Summary:** 本文提出随机顺序生成范式，通过动态调整生物序列生成顺序，显著提升了 RNA 和蛋白质逆折叠任务中的序列与结构恢复效果，并倡导基于结构的评估体系，为生物语言建模提供了新思路。 

> **Keywords:** LLM, Biological Sequences, Inverse Folding, Stochastic Generation, Structural Semantics

**Authors:** Ke Liu, Shuaike Shen, Hao Chen

**Institution(s):** Zhejiang University


## Problem Background

大型语言模型（LLMs）在自然语言处理（NLP）中的成功范式被迁移到生物序列建模（如蛋白质、RNA、DNA），但自然语言与生物语言在结构相关性和语义表达上存在根本差异：生物序列具有更强的长距离依赖性（如 RNA 碱基配对、蛋白质氢键网络），且其语义直接体现在物理三维结构中，而非抽象概念。这种差异导致传统的自回归生成范式和 NLP 评估指标（如 BLEU、ROUGE）不适用于生物序列建模。本文以逆折叠问题（Inverse Folding）为案例，研究如何设计更适合生物语言的生成范式和评估方法。

## Method

*   **核心思想:** 针对生物序列的长距离依赖性和结构语义特性，提出随机顺序生成（Stochastic-Order Generation）范式，替代传统的顺序自回归生成（Sequential-Order Generation），以更好地捕捉序列内部的复杂依赖关系；同时，构建基于结构的评估体系，强调结构保真度而非单纯的序列相似性。
*   **生成范式:** 
    *   随机顺序生成允许在生成过程中以任意位置生成 token，而非严格从左到右顺序生成。具体实现中，通过每次基于置信度选择生成位置，动态调整生成顺序，从而优先处理具有强依赖关系的 token（如 RNA 中的碱基配对）。
    *   结合束搜索（Beam Search）进一步优化生成结果，包括基于解码位置（Decode Position-Based）和基于类型（Decode Type-Based）的束搜索策略，扩展搜索空间并提高生成质量。
*   **模型实现:** 
    *   提出 RiFold 模型，针对 RNA 逆折叠任务，采用与 RDesign 相同的特征提取器（Featurizer），结合编码器和自回归解码器，利用节点属性（Node Attributes，如二面角、空间距离）和边属性（Edge Attributes，如相对旋转、方向向量）捕捉局部和全局几何关系。
    *   对于蛋白质逆折叠，基于 PiFold 框架实现自回归解码器（PiFold-AR），验证随机顺序生成在不同生物分子上的适用性。
*   **评估体系:** 
    *   提出结构感知的评估流程，结合结构恢复指标（TM-score、RMSD）、能量稳定性（Energy）和序列恢复指标（NSR、Macro-F1），以更全面地衡量生物序列的语义一致性，而非仅依赖序列匹配。
*   **关键点:** 不改变模型架构，仅通过生成顺序的调整即可显著提升性能，同时强调评估指标需与生物学语义（3D 结构）对齐。

## Experiment

*   **有效性:** 在 RNA 逆折叠任务中，RiFold 模型在 RNAsolo 数据集上相比 SOTA 模型 RDesign，序列恢复（NSR）提升了 3.64%（从 41.53% 到 43.04%），Macro-F1 提升了 5.57%（从 40.89% 到 43.17%）；结构恢复方面，TM-score 提升了 13.88%，RMSD 提升了 5.86%，且 60.22% 的预测序列能量低于 RDesign，表明结构稳定性更优。在蛋白质逆折叠任务（CATH 数据集）中，自回归方法（如 PiFold-AR）在结构恢复上优于非自回归方法，验证了随机顺序生成的适用性。
*   **全面性与合理性:** 实验覆盖 RNA（RNAsolo、RNA-Puzzles）和蛋白质（CATH）两种生物分子，评估指标包括序列恢复（NSR、Macro-F1、Perplexity）和结构恢复（TM-score、RMSD、Energy），数据分割和对比方法（如 RDesign、PiFold）选择合理，实验设计较为全面。消融研究进一步验证了随机顺序解码和束搜索的有效性，尤其在短 RNA 序列上提升显著。
*   **局限性:** 随机顺序生成计算成本较高，论文未提供详细开销对比；结构恢复评估依赖现有预测工具（如 ESMFold、E2EFold），可能引入偏差；此外，序列恢复与结构恢复相关但不一致（Pearson 相关系数为 0.6302），提示传统序列指标的局限性。

## Further Thoughts

随机顺序生成的思路启发我们可以在生成任务中动态优化生成顺序，而不仅仅依赖固定顺序，是否可以通过强化学习或图神经网络进一步优化生成顺序的选择策略？此外，生物序列的语义直接 grounding 在物理结构上，这一特性是否可以推广到其他领域（如材料科学），通过引入领域知识（如热力学、量子化学）来指导生成模型设计？最后，序列与结构恢复的不一致性提示未来评估体系可能需要多层次指标，既关注低层次匹配，也关注高层次功能一致性。