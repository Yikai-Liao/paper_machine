---
title: "LTL$_f$ Learning Meets Boolean Set Cover"
pubDatetime: 2025-09-29T11:20:20+00:00
slug: "2025-09-ltlf-boolean-cover"
type: "arxiv"
id: "2509.24616"
score: 0.6654455246088847
author: "grok-3-latest"
authors: ["Gabriel Bathie", "Nathanaël Fijalkow", "Théo Matricon", "Baptiste Mouillon", "Pierre Vandenhove"]
tags: ["LTL Learning", "Boolean Set Cover", "Temporal Logic", "Specification Mining", "Combinatorial Search"]
institution: ["LaBRI, Université de Bordeaux, France", "DIENS, Paris, France", "CNRS, France", "Univ Rennes, Inria, IRISA, Rennes, France", "UMONS – Université de Mons, Belgium"]
description: "本文提出了一种结合 LTL_f 学习与 Boolean Set Cover 的框架，通过分层处理布尔运算和时态运算，实现工具 Bolt，在 CPU 环境下显著提升学习效率和公式规模的平衡。"
---

> **Summary:** 本文提出了一种结合 LTL_f 学习与 Boolean Set Cover 的框架，通过分层处理布尔运算和时态运算，实现工具 Bolt，在 CPU 环境下显著提升学习效率和公式规模的平衡。 

> **Keywords:** LTL Learning, Boolean Set Cover, Temporal Logic, Specification Mining, Combinatorial Search

**Authors:** Gabriel Bathie, Nathanaël Fijalkow, Théo Matricon, Baptiste Mouillon, Pierre Vandenhove

**Institution(s):** LaBRI, Université de Bordeaux, France, DIENS, Paris, France, CNRS, France, Univ Rennes, Inria, IRISA, Rennes, France, UMONS – Université de Mons, Belgium


## Problem Background

从有限轨迹中学习线性时态逻辑（LTL_f）公式是一个基础性研究问题，广泛应用于人工智能、软件工程、形式化方法和机器人学等领域。
核心挑战在于给定正例和反例轨迹，找到一个最小的 LTL_f 公式以区分两者，而现有方法在计算复杂性和公式规模上存在瓶颈，尤其是在工业规模问题中，学习效率和公式大小难以平衡。

## Method

*   **核心思想:** 提出一种结合 LTL_f 学习与 Boolean Set Cover 的框架，通过分层处理布尔运算和时态运算，突破传统枚举方法的规模限制。
*   **具体实现:** 
    *   **第一阶段 - VFB 算法有限枚举:** 使用 VFB 算法（基于组合搜索、快速评估和观察等价性）枚举小规模 LTL_f 公式（大小限制为超参数 LTL2BS-switch，通常为 8），生成基础公式集，并通过特征表快速评估公式是否满足轨迹，同时利用观察等价性减少冗余。
    *   **第二阶段 - Boolean Set Cover 子程序:** 当枚举达到限制时，将问题转化为 Boolean Set Cover，即通过布尔运算（与、或）组合已生成公式，寻找区分正反例的组合公式。采用基于束搜索（beam search）的贪心算法，结合支配（domination）减少冗余公式和分治（divide-and-conquer）策略处理复杂任务。此外，通过将特征表简化为特征向量（仅关注首比特），显著降低内存占用并加速评估。
*   **关键创新:** 将布尔运算与时态运算分开处理，利用 Boolean Set Cover 的近似求解能力，在不牺牲完整性的前提下提升效率，并实现了一个名为 Bolt 的工具。

## Experiment

*   **有效性:** Bolt 在超过 15,000 个 LTL_f 学习任务的基准测试集中表现出显著优势，解决了 14,374 个任务（相比 Scarlet 的 11,468 个），平均运行时间从 4.21 秒降至 2.05 秒，平均公式大小比例从 1.15 降至 1.01。
*   **优越性:** 在 70% 的基准测试中，Bolt 比 Scarlet 快 100 倍以上，在 98% 的情况下生成的公式大小相等或更小；与 GPU 算法相比，Bolt 在 CPU 环境下具有竞争力，尤其在小任务上表现更优。
*   **消融研究:** 移除 Boolean Set Cover 后，Bolt 解决任务数显著减少，表明该子程序对处理困难任务至关重要。
*   **实验设置合理性:** 基准测试集覆盖多种任务难度和类型，参数设置（如轨迹长度、原子命题数量）合理，实验在 Grid’5000 集群上运行，硬件配置一致，确保结果可重复性。

## Further Thoughts

Boolean Set Cover 的思想不仅适用于 LTL_f 学习，还可能推广到其他逻辑或规范语言（如正则表达式、布尔电路）的学习问题中，启发我们思考是否可以将分层策略（先生成基础单元，再通过组合优化）应用于大型语言模型的规范学习；此外，探索 Boolean Set Cover 在 GPU 环境下的并行化潜力，或结合强化学习优化组合策略，可能进一步提升效率。