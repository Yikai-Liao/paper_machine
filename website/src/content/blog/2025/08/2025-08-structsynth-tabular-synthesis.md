---
title: "StructSynth: Leveraging LLMs for Structure-Aware Tabular Data Synthesis in Low-Data Regimes"
pubDatetime: 2025-08-04T16:55:02+00:00
slug: "2025-08-structsynth-tabular-synthesis"
type: "arxiv"
id: "2508.02601"
score: 0.42002802976526943
author: "grok-3-latest"
authors: ["Siyi Liu", "Yujia Zheng", "Yongqi Zhang"]
tags: ["LLM", "Tabular Data", "Data Synthesis", "Structure Learning", "Low-Data Regime"]
institution: ["The Hong Kong University of Science and Technology (Guangzhou)", "Carnegie Mellon University"]
description: "StructSynth 提出了一种两阶段框架，通过 LLM 驱动的依赖结构发现和结构引导合成，在低数据场景下生成高质量表格数据，显著提升下游任务性能并平衡隐私与保真度。"
---

> **Summary:** StructSynth 提出了一种两阶段框架，通过 LLM 驱动的依赖结构发现和结构引导合成，在低数据场景下生成高质量表格数据，显著提升下游任务性能并平衡隐私与保真度。 

> **Keywords:** LLM, Tabular Data, Data Synthesis, Structure Learning, Low-Data Regime

**Authors:** Siyi Liu, Yujia Zheng, Yongqi Zhang

**Institution(s):** The Hong Kong University of Science and Technology (Guangzhou), Carnegie Mellon University


## Problem Background

表格数据在医疗、金融、教育等领域广泛应用，但数据稀缺性（尤其在罕见病、新兴领域等低数据场景）限制了机器学习模型的训练和应用。
传统生成模型（如变分自编码器、生成对抗网络）在数据稀缺时难以捕捉复杂特征依赖，而大型语言模型（LLMs）虽具备强大生成能力，但因依赖线性化文本输入，忽略表格数据的显式依赖结构，导致生成数据保真度不足。
论文旨在解决两个关键问题：如何在数据有限时可靠地发现依赖结构，以及如何利用 LLMs 生成符合该结构的高质量合成数据。

## Method

*   **核心思想:** 提出 StructSynth，一个两阶段框架，通过解耦结构发现与数据生成，结合 LLMs 的生成能力与显式结构控制，确保合成表格数据在低数据场景下具有高保真度和结构一致性。
*   **第一阶段 - 依赖结构发现 (Dependency Structure Discovery):** 
    *   利用 LLM 通过迭代的广度优先搜索（BFS）方法，从有限训练数据中推断特征间的依赖关系，并表示为有向无环图（DAG）。
    *   具体步骤包括：初始化源节点（不依赖其他属性的特征）；基于统计关联分数（如 Pearson 相关系数、Cramér’s V）扩展图结构，提出新的依赖边及推理依据；通过 LLM 解决图中可能出现的环路，确保 DAG 性质。
    *   这一阶段结合 LLM 的推理能力和统计指标作为弱监督信号，提升结构发现的可靠性。
*   **第二阶段 - 结构引导合成 (Structure-Guided Synthesis):** 
    *   利用学习到的 DAG 作为蓝图，指导 LLM 按拓扑顺序自回归生成合成数据，确保每个特征的生成条件依赖于其父节点，保持结构一致性。
    *   对于不包含在 DAG 中的独立特征，在单独步骤中生成，条件依赖于已生成的图结构特征值。
    *   最终通过拼接图结构依赖值和独立值，生成完整的合成数据点。
*   **关键创新:** 解耦结构学习与数据生成，通过显式 DAG 约束 LLM 的生成过程，弥补 LLM 对复杂依赖结构的感知不足，同时避免传统生成模型在低数据场景下的性能下降。

## Experiment

*   **有效性:** StructSynth 在六个真实世界数据集（涵盖社会、医疗、商业领域）上的下游模型性能（AUC/R²）显著优于所有基线方法，平均得分为 75.01，优于最强竞争对手 CLLM 的 73.36 和原始训练数据的 71.54，表明其生成的合成数据能有效提升下游任务性能。
*   **隐私与保真度权衡:** 在隐私保护（平均排名 1.33）和统计保真度（平均排名 5.33）上均表现优异，优于 GReaT（高保真但隐私泄露严重）和 CLLM（隐私好但保真度较低），成功平衡了两者矛盾。
*   **消融研究:** 消融实验验证了结构引导和 LLM 生成的协同作用，移除结构引导导致 AUC 下降 1.6 点，用传统贝叶斯采样器替代 LLM 下降 4.4 点，证明两者的不可或缺性。
*   **数据效率与通用性:** 在极低样本量（n=20）下仍保持高性能，且对不同 LLM（如 Qwen-2.5、GPT-4o）均有提升，显示出鲁棒性和通用性。
*   **实验设置合理性:** 实验涵盖多种领域和任务类型（分类、回归），指标设计全面（实用性、统计特性、隐私），低数据场景模拟合理（n=100 生成 1000 样本），重复 10 次确保结果稳健。
*   **局限性:** 未深入探讨 LLM 推理幻觉对结构发现的影响，也未详细分析计算成本（多次 LLM 查询的开销）。

## Further Thoughts

StructSynth 的显式结构约束思想可扩展到其他结构化数据（如知识图谱、时间序列）或非结构化数据（如文本生成中的逻辑一致性约束）；其结合统计信号为 LLM 提供弱监督的方式，启发我们在因果推理或异常检测中利用传统方法减少 LLM 推理偏差；此外，是否可设计动态结构调整机制，让 LLM 在生成过程中根据新数据反馈更新依赖结构，进一步提升适应性？