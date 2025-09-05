---
title: "epiGPTope: A machine learning-based epitope generator and classifier"
pubDatetime: 2025-09-03T14:36:06+00:00
slug: "2025-09-epigptope-epitope-design"
type: "arxiv"
id: "2509.03351"
score: 0.3868141429755323
author: "grok-3-latest"
authors: ["Natalia Flechas Manrique", "Alberto Martínez", "Elena López-Martínez", "Luc Andrea", "Román Orus", "Aitor Manteca", "Aitziber L. Cortajarena", "Llorenç Espinosa-Portalés"]
tags: ["LLM", "Epitope Design", "Sequence Generation", "Classification", "Protein Modeling"]
institution: ["Multiverse Computing", "Centre for Cooperative Research in Biomaterials (CIC biomaGUNE)", "Donostia International Physics Center", "Ikerbasque Foundation for Science"]
description: "本文提出了一种基于大型语言模型的表位生成与分类方法 epiGPTope，通过生成符合自然表位统计特征的候选序列并结合分类器筛选特定来源表位，显著降低了实验成本并加速了表位发现过程。"
---

> **Summary:** 本文提出了一种基于大型语言模型的表位生成与分类方法 epiGPTope，通过生成符合自然表位统计特征的候选序列并结合分类器筛选特定来源表位，显著降低了实验成本并加速了表位发现过程。 

> **Keywords:** LLM, Epitope Design, Sequence Generation, Classification, Protein Modeling

**Authors:** Natalia Flechas Manrique, Alberto Martínez, Elena López-Martínez, Luc Andrea, Román Orus, Aitor Manteca, Aitziber L. Cortajarena, Llorenç Espinosa-Portalés

**Institution(s):** Multiverse Computing, Centre for Cooperative Research in Biomaterials (CIC biomaGUNE), Donostia International Physics Center, Ikerbasque Foundation for Science


## Problem Background

表位（epitope）是免疫系统识别的关键抗原片段，在疫苗、免疫疗法和诊断工具开发中至关重要，但由于序列空间的组合爆炸性（长度为 n 的线性表位有 20^n 种可能），通过实验筛选所有候选序列不可行。
论文旨在通过计算方法生成具有生物学可行性的表位候选序列，并筛选出特定来源（如细菌或病毒）的表位，以降低实验成本并加速发现过程。

## Method

*   **生成模型（epiGPTope）:** 基于预训练的大型语言模型 ProtGPT2，通过在 Immune Epitope Database (IEDB) 的线性表位数据上进行微调，采用无监督学习方式，最小化负对数似然损失，学习表位序列的统计分布，从而生成符合表位特征的新序列；生成过程中调整温度（temperature）和重复惩罚（repetition penalty）等超参数，以平衡生成序列的多样性和质量。
*   **分类模型:** 训练两类分类器以筛选生成的序列：一是基于集成方法（ensemble-based，使用 XGBoost 结合向量嵌入），二是基于大型语言模型的分类器（如 ProtGPT2 和 ProtBERT 的微调版本）；分类器在 IEDB 数据子集上训练，目标是区分表位与非表位序列，并分类表位来源（细菌或病毒），通过正似然比（Positive Likelihood Ratio, LR+）评估过滤效果。
*   **数据准备与统计分析:** 从 IEDB 获取数据，过滤条件包括限制序列长度为 11 个残基以下、去除重复序列等；对生成的序列进行统计分析，包括序列长度分布、相对熵（relative entropy）、香农熵（Shannon entropy）等，以验证其与自然表位的相似性。

## Experiment

*   **生成效果:** 生成模型成功生成了 192,222 个独特的合成表位序列，其统计特性（如长度分布集中在 7-9 个氨基酸、末端位置高相对熵等）与自然表位高度一致，表明模型有效捕捉了表位特征。
*   **分类性能:** 分类器性能令人满意，特别是基于大型语言模型的分类器在 MHC 结合实验数据上表现最佳，F1 分数、ROC AUC 和 LR+ 等指标较高，证明其作为过滤工具的有效性。
*   **实验设置合理性:** 实验涵盖了不同超参数组合、统计分析和分类任务的对比，设置较为全面；但分类器性能在不同数据子集上的差异表明数据来源和实验标注质量对模型效果有显著影响。
*   **局限性:** 维度降低分析（PCA 和 UMAP）未能发现明显的聚类模式，反映表位数据的高度复杂性，可能需要结合结构信息或更复杂的特征提取方法进一步提升性能。

## Further Thoughts

论文展示的大型语言模型在生物序列生成中的应用潜力令人启发，这种方法不仅限于表位设计，还可扩展至其他功能性蛋白质或肽的设计；此外，生成模型与分类模型结合的策略为解决生物学中搜索空间过大的问题提供了一种通用框架，未来可应用于药物设计或酶工程；同时，模型性能与训练数据生物学特异性的强相关性提示我们，构建机器学习模型时应优先考虑高质量、实验验证的数据，而非单纯追求数据量。