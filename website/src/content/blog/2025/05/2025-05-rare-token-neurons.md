---
title: "Emergent Specialization: Rare Token Neurons in Language Models"
pubDatetime: 2025-05-19T08:05:13+00:00
slug: "2025-05-rare-token-neurons"
type: "arxiv"
id: "2505.12822"
score: 0.6882641539980316
author: "grok-3-latest"
authors: ["Jing Liu", "Haozheng Wang", "Yueheng Li"]
tags: ["LLM", "Rare Tokens", "Functional Specialization", "Heavy-Tailed Distribution", "Neuron Analysis"]
institution: ["ENS, Université PSL", "EHESS", "CNRS", "Sorbonne Université", "Laboratoire de Physique (LPENS)"]
description: "本文通过多维分析揭示了大型语言模型中针对稀有词汇的自发功能特化机制，识别出稀有词汇神经元及其三阶段影响分布，为数据高效训练和领域适应提供了理论和实证支持。"
---

> **Summary:** 本文通过多维分析揭示了大型语言模型中针对稀有词汇的自发功能特化机制，识别出稀有词汇神经元及其三阶段影响分布，为数据高效训练和领域适应提供了理论和实证支持。 

> **Keywords:** LLM, Rare Tokens, Functional Specialization, Heavy-Tailed Distribution, Neuron Analysis

**Authors:** Jing Liu, Haozheng Wang, Yueheng Li

**Institution(s):** ENS, Université PSL, EHESS, CNRS, Sorbonne Université, Laboratoire de Physique (LPENS)


## Problem Background

大型语言模型（LLMs）在处理稀有词汇（Rare Tokens）时面临显著挑战，这些词汇在训练数据中频率极低，但对专业领域应用至关重要。
由于自然语言的幂律分布特性，模型往往难以有效表示和生成长尾词汇，这不仅影响基础语言建模，还限制了在特定领域的应用。
作者提出核心问题：LLMs 在预训练过程中是否会自发形成专门处理稀有词汇的内部机制？这一问题受到人类语言习得中快速映射能力和互补学习系统（CLS）理论的启发。

## Method

*   **核心目标：** 识别并分析语言模型中专门处理稀有词汇的神经元（Rare Token Neurons），探索其形成机制和行为特性。
*   **具体步骤：**
    *   **稀有词汇神经元识别：** 在最后一层 MLP 子层中，通过消融实验（Ablation Experiments）计算每个神经元对稀有词汇预测的影响（Neuron Effect），即消融后 token 级损失的变化（∆loss），从而识别对稀有词汇影响显著的神经元。
    *   **激活空间几何分析：** 研究稀有词汇神经元的协同激活模式（Co-activation Patterns），通过主成分分析（PCA）计算有效维度（Effective Dimensionality）以评估激活分布的维度压缩程度，并通过成对余弦相似度（Pairwise Cosine Similarity）分析神经元间的激活相关性，揭示其协调性。
    *   **神经元影响分布与相变：** 分析神经元影响（∆loss）的分布特征，识别出三种阶段（高原期、幂律衰减期、快速衰减期），并通过有限差分方法和变化点检测算法确定阶段边界，探索功能特化在训练中的动态演化。
    *   **权重特征谱分析：** 基于重尾自正则化理论（Heavy-Tailed Self-Regularization, HT-SR），分析不同神经元组的权重矩阵特征值分布，计算幂律指数（α_Hill）以评估重尾特性，探索功能特化与统计力学临界状态（Criticality）之间的联系。
*   **创新点：** 方法结合了因果分析（消融实验）、几何分析（激活空间）和统计力学视角（重尾分布），多维度揭示了稀有词汇处理机制的内在原理。

## Experiment

*   **有效性：** 实验基于 Pythia 模型家族，使用 C4 语料库和 The Pile 数据集，发现一小部分神经元（Rare Token Neurons）对稀有词汇预测有显著影响，其影响分布呈现高原期（1.7% 神经元）、幂律衰减期（10% 神经元）和快速衰减期（87.4% 神经元）三个阶段。
*   **动态演化：** 这三个阶段在训练过程中逐步形成，从初始的均匀状态演变为功能分化的架构，表明功能特化是训练的自发结果。
*   **协调性：** 稀有词汇神经元之间表现出显著协同激活（Co-activation），有效维度较低（0.49 vs. 随机神经元的 0.56），且与处理常见词汇的神经元呈现反相关，表明其功能模块化。
*   **统计特性：** 稀有词汇神经元的权重分布表现出更强的重尾特性（较低的 α_Hill 值），支持 HT-SR 理论关于功能特化与临界状态的联系。
*   **实验设置合理性：** 实验涵盖了不同模型规模、训练阶段和多维分析，数据处理（如稀有词汇过滤）合理，结果显著，尤其在揭示功能特化和统计力学联系方面表现突出。

## Further Thoughts

论文揭示了语言模型中功能特化的自发形成（Emergent Specialization），这与人类大脑的互补学习系统（CLS）有类似机制，可能启发设计更高效的模型；
重尾分布与功能特化之间的联系提示模型优化过程可能隐含统计力学原理，未来可探索通过调控权重分布增强对稀有事件的处理能力；
稀有词汇神经元的协同激活模式表明模型内部可能存在子网络（Subnetworks），这为解释模型行为和设计可解释性工具提供了新思路。