---
title: "Grammars of Formal Uncertainty: When to Trust LLMs in Automated Reasoning Tasks"
pubDatetime: 2025-05-26T14:34:04+00:00
slug: "2025-05-formal-uncertainty-llm"
type: "arxiv"
id: "2505.20047"
score: 0.5170680505171702
author: "grok-3-latest"
authors: ["Debargha Ganguly", "Vikash Singh", "Sreehari Sankar", "Biyao Zhang", "Xuecen Zhang", "Srinivasan Iyengar", "Xiaotian Han", "Amit Sharma", "Shivkumar Kalyanaraman", "Vipin Chaudhary"]
tags: ["LLM", "Uncertainty Quantification", "Formal Verification", "Reasoning", "Probabilistic Modeling"]
institution: ["Case Western Reserve University", "Microsoft Corporation", "Microsoft Research"]
description: "本文提出基于概率上下文无关文法（PCFG）的框架，量化大型语言模型在形式化推理中的不确定性，通过任务依赖指标和信号融合实现选择性验证，显著降低错误率。"
---

> **Summary:** 本文提出基于概率上下文无关文法（PCFG）的框架，量化大型语言模型在形式化推理中的不确定性，通过任务依赖指标和信号融合实现选择性验证，显著降低错误率。 

> **Keywords:** LLM, Uncertainty Quantification, Formal Verification, Reasoning, Probabilistic Modeling

**Authors:** Debargha Ganguly, Vikash Singh, Sreehari Sankar, Biyao Zhang, Xuecen Zhang, Srinivasan Iyengar, Xiaotian Han, Amit Sharma, Shivkumar Kalyanaraman, Vipin Chaudhary

**Institution(s):** Case Western Reserve University, Microsoft Corporation, Microsoft Research


## Problem Background

形式化方法在系统可靠性验证中提供数学保证，但其高门槛（专业知识和劳动需求）限制了广泛应用。
大型语言模型（LLMs）展现出生成形式化规范（如代码、证明）的潜力，有望民主化形式化推理，然而其概率性输出与形式化验证对确定性保证的需求存在根本矛盾。
论文聚焦这一矛盾，试图解决如何利用 LLMs 的生成能力进行自动化推理，同时确保结果符合形式化验证的严谨标准，关键在于量化和管理 LLM 输出中的不确定性。

## Method

*   **核心思想:** 通过建模 LLM 生成形式化产物（如 SMT-LIB 程序）的概率分布，量化其不确定性，并利用这些不确定性信号指导验证流程，减少错误。
*   **具体实现步骤:**
    *   **概率空间定义:** 形式化 SMT-LIB 程序的概率空间，假设 LLM 输出的分布可通过概率上下文无关文法（PCFG）近似。
    *   **PCFG 建模:** 基于 SMT-LIB v2 文法，从 LLM 生成的多个样本中解析语法树，使用最大似然估计（MLE）或平滑方法（如 Lidstone 平滑）估计生产规则的概率分布。
    *   **不确定性指标提取:** 从 PCFG 中衍生多种指标，包括信息论指标（如 Shannon 熵、Rényi 熵、困惑度）、结构复杂度指标（如谱半径、分支因子）以及规则概率分布的统计特征（如峰度、偏度），同时结合自一致性指标（如文本与 SMT 输出的一致性）。
    *   **信号融合与选择性验证:** 开发轻量级、模型无关的融合策略（如加权平均、机器学习模型），整合多种不确定性信号，通过设定阈值实现选择性验证，优先验证高不确定性输出或弃权以降低错误率。
*   **关键创新:** 将神经模型与形式化验证桥梁化，通过结构化的不确定性分析超越单一最高概率输出策略，提升形式化推理的可靠性。

## Experiment

*   **有效性:** 实验评估了五个前沿 LLMs 在四个推理数据集上的表现，SMT 形式化在逻辑任务（如 ProofWriter）上显著提升准确率（最高 +34.8%），但在事实性任务（如 FOLIO）上表现较差（下降高达 -44.5%），表明方法效果高度任务依赖。
*   **不确定性指标表现:** PCFG 衍生的指标（如语法熵）在逻辑任务中表现出色（AUROC > 0.93），能精准识别错误；而在知识密集型任务中，自一致性指标（如文本-SMT 一致性）更有效，验证了不确定性信号的任务依赖性。
*   **选择性验证效果:** 通过融合不确定性信号，选择性验证大幅降低错误率（14%-100%），且弃权率较低，显示出方法在实际应用中的高效性。
*   **实验设置合理性:** 数据集涵盖逻辑、事实和混合推理任务，模型选择包括多种前沿 LLMs，样本量（N=100）支持 PCFG 估计，实验设计对比了文本与 SMT 输出，较为全面；但未深入探讨模型规模或训练数据的影响，可能存在一定局限。

## Further Thoughts

论文揭示了语法异常性与语义错误的高度相关性，这启发我们可以在其他结构化输出领域（如代码生成）中探索语法与功能的关联；此外，任务依赖的不确定性信号提示未来 AI 系统设计应根据任务特性定制不确定性量化策略；最后，形式化与文本推理路径的不对称性表明联合训练或多模态对齐可能是弥合差距的关键方向。