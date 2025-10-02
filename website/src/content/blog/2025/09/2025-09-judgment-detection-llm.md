---
title: "Who's Your Judge? On the Detectability of LLM-Generated Judgments"
pubDatetime: 2025-09-29T17:54:57+00:00
slug: "2025-09-judgment-detection-llm"
type: "arxiv"
id: "2509.25154"
score: 0.7175290593914229
author: "grok-3-latest"
authors: ["Dawei Li", "Zhen Tan", "Chengshuai Zhao", "Bohan Jiang", "Baixiang Huang", "Pingchuan Ma", "Abdullah Alnaibari", "Kai Shu", "Huan Liu"]
tags: ["LLM", "Judgment Detection", "Bias Analysis", "Feature Augmentation", "Interpretability"]
institution: ["Arizona State University", "Emory University"]
description: "本文提出判断检测任务并设计 *J-Detector*，通过特征增强有效区分 LLM 与人类判断，确保评估公平性，并揭示影响检测性的关键因素。"
---

> **Summary:** 本文提出判断检测任务并设计 *J-Detector*，通过特征增强有效区分 LLM 与人类判断，确保评估公平性，并揭示影响检测性的关键因素。 

> **Keywords:** LLM, Judgment Detection, Bias Analysis, Feature Augmentation, Interpretability

**Authors:** Dawei Li, Zhen Tan, Chengshuai Zhao, Bohan Jiang, Baixiang Huang, Pingchuan Ma, Abdullah Alnaibari, Kai Shu, Huan Liu

**Institution(s):** Arizona State University, Emory University


## Problem Background

大型语言模型（LLM）被广泛用于自动化判断任务（如学术同行评审），以其高效和低成本的优势评估候选内容并提供评分。然而，LLM 生成的判断存在系统性偏见（如偏好内容长度或表面流畅性）和脆弱性（易被恶意输入操控），可能导致不公平或不可靠的评估结果，特别是在敏感场景中。因此，区分 LLM 生成的判断与人类判断成为确保评估公平性和可靠性的迫切需求。

## Method

*   **核心思想:** 设计一种轻量级、可解释的神经检测器 *J-Detector*，通过特征增强捕捉判断分数与候选内容之间的交互信息，区分 LLM 生成的判断与人类判断。
*   **特征提取:** 
    *   **判断内在特征（Judgment-Intrinsic Features）:** 分析判断分数本身的分布模式，例如多维度评分中的相关性或单维度评分中的一致性。
    *   **判断-候选交互特征（Judgment-Candidate Interaction Features）:** 结合候选内容的特性与判断分数的关系，提取两类特征：
        *   **语言学特征（Linguistic Features）:** 包括长度（词数、句长）、词汇多样性、可读性指数、句法复杂度和话语标记频率等，用于捕捉 LLM 判断中常见的偏见（如长度偏见、流畅性偏见）。
        *   **LLM 增强特征（LLM-Enhanced Features）:** 利用 LLM（如 Qwen-3-8B）生成与任务对齐的特征，如风格、格式、内容维度评分（帮助性、正确性等），以捕捉 LLM 判断中的表面偏好和深层模式。
*   **模型训练:** 使用增强后的特征向量（结合基础判断分数、语言学特征和 LLM 增强特征）训练轻量级二分类器（如 RandomForest、LGBM、XGB），输出判断为 LLM 生成的概率。
*   **群体级聚合:** 对于批量判断，采用简单求和聚合方法，将个体判断的 logits 汇总为群体级评分，以提高检测鲁棒性。
*   **设计原则:** 强调准确性（捕捉两种关键特征）、高效性（低计算开销）和可解释性（支持 LLM 偏见分析），适用于大规模部署。

## Experiment

*   **有效性:** 在 *JD-Bench* 数据集（涵盖点式、成对、列表式判断及多种场景）上，*J-Detector* 的 F1 和 AUROC 指标显著优于基线方法（如 SLM-based 和 LLM-based 检测器），例如在 Helpsteer2 上 F1 达 99.8%，在单维度数据集 Helpsteer3 和 ANTIQUE 上 F1 分别提升至 74.0% 和 85.4%（基线仅 50-55%）。
*   **实验设置合理性:** 数据集覆盖多种判断类型和应用场景（如学术评审、文档排名），包括闭源和开源 LLM 生成的判断，确保了测试的多样性和代表性；对比方法包括 SLM（如 RoBERTa）和 LLM 基线，评估全面。
*   **影响因素分析:** 实验验证了群体大小、判断维度和评分尺度对检测性的正向影响，例如群体大小从 1 增至 16 时 F1 显著提升，多维度判断（如 NeurIPS）比单维度判断（如 ANTIQUE）更易检测。
*   **局限性:** 在多 LLM 评委场景下，检测性能下降明显（如 Helpsteer2 的 F1 从 99.8% 降至 66.9%），表明需要进一步研究异构模型判断的检测方法。
*   **实用性:** 在真实场景（如少样本和缺失文本评审）中，*J-Detector* 与文本检测器结合后性能最佳（F1 达 99.3%），证明其在低资源环境下的价值。

## Further Thoughts

本文通过特征增强捕捉 LLM 判断中的系统性偏见，为检测提供了新思路，这种方法是否能推广至其他生成内容（如文本、代码）的检测？此外，随着 LLM 与人类偏见对齐程度的提高，其判断可检测性降低，是否需要设计动态检测机制，结合上下文或用户行为数据？*J-Detector* 的可解释性为量化偏见提供了可能，是否能进一步用于改进 LLM 训练，减少评估中的不公平性？