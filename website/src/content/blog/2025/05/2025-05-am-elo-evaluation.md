---
title: "am-ELO: A Stable Framework for Arena-based LLM Evaluation"
pubDatetime: 2025-05-06T12:28:50+00:00
slug: "2025-05-am-elo-evaluation"
type: "arxiv"
id: "2505.03475"
score: 0.4704725236311061
author: "grok-3-latest"
authors: ["Zirui Liu", "Jiatong Li", "Yan Zhuang", "Qi Liu", "Shuanghong Shen", "Jie Ouyang", "Mingyue Cheng", "Shijin Wang"]
tags: ["LLM", "Evaluation Framework", "Ranking System", "Annotator Modeling", "Stability"]
institution: ["State Key Laboratory of Cognitive Intelligence, University of Science and Technology of China", "Institute of Artificial Intelligence, Hefei Comprehensive National Science Center", "iFLYTEK Co., Ltd"]
description: "本文提出 am-ELO 框架，通过最大似然估计和标注者能力建模，显著提升了大型语言模型竞技场评估中 ELO 评分系统的稳定性和准确性。"
---

> **Summary:** 本文提出 am-ELO 框架，通过最大似然估计和标注者能力建模，显著提升了大型语言模型竞技场评估中 ELO 评分系统的稳定性和准确性。 

> **Keywords:** LLM, Evaluation Framework, Ranking System, Annotator Modeling, Stability

**Authors:** Zirui Liu, Jiatong Li, Yan Zhuang, Qi Liu, Shuanghong Shen, Jie Ouyang, Mingyue Cheng, Shijin Wang

**Institution(s):** State Key Laboratory of Cognitive Intelligence, University of Science and Technology of China, Institute of Artificial Intelligence, Hefei Comprehensive National Science Center, iFLYTEK Co., Ltd


## Problem Background

大型语言模型（LLMs）的竞技场式评估（Arena-based Evaluation）是当前模型比较的重要范式，但传统 ELO 评分系统存在显著的不稳定性问题，主要源于算法对数据顺序的敏感性（导致评分不一致）和对标注者能力差异的忽视（引入偏见），从而降低了评估结果的可信度和实用性，尤其在高风险决策（如模型部署）中可能带来误导。

## Method

*   **m-ELO（基于最大似然估计的 ELO）**：
    *   针对传统 ELO 评分系统对数据顺序敏感的问题，提出使用最大似然估计（Maximum Likelihood Estimation, MLE）替代迭代更新方法。
    *   通过对整个数据集的全局优化，计算模型的 ELO 评分，避免了数据顺序的影响，确保评分结果的一致性。
    *   理论上证明了 m-ELO 的收敛性和稳定性（Theorem 4.1），即在固定一个模型评分的情况下，似然函数是凹函数，具有唯一极值点。
*   **am-ELO（标注者能力建模的 ELO）**：
    *   在 m-ELO 基础上，引入标注者能力建模（Annotator Modeling），通过修改 ELO 概率函数，将每个标注者的判别能力（Discriminative Ability）作为参数 *θ_k* 引入。
    *   借鉴心理测量学中的项目反应理论（Item Response Theory, IRT），通过 MLE 同时估计模型评分和标注者能力，实现更准确的评估结果聚合。
    *   设计归一化约束（确保标注者能力总和为 1）和异常标注者过滤机制（基于能力阈值 *ϵ*），避免评分反转问题并提升稳定性。
*   **稳定竞技场框架（Stable Arena Framework）**：
    *   整合 m-ELO 和 am-ELO，提出一个完整的评估框架，通过数据筛选（过滤标注记录不足的标注者）和能力评估（排除异常标注者），进一步增强评估的鲁棒性。

## Experiment

*   **数据集与设置**：基于 Chatbot Arena 真实数据集（33,000 条对话记录，过滤后 4,321 条有效记录），对比传统 ELO、m-ELO 和 am-ELO 的性能，实验包括真实数据评估和模拟扰动实验。
*   **有效性**：am-ELO 在对数似然损失（Log-Likelihood Loss）和预测准确率（Accuracy 提升至 0.7581）上显著优于传统 ELO 和 m-ELO，表明其拟合能力和泛化能力更强。
*   **稳定性**：通过四种扰动策略（Random, Equal, Flip, Mixed）模拟异常标注者，am-ELO 的评分一致性（Consistency）比传统方法高约 70%，不一致率降低至 30%。
*   **标注者建模**：am-ELO 能有效识别异常标注者（F1 分数高达 95%），通过能力参数 *θ_k* 量化标注者可靠性，支持奖励机制设计。
*   **实验局限**：实验设计较为全面，但数据集规模和任务类型较单一，未探讨方法在多语言或多领域任务中的适用性，可能存在一定的泛化性限制。

## Further Thoughts

标注者能力建模的概念可扩展至其他主观性评估场景（如众包任务），通过量化个体差异提升结果可靠性；MLE 替代迭代更新的思路启发在动态评分系统中引入全局优化方法；异常标注者过滤机制结合主动学习（Active Learning）可动态优化标注者选择，为构建高质量评估数据集提供新思路。