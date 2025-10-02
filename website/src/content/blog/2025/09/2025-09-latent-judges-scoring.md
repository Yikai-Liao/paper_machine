---
title: "Reference-Free Rating of LLM Responses via Latent Information"
pubDatetime: 2025-09-29T12:15:52+00:00
slug: "2025-09-latent-judges-scoring"
type: "arxiv"
id: "2509.24678"
score: 0.7432290080101612
author: "grok-3-latest"
authors: ["Leander Girrbach", "Chi-Ping Su", "Tankred Saanum", "Richard Socher", "Eric Schulz", "Zeynep Akata"]
tags: ["LLM", "Evaluation", "Latent Information", "Scoring", "Ranking"]
institution: ["Technical University of Munich", "National Yang Ming Chiao Tung University", "Helmholtz Munich", "Harvard University", "you.com"]
description: "本文提出 'Latent Judges' 方法，利用大型语言模型内部潜在信息生成确定性、细粒度的无参考评分，显著改进传统 LLM-as-a-Judge 方法在稳定性和区分度上的表现。"
---

> **Summary:** 本文提出 'Latent Judges' 方法，利用大型语言模型内部潜在信息生成确定性、细粒度的无参考评分，显著改进传统 LLM-as-a-Judge 方法在稳定性和区分度上的表现。 

> **Keywords:** LLM, Evaluation, Latent Information, Scoring, Ranking

**Authors:** Leander Girrbach, Chi-Ping Su, Tankred Saanum, Richard Socher, Eric Schulz, Zeynep Akata

**Institution(s):** Technical University of Munich, National Yang Ming Chiao Tung University, Helmholtz Munich, Harvard University, you.com


## Problem Background

大型语言模型（LLM）作为评判模型（LLM-as-a-Judge）在无参考情况下对单一响应进行评分时，面临评分不稳定（因采样随机性）和校准不良（评分集中在量表顶部，导致区分度低和频繁平局）的问题，限制了其在 Best-of-N 选择、多教师蒸馏和模型路由等应用中的可靠性。
论文旨在解决如何在无参考情况下生成稳定、细粒度且确定性的评分，以提升评判任务的实用性。

## Method

*   **核心思想:** 提出 'Latent Judges' 概念，通过提取模型内部的潜在信息（latent information）生成评分，替代传统基于生成 token 的评分方法，以实现确定性和更高的区分度。
*   **具体实现:** 包括三种方法：
    *   **概率加权评分 (Probability-Weighted Ratings):** 提示模型为响应评分后（如 1-10 分），计算下一个 token 预测时对应各整数评分的概率分布，进行加权平均，得到实值评分，避免离散评分的限制。
    *   **验证器风格评分 (Verifier-Style Ratings):** 通过二元问题（如 '这个响应好吗？'）提示模型输出 '是' 或 '否'，取预测 '是' 的概率作为评分，方法简单但区分度可能较低。
    *   **潜在探针 (Latent Probes):** 在模型处理提示后提取内部激活值（隐藏层表示），训练轻量级分类器（如线性探针或小型多层感知机）预测响应质量评分，特别适用于输出 logits 校准不良的情况。
*   **关键优势:** 这些方法不依赖采样生成的离散 token，避免随机性；实值评分提供更高区分度；可通过缩放和偏移解决校准问题。

## Experiment

*   **有效性:** 在成对比较基准（如 MT-Bench, RewardBench）上，概率加权评分和验证器风格评分准确率与传统 10 分制提示相当或更优（提升高达 5 个百分点）；潜在探针在 logits 校准不良时能有效提取质量信号。
*   **单一评分表现:** 在与人类或 GPT-4 评分的 Pearson 相关性测试中，概率加权评分表现最佳（平均相关性约 0.6），验证器风格评分和潜在探针因输出值偏极端相关性较低。
*   **应用场景:** 在列表排名和模型路由任务中，基于潜在信息的方法显著优于传统方法，Spearman 排名相关性更高，显示出实际应用潜力。
*   **实验设置合理性:** 实验覆盖多种模型（包括专门微调的评判模型如 Prometheus）、数据集和任务类型，设置全面；但单一评分相关性受基准数据正偏分布影响，仍有提升空间；对平局的随机打破可能高估传统方法表现，进一步凸显新方法优势。

## Further Thoughts

利用模型内部潜在信息改进评判任务的思路非常启发性，不仅限于评分，还可扩展至检测生成内容幻觉（通过探针识别内部激活不一致信号）、强化学习奖励建模（提供细粒度反馈）以及模型对齐（通过潜在信息理解决策过程，设计更有效对齐策略），为模型内部表示的可解释性和实用性开辟了新视角。