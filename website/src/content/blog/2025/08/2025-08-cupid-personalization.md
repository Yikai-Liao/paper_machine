---
title: "CUPID: Evaluating Personalized and Contextualized Alignment of LLMs from Interactions"
pubDatetime: 2025-08-03T09:04:48+00:00
slug: "2025-08-cupid-personalization"
type: "arxiv"
id: "2508.01674"
score: 0.6495677650508124
author: "grok-3-latest"
authors: ["Tae Soo Kim", "Yoonjoo Lee", "Yoonah Park", "Jiho Kim", "Young-Ho Kim", "Juho Kim"]
tags: ["LLM", "Personalization", "Contextual Preferences", "Inference", "Interaction History"]
institution: ["KAIST", "Seoul National University", "Calvin University", "NAVER AI LAB"]
description: "本文提出CUPID基准数据集，通过756个交互会话历史评估大型语言模型在推断和应用情境化偏好方面的能力，揭示了当前模型的不足并为个性化交互研究提供了重要资源。"
---

> **Summary:** 本文提出CUPID基准数据集，通过756个交互会话历史评估大型语言模型在推断和应用情境化偏好方面的能力，揭示了当前模型的不足并为个性化交互研究提供了重要资源。 

> **Keywords:** LLM, Personalization, Contextual Preferences, Inference, Interaction History

**Authors:** Tae Soo Kim, Yoonjoo Lee, Yoonah Park, Jiho Kim, Young-Ho Kim, Juho Kim

**Institution(s):** KAIST, Seoul National University, Calvin University, NAVER AI LAB


## Problem Background

大型语言模型（LLMs）在个性化对齐方面的研究多假设用户偏好是静态的，不随情境变化。然而，现实中用户偏好是动态且情境依赖的，现有模型难以从多轮交互历史中准确推断情境化偏好并应用于新请求，导致个性化交互效果不佳。本文旨在解决这一问题，评估LLMs在理解和应用情境化偏好方面的能力不足。

## Method

* **核心思想:** 构建一个名为CUPID的基准数据集，用于评估LLMs从用户交互历史中推断情境化偏好并生成符合偏好响应的能力。
* **数据集构建:** CUPID包含756个由人工验证的交互会话历史，每个会话模拟用户与LLM的多轮对话，逐步揭示特定情境下的用户偏好。数据生成流程包括：
  * **用户画像生成（Persona Pool）:** 创建252个独特的用户画像，包含职业、个性、价值观等特征，确保偏好个性化。
  * **情境因素与偏好设计（Context Factors and Preferences）:** 为每个画像生成8个情境因素（如人、地点、工具）和对应的情境化偏好，确保偏好复杂且依赖情境。
  * **交互会话生成（Interaction Sessions）:** 按时间顺序生成13个会话，覆盖一致性（Consistent）、对比性（Contrastive）和变化性（Changing）偏好模式。
  * **对话模拟（Dialogue Simulation）:** 使用两个LLM分别模拟用户和助手，通过多轮反馈逐步揭示偏好。
* **评估任务:** 包括推断任务（Inference Task，从历史交互推断当前请求的偏好）和生成任务（Generation Task，生成符合偏好的响应）。
* **评估指标:** 设计基于LLM的偏好匹配（Preference Match）指标，计算精度（Precision）、召回率（Recall）和F1分数；以及偏好对齐（Preference Alignment）指标，评估生成响应的符合度（1-10分）。
* **辅助工具:** 微调PREFMATCHER-7B模型以降低评估成本，并测试总结策略（Summaries）对模型性能的影响。

## Experiment

* **有效性:** 评估了10个开源和专有LLMs（如GPT-4o、Claude 3.7 Sonnet、Llama 3.1 405B），结果显示所有模型在推断情境化偏好方面表现不佳，精度低于50%，召回率低于65%，F1分数最高仅为55.8%（Claude 3.7 Sonnet）。生成任务表现与推断任务高度相关（相关系数0.764），表明推断能力是关键瓶颈。
* **实验设置合理性:** 实验覆盖三种实例类型（Consistent, Contrastive, Changing），并通过Oracle设置（仅提供相关会话）和Oracle Preference设置（直接提供真实偏好）测试模型上限。结果显示Oracle设置下性能提升20-30个百分点，表明检索相关情境是主要难点。
* **总结策略效果:** 使用交互历史总结（Summaries）后，小模型（如Mistral 7B）性能显著提升（F1分数提升13点），而大模型（如Claude 3.7 Sonnet）略有下降，显示总结策略对本地化、隐私保护的个性化有潜力，但可能导致信息丢失。
* **局限性分析:** 模型在长交互历史和多情境分辨上表现较差，性能随历史长度增加而下降，且易受近期会话影响（Changing实例表现最佳）。

## Further Thoughts

CUPID数据集揭示了情境化偏好的动态性和复杂性，启发我们思考是否可以通过引入时间维度或情境图谱（Context Graph）来建模用户偏好的演变轨迹，以提升模型对长期交互的理解能力。此外，总结策略（Summaries）对小模型性能的提升提示了一种可能性：设计分层记忆机制（Hierarchical Memory），将长期交互历史压缩为关键情境和偏好摘要，既提升效率又保护用户隐私。