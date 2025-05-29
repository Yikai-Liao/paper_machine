---
title: "Evaluating LLM Adaptation to Sociodemographic Factors: User Profile vs. Dialogue History"
pubDatetime: 2025-05-27T15:52:39+00:00
slug: "2025-05-llm-sociodemographic-adaptation"
type: "arxiv"
id: "2505.21362"
score: 0.6224005431948516
author: "grok-3-latest"
authors: ["Qishuai Zhong", "Zongmin Li", "Siqi Fan", "Aixin Sun"]
tags: ["LLM", "Behavioral Adaptation", "Sociodemographic Factors", "Dialogue History", "Reasoning"]
institution: ["Nanyang Technological University", "University of Electronic Science and Technology of China"]
description: "本文提出一个评估框架，通过合成数据集和价值观调查模块，系统研究了大型语言模型对社会人口统计特征的适应能力和跨格式一致性，揭示了推理能力对稳健适应的关键作用。"
---

> **Summary:** 本文提出一个评估框架，通过合成数据集和价值观调查模块，系统研究了大型语言模型对社会人口统计特征的适应能力和跨格式一致性，揭示了推理能力对稳健适应的关键作用。 

> **Keywords:** LLM, Behavioral Adaptation, Sociodemographic Factors, Dialogue History, Reasoning

**Authors:** Qishuai Zhong, Zongmin Li, Siqi Fan, Aixin Sun

**Institution(s):** Nanyang Technological University, University of Electronic Science and Technology of China


## Problem Background

大型语言模型（LLMs）在与用户交互时需根据社会人口统计特征（如年龄、职业、教育水平）调整响应，以提升交互质量和用户信任。然而，现有研究多聚焦于单轮对话中显式提供用户画像的场景，忽略了通过多轮对话历史隐式推断特征的情况，导致对模型行为一致性和适应能力的评估不足，存在刻板印象或不一致性风险。

## Method

* **评估框架设计**：提出两种评估场景以测试 LLMs 的行为适应能力：BA_user（基于显式用户画像）和 BA_dialogue（基于隐式对话历史），并通过 Consistency 场景评估跨格式一致性。BA_user 直接在提示中提供用户画像信息，BA_dialogue 要求模型从对话历史中推断特征。
* **数据集构建**：设计多代理生成流程，使用 GPT-4o 作为用户模拟器、上下文一致性检测器和问答模型，生成职业建议主题的合成对话数据集，每个对话与唯一用户画像相关联，确保属性一致性。
* **价值观测量**：采用 Hofstede 的价值观调查模块（VSM 2013），选取 18 个问题，通过模型对问卷的回答量化其价值观表达。
* **度量方法**：行为适应能力通过 Jensen-Shannon 散度（JSD）计算不同用户群体间响应分布差异，并与基线对比；一致性通过 Earth Mover’s Distance（EMD）测量同一用户在两种格式下的响应序列差异。
* **模型测试**：评估多个开源 LLMs（如 Qwen2.5、Llama3.1、DeepSeek-V3 和 QwQ-32B），观察其在不同场景下的表现差异。

## Experiment

* **行为适应能力**：大多数模型能根据社会人口统计特征（如年龄、教育水平）调整价值观表达，属性差异越大（如年龄从‘<30’到‘>60’），响应差异越显著，表明模型对显式和隐式输入均有适应能力，尤其在对话历史场景中对年龄适应更稳定。
* **一致性表现**：跨格式一致性因模型而异，较小模型（如参数少于 10B）差异较大（EMD 值接近基线），而较大模型和推理增强模型（如 QwQ-32B）差异最小（EMD 值为 0.112，基线比例 0.896），表现出更强一致性。
* **推理能力影响**：推理能力强的模型通过系统性整合上下文信息，实现更精准响应和更高一致性，QwQ-32B 表现最佳。
* **实验设置合理性**：实验覆盖多种社会人口统计维度，通过合成数据集控制变量，质量经人工和 LLM 评判验证（评分较高），但合成数据可能限制泛化性，VSM 问卷范围有限。

## Further Thoughts

论文提出的跨格式一致性评估方法启发我思考是否可以通过设计特定训练数据或提示策略提升模型对输入形式的鲁棒性；推理能力对适应和一致性的重要性让我考虑是否可以开发专门推理模块或通过强化学习增强模型在复杂上下文中的表现；合成数据集生成方法也启发我探索类似技术在其他领域（如情感分析或文化对齐）中的应用。