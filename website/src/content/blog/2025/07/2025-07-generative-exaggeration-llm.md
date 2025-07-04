---
title: "Generative Exaggeration in LLM Social Agents: Consistency, Bias, and Toxicity"
pubDatetime: 2025-07-01T10:54:51+00:00
slug: "2025-07-generative-exaggeration-llm"
type: "arxiv"
id: "2507.00657"
score: 0.6697947321734513
author: "grok-3-latest"
authors: ["Jacopo Nudo", "Mario Edoardo Pandolfo", "Edoardo Loru", "Mattia Samory", "Matteo Cinelli", "Walter Quattrociocchi"]
tags: ["LLM", "Social Agents", "Bias", "Polarization", "Generative Exaggeration"]
institution: ["Sapienza University of Rome"]
description: "本文通过构建 LLM 社交代理模拟政治话语，发现其存在‘生成夸张’现象，即系统性放大用户显著特征，导致意识形态极化、毒性增加和行为失真，挑战了 LLMs 在高风险社交应用中的可靠性。"
---

> **Summary:** 本文通过构建 LLM 社交代理模拟政治话语，发现其存在‘生成夸张’现象，即系统性放大用户显著特征，导致意识形态极化、毒性增加和行为失真，挑战了 LLMs 在高风险社交应用中的可靠性。 

> **Keywords:** LLM, Social Agents, Bias, Polarization, Generative Exaggeration

**Authors:** Jacopo Nudo, Mario Edoardo Pandolfo, Edoardo Loru, Mattia Samory, Matteo Cinelli, Walter Quattrociocchi

**Institution(s):** Sapienza University of Rome


## Problem Background

本文聚焦于大型语言模型（LLMs）在模拟社交媒体政治话语时的行为模式，特别是在高风险、高极化的场景（如 2024 年美国总统大选）中，探讨 LLMs 作为社交代理（Social Agents）是否能真实模拟人类用户行为，或是否会引入系统性偏差和扭曲。
关键问题在于：LLMs 在模拟用户时，是否会忠实再现其语言风格、意识形态一致性和行为特征，还是会通过‘生成夸张’（Generative Exaggeration）放大显著特征，导致模拟失真？这不仅关乎技术性能，还涉及 LLMs 在内容审核、政策建模等高风险应用中的可靠性。

## Method

* **实验框架**：基于 X 平台上 21 百万条互动数据，针对 1186 名真实用户构建 LLM 代理，模拟其对政治话题的回复，并与人类真实回复进行一对一比较。
* **模型选择**：选用三个模型家族（Gemini、Mistral、DeepSeek），每个家族包含大、小规模变体，以考察模型规模和地域背景（美国、中国、欧洲）对结果的影响。
* **初始化策略**：采用两种提示方式：Zero Shot（仅提供用户政治倾向）和 Few Shot（提供用户昵称、简介及 30 条历史推文），以测试上下文信息对模拟效果的影响。
* **评估维度**：从四个方面分析 LLM 输出：
  * 语言风格：通过 LogTTR 测量词汇多样性，评估生成文本的语言丰富度。
  * 意识形态一致性：使用政治倾向分类器和一致性损失指标，评估生成内容与用户政治立场的匹配度。
  * 毒性：借助 Perspective API 计算生成内容中毒性语言（得分 > 0.6）的比例，衡量有害内容程度。
  * 生成夸张：通过分析表情符号和政治标签的使用频率，检测模型是否过度放大显著特征。
* **控制变量**：确保每个 LLM 代理回复的推文与人类用户回复的推文相同，以隔离提示策略和模型差异的影响。

## Experiment

* **词汇多样性**：LLMs 在小样本推文中词汇多样性（LogTTR）高于人类，但在更大样本中趋于重复，表明其语言生成受内部模式限制；Few Shot 策略使输出更接近人类行为。
* **意识形态一致性**：Zero Shot 下，模型输出缺乏一致性，倾向于生成中立或分散内容；Few Shot 下，一致性显著提升（如 DeepSeek 对共和党用户模拟一致性从 45% 升至 95%），但也增加了极端化倾向，甚至对中立用户表现出极化。
* **毒性**：Zero Shot 下毒性内容较少；Few Shot 下毒性显著增加，尤其是 Gemini 模型，毒性比例远超人类基准（5%）；毒性分布显示模型倾向于放大提示中的毒性特征。
* **生成夸张**：Few Shot 下，模型过度使用表情符号和政治标签（如 #MAGA 频率比人类高 10-15 倍），表现出对显著特征的系统性放大，形成刻板化模拟。
* **实验设置评价**：实验设计全面，涵盖不同模型、初始化策略和评估维度，数据量（21 百万互动）充足；但局限在于仅限于美国政治语境和单轮交互，未考虑多轮对话或跨文化差异。

## Further Thoughts

论文提出的‘生成夸张’（Generative Exaggeration）概念揭示了 LLMs 在追求一致性或显著性时可能牺牲行为真实性，这可能源于训练中优化目标（最小化下一词预测误差）导致对高显著性特征的过度关注。启发我们思考如何设计更平衡的优化目标或对齐策略，减少夸张效应；此外，Few Shot 策略提高一致性但放大毒性和极化，提示我们在增加上下文时需权衡安全性和真实性，或许可以通过调整提示设计或引入反向约束（如降低毒性权重）来缓解这一问题。