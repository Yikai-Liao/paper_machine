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
institution: ["Nanyang Technological University, Singapore", "University of Electronic Science and Technology of China, Chengdu, China"]
description: "本文提出了一种评估大型语言模型行为适应性的框架，通过对比显式用户画像和隐式对话历史两种输入格式，揭示了推理能力对跨格式一致性的关键作用。"
---

> **Summary:** 本文提出了一种评估大型语言模型行为适应性的框架，通过对比显式用户画像和隐式对话历史两种输入格式，揭示了推理能力对跨格式一致性的关键作用。 

> **Keywords:** LLM, Behavioral Adaptation, Sociodemographic Factors, Dialogue History, Reasoning

**Authors:** Qishuai Zhong, Zongmin Li, Siqi Fan, Aixin Sun

**Institution(s):** Nanyang Technological University, Singapore, University of Electronic Science and Technology of China, Chengdu, China


## Problem Background

大型语言模型（LLMs）在与用户交互时需要根据社会人口统计特征（如年龄、职业、教育水平）调整响应内容以提升交互质量。然而，现有研究多集中于单轮对话中显式提供用户画像的情况，忽略了多轮对话中隐式推断用户特征的能力。本文旨在解决这一空白，评估LLMs在显式用户画像和隐式对话历史两种输入格式下的行为适应性及其一致性，探索模型是否能准确推断特征并据此调整响应。

## Method

* **核心框架**：提出了一种评估LLMs行为适应性的框架，对比两种输入格式：(1) 单轮提示中显式提供的用户画像（BA_user），(2) 多轮对话历史中隐式推断的用户特征（BA_dialogue），并评估跨格式的一致性（Consistency）。
* **数据集构建**：由于缺乏带有社会人口统计特征标注的对话数据集，设计了一个多智能体生成流程，使用GPT-4o等模型模拟用户和问答机器人，生成职业建议领域的合成对话数据集。流程包括用户模拟器（user_simulator，用于生成用户问题）、上下文一致性检测器（ooc_detector，确保问题与用户画像一致）和问答模型（qa_llm，模拟真实聊天机器人响应），确保对话自然嵌入用户特征。
* **评估工具与指标**：采用价值调查模块（Value Survey Module, VSM 2013）作为评估工具，通过18个多选题量化模型的价值表达。行为适应性通过Jensen-Shannon散度（JSD）计算不同用户群体间响应概率分布的差异；一致性通过Earth Mover’s Distance（EMD）衡量同一用户在两种输入格式下的响应差异。
* **实验流程**：测试多个开源LLMs（如Qwen2.5、Llama3.1、DeepSeek-V3、QwQ-32B），记录模型在不同场景下的响应（包括选择选项、概率分布和推理过程），并通过分组分析（按年龄、教育等维度）量化适应性和一致性。

## Experiment

* **行为适应性效果**：大多数模型能够根据社会人口统计特征调整价值表达，尤其是在年龄和教育水平上的适应性显著。例如，年龄差距大的群体（如<30岁 vs. >60岁）间响应散度远高于接近年龄组，表明模型对特征变化敏感。
* **一致性表现**：在显式用户画像和隐式对话历史两种格式下，较小模型（参数<10B）一致性较差，响应差异接近随机基线；而较大模型和推理增强模型（如QwQ-32B）一致性更高，EMD值最低（0.112 vs. 基线0.125），表明推理能力对跨格式适应至关重要。
* **实验设置合理性**：实验覆盖多种模型和多个社会人口统计维度（年龄、教育、职业、国籍），通过JSD和EMD等统计方法量化结果，并验证了合成数据集质量（人工和LLM评判均给高分）。但论文也指出局限，如对话数据仅由GPT-4o生成可能引入风格偏见，VSM 2013问卷覆盖范围有限。
* **效果显著性**：推理增强模型QwQ-32B在一致性上的提升显著，其相对基线比率最低（0.896），表明推理能力对提升适应性和一致性有重要作用。

## Further Thoughts

论文揭示了推理能力在LLMs行为适应性中的关键作用，QwQ-32B通过系统性回顾和整合社会人口统计特征实现了更高一致性，这启发我们思考是否可以通过专门的推理训练或提示设计进一步提升模型在复杂上下文中的适应能力。此外，多智能体数据集生成方法也具有潜力，未来可扩展到更多领域（如医疗、教育）或更复杂的文化背景，测试模型泛化能力。另一个发散性想法是，是否可以动态调整社会人口统计特征的权重，例如在某些文化中年龄可能比职业更重要，这种差异是否可以通过模型训练或提示设计捕捉？