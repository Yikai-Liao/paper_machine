---
title: "Can Large Language Models Capture Human Risk Preferences? A Cross-Cultural Study"
pubDatetime: 2025-06-29T06:16:57+00:00
slug: "2025-06-llm-risk-preference"
type: "arxiv"
id: "2506.23107"
score: 0.47672858147609165
author: "grok-3-latest"
authors: ["Bing Song", "Jianing Liu", "Sisi Jian", "Chenyang Wu", "Vinayak Dixit"]
tags: ["LLM", "Risk Preference", "Cross-Cultural Study", "Role-Playing", "Decision Making"]
institution: ["The Hong Kong University of Science and Technology", "Northwestern Polytechnical University", "Imperial College London", "National Key Laboratory of Aircraft Configuration Design", "The University of New South Wales"]
description: "本文通过跨文化和跨语言实验，揭示了大型语言模型在模拟人类风险偏好时的系统性风险厌恶偏差和局限性，为未来模型改进提供了重要参考。"
---

> **Summary:** 本文通过跨文化和跨语言实验，揭示了大型语言模型在模拟人类风险偏好时的系统性风险厌恶偏差和局限性，为未来模型改进提供了重要参考。 

> **Keywords:** LLM, Risk Preference, Cross-Cultural Study, Role-Playing, Decision Making

**Authors:** Bing Song, Jianing Liu, Sisi Jian, Chenyang Wu, Vinayak Dixit

**Institution(s):** The Hong Kong University of Science and Technology, Northwestern Polytechnical University, Imperial College London, National Key Laboratory of Aircraft Configuration Design, The University of New South Wales


## Problem Background

随着大型语言模型（LLMs）在对话系统、内容生成和领域特定咨询任务中的广泛应用，其在模拟复杂人类行为（特别是涉及风险的决策）方面的可靠性受到关注。
论文试图解决的关键问题是：LLMs 是否能够准确捕捉人类在风险情景下的决策偏好？是否存在系统性偏差？此外，研究还探索了跨文化和跨语言背景下模型表现的差异，这是一个相对未被充分研究的领域。

## Method

* **核心思想**：通过角色扮演语言代理（Role-Playing Language Agents, RPLA）架构，利用 LLMs 模拟人类在风险决策场景中的行为，评估其与真实人类选择的契合度。
* **具体实现**：
  - **RPLA 架构**：包含四个组件：个人档案（Profile）、记忆（Memory）、规划（Planning）和行动（Action）。个人档案基于调查数据构建，包含年龄、性别、教育水平和收入等信息，并将城市作为文化上下文因素；记忆机制通过短期记忆存储交互历史，确保决策一致性；规划采用链式思维（Chain-of-Thought, CoT）方法，结合背景信息和预期收益进行推理；行动限制输出为具体选择（如选择选项1或2），避免开放式推理带来的噪声。
  - **风险偏好评估**：采用常数相对风险厌恶（Constant Relative Risk Aversion, CRRA）模型量化风险态度，通过比较模型模拟选择与真实人类选择的差异来评估表现。
  - **跨语言实验**：针对香港和南京的数据，分别使用英文和中文提示（Prompt）进行模拟，探索语言对模型预测的影响。
* **关键点**：方法注重结构化提示设计，通过控制变量减少随机性，同时关注文化和语言差异对模型表现的影响。

## Experiment

* **有效性**：实验在悉尼、香港、达卡和南京四个城市的数据集上进行，结果显示 ChatGPT 4o 和 ChatGPT o1-mini 均表现出比真实人类更强的风险厌恶倾向，但 o1-mini 的预测更接近人类数据。
* **跨文化差异**：在达卡数据集（主要为出租车司机，社会经济多样性低）上，两个模型表现差异不显著，表明 LLMs 在处理同质性高的人群时可能无法捕捉细微差异。
* **语言影响**：使用中文提示时，模型预测与真实数据的偏差更大，尽管中文是香港和南京受访者的母语，这可能与模型训练数据以英文为主有关。
* **实验设置合理性**：实验覆盖多个文化和语言场景，通过重复模拟（每任务模拟三次取多数结果）减少随机性影响，设置较为全面；但模型的系统性风险厌恶偏差表明其在捕捉人类风险偏好的细微差异方面仍有局限。

## Further Thoughts

论文揭示了语言和文化上下文对 LLMs 模拟人类行为的影响，启发我们思考如何通过多语言训练或文化特定微调提升模型表现；同时，社会经济多样性对模型预测的影响提示未来可探索针对特定人群的个性化建模方法；此外，风险决策的复杂性也启发我们在其他复杂决策场景（如时间折扣、道德困境）中进一步测试 LLMs 的能力。