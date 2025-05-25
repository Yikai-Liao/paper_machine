---
title: "R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning"
pubDatetime: 2025-05-22T17:58:26+00:00
slug: "2025-05-r1-searcher-dynamic-knowledge"
type: "arxiv"
id: "2505.17005"
score: 0.649542120924704
author: "grok-3-latest"
authors: ["Huatong Song", "Jinhao Jiang", "Wenqing Tian", "Zhipeng Chen", "Yuhuan Wu", "Jiahao Zhao", "Yingqian Min", "Wayne Xin Zhao", "Lei Fang", "Ji-Rong Wen"]
tags: ["LLM", "Retrieval-Augmented Generation", "Reinforcement Learning", "Internal Knowledge", "Dynamic Acquisition"]
institution: ["Renmin University of China", "DataCanvas Alaya NeW", "Beijing Institute of Technology"]
description: "本文提出 R1-Searcher++ 框架，通过两阶段训练（SFT + RL）和奖励与记忆机制，让大型语言模型动态整合内部知识与外部检索，实现高效的检索增强推理。"
---

> **Summary:** 本文提出 R1-Searcher++ 框架，通过两阶段训练（SFT + RL）和奖励与记忆机制，让大型语言模型动态整合内部知识与外部检索，实现高效的检索增强推理。 

> **Keywords:** LLM, Retrieval-Augmented Generation, Reinforcement Learning, Internal Knowledge, Dynamic Acquisition

**Authors:** Huatong Song, Jinhao Jiang, Wenqing Tian, Zhipeng Chen, Yuhuan Wu, Jiahao Zhao, Yingqian Min, Wayne Xin Zhao, Lei Fang, Ji-Rong Wen

**Institution(s):** Renmin University of China, DataCanvas Alaya NeW, Beijing Institute of Technology


## Problem Background

大型语言模型（LLMs）依赖静态知识，容易产生幻觉，尤其在开放性任务中表现不佳；传统检索增强生成（RAG）方法虽引入外部信息，但成本高、泛化能力差，且常忽视模型内部知识；本文旨在解决如何让 LLMs 自适应地动态切换内部知识与外部检索，并通过记忆机制内化外部信息以提升效率和智能水平。

## Method

* **核心思想**：通过两阶段训练策略（SFT Cold-Start 和 RL for Dynamic Knowledge Acquisition），让 LLMs 自适应利用内部知识和外部信息源，同时通过记忆机制将外部信息内化为内部知识，减少重复检索。
* **第一阶段 - SFT Cold-Start**：采用拒绝采样（Rejection Sampling）生成高质量训练数据，确保模型输出符合特定格式（使用 `<internal>` 和 `<external>` 标记区分知识来源），通过监督微调（SFT）初步训练模型自适应调用内部知识和外部检索能力，奠定格式规范和基础能力。
* **第二阶段 - RL for Dynamic Knowledge Acquisition**：基于结果的强化学习（Outcome-based RL）进一步优化模型，采用 REINFORCE++ 算法，结合 KL 正则化和掩码机制（屏蔽外部文档对损失计算的影响）确保训练稳定；具体包括：
  * **内部知识利用激励**：设计多维度奖励函数，包括格式奖励（确保输出格式正确）、答案奖励（基于 Cover Exact Match 评估答案准确性，限制答案长度避免奖励欺骗）、组奖励（通过正确回答中检索次数的标准差和最小检索次数，鼓励减少不必要检索），引导模型在自信时优先使用内部知识。
  * **外部知识记忆机制**：训练一个重写模型（Rewriting Model）将检索到的外部信息转化为内部知识格式，通过附加的记忆损失函数（Memorization Loss）让模型记忆这些信息，加入权重系数避免过度依赖记忆，确保模型在未来推理中直接调用已记忆知识，减少重复检索。
* **关键创新**：通过奖励设计和记忆机制实现内部知识与外部检索的动态平衡，并随时间不断丰富模型内部知识库，提升推理效率和智能水平。

## Experiment

* **性能提升**：R1-Searcher++ 在四个多跳问答数据集（HotpotQA, 2WikiMultiHopQA, Bamboogle, Musique）上显著优于基线方法，在整体测试集上比最强 RL 基线 R1-Searcher 提升 4.3%（LLM-as-Judge 指标），比基于树搜索的 CR-Planner 提升 25.7%，证明其在准确性和检索增强推理上的优越性。
* **效率提升**：相比 vanilla RL 方法，R1-Searcher++ 显著降低检索次数（比 R1-Searcher 减少 30.0%，比 Search-R1 减少 52.9%），表明其有效平衡了内部知识与外部检索，减少不必要开销。
* **泛化能力**：尽管仅在约 9000 个样本上训练，模型在域内和域外数据集上均表现出色，且在未训练的在线搜索场景（使用 Google API）中保持优越性能，证明其泛化能力。
* **实验设置合理性**：实验覆盖多个基准数据集，包含域内和域外测试，评价指标结合 F1 分数和 LLM-as-Judge，基线选择全面（从简单生成到复杂 RL 方法），消融实验验证了各组件（SFT 阶段、RL 阶段、组奖励、记忆机制）的必要性，设置全面且数据支持结论可信。

## Further Thoughts

R1-Searcher++ 的动态知识获取机制模仿人类‘先回忆后搜索’的思维模式，通过奖励引导模型自适应切换知识来源，未来可引入信心评估模块量化内部知识信任度以更精准决策；记忆机制将外部信息内化，启发是否能设计基于重要性或频率的选择性记忆策略避免知识过载；组奖励的精细化设计为 RL 在 RAG 场景中的应用提供了新思路，是否可探索多目标奖励函数平衡准确性、效率和知识增长？