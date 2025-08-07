---
title: "Balancing Information Accuracy and Response Timeliness in Networked LLMs"
pubDatetime: 2025-08-04T09:00:01+00:00
slug: "2025-08-networked-llm-timeliness"
type: "arxiv"
id: "2508.02209"
score: 0.6299554430405973
author: "grok-3-latest"
authors: ["Yigit Turkmen", "Baturalp Buyukates", "Melih Bastopcu"]
tags: ["LLM", "Multi-Agent System", "Query Routing", "Accuracy Optimization", "Response Timeliness"]
institution: ["Bilkent University", "University of Birmingham"]
description: "本文提出了一种网络化 LLM 系统，通过多代理协作和数学优化平衡信息准确性和响应及时性，并通过实验验证了联合响应的显著性能提升。"
---

> **Summary:** 本文提出了一种网络化 LLM 系统，通过多代理协作和数学优化平衡信息准确性和响应及时性，并通过实验验证了联合响应的显著性能提升。 

> **Keywords:** LLM, Multi-Agent System, Query Routing, Accuracy Optimization, Response Timeliness

**Authors:** Yigit Turkmen, Baturalp Buyukates, Melih Bastopcu

**Institution(s):** Bilkent University, University of Birmingham


## Problem Background

大型语言模型（LLMs）在实际部署中面临高计算成本、响应质量波动和低延迟需求的挑战，特别是在用户端应用中，单一模型难以同时满足信息准确性和响应及时性；
本文提出了一种网络化 LLM 系统，通过多个小型、专题化的 LLM 集群协作，探索如何在准确性和及时性之间找到平衡，解决多代理架构中缺乏正式分析模型的问题。

## Method

*   **系统架构**：设计了一个包含多用户、中央任务处理器和多个专题化 LLM 集群的网络化系统，用户提交二元查询（true/false），处理器将查询路由到对应的集群（每个集群有 m 个 LLM）。
*   **响应聚合机制**：任务处理器收集集群内 m 个 LLM 的独立响应，使用最大后验概率（MAP）估计器生成最终答案；MAP 估计器根据查询的先验概率（w_i）和 LLM 的准确率（p_i）调整决策阈值，形成自适应多数投票规则，而非简单的固定多数。
*   **及时性度量**：定义响应及时性为同一用户连续两个正确答案之间的时间间隔，包含查询等待时间和处理时间（传输时间为指数分布，处理时间固定）。
*   **优化策略**：通过调整集群中 LLM 数量 m，制定联合优化问题，最小化准确性（以联合准确率倒数表示）和系统时间的加权目标函数，权重参数 θ 控制两者平衡；优化考虑了准确性随 m 增加而提高，但系统时间也增加的权衡。
*   **理论分析**：推导了联合准确率（p_i,joint）的闭合表达式及近似，并分析了系统时间的数学期望，为优化提供理论支持。

## Experiment

*   **准确性提升**：在多个问答数据集（如 TriviaQA、Arc-Easy）上，使用 7 个开源预训练 LLM（如 Mistral-7B、Llama 3.1-8B）进行测试，聚合响应后的联合准确性显著高于单个 LLM；例如，在 Arc-Easy 上，7 个 LLM 联合准确性达 91.0%，比最佳单模型高 3.4 个百分点，尤其当集群内 LLM 准确性相近时提升更明显。
*   **实验设置合理性**：实验考虑了所有模型排列组合（m! 种），对正负查询平衡设计（w_i=0.5），减少顺序和数据集偏差；理论准确率与实证结果对比，提供了上下界和平均值估计，验证了模型适用性。
*   **权衡验证**：通过调整权重 θ 优化 m 值，展示了准确性和及时性的动态平衡；例如，θ=0.1（重视准确性）时最优 m=21，θ=0.4（重视及时性）时最优 m=11。
*   **局限性**：实验假设集群内 LLM 准确性和处理时间相同，但实际模型表现差异较大；理论模型假设响应独立性可能与实际不符。

## Further Thoughts

论文通过聚合多个专题化小型 LLM 提升整体性能的思路，启发我们在资源受限环境中（如边缘计算或个性化设备）利用多代理协作替代单一大型模型；
MAP 估计器的自适应多数投票机制提示可以在其他多代理系统中引入动态阈值调整，以适应任务先验分布和代理能力差异；
此外，集群内 LLM 准确性相近时联合效果更佳的发现，引导未来研究如何设计同质性或异质性代理组合以优化系统性能。