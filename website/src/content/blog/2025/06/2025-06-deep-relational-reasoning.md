---
title: "Are Large Language Models Capable of Deep Relational Reasoning? Insights from DeepSeek-R1 and Benchmark Comparisons"
pubDatetime: 2025-06-29T07:37:49+00:00
slug: "2025-06-deep-relational-reasoning"
type: "arxiv"
id: "2506.23128"
score: 0.7517200814186581
author: "grok-3-latest"
authors: ["Chi Chiu So", "Yueyue Sun", "Jun-Min Wang", "Siu Pang Yung", "Anthony Wai Keung Loh", "Chun Pong Chau"]
tags: ["LLM", "Relational Reasoning", "Deep Reasoning", "Chain of Thought", "Benchmarking"]
institution: ["The Hong Kong Polytechnic University", "Beijing Institute of Technology", "The University of Hong Kong"]
description: "本文通过家族树和一般图推理基准测试，首次全面评估了具备深度推理能力的大型语言模型（特别是 DeepSeek-R1）在关系推理上的表现，揭示其在小规模问题上的显著优势及 token 限制带来的局限。"
---

> **Summary:** 本文通过家族树和一般图推理基准测试，首次全面评估了具备深度推理能力的大型语言模型（特别是 DeepSeek-R1）在关系推理上的表现，揭示其在小规模问题上的显著优势及 token 限制带来的局限。 

> **Keywords:** LLM, Relational Reasoning, Deep Reasoning, Chain of Thought, Benchmarking

**Authors:** Chi Chiu So, Yueyue Sun, Jun-Min Wang, Siu Pang Yung, Anthony Wai Keung Loh, Chun Pong Chau

**Institution(s):** The Hong Kong Polytechnic University, Beijing Institute of Technology, The University of Hong Kong


## Problem Background

大型语言模型（LLMs）在深度关系推理（Deep Relational Reasoning）方面的能力被认为是通向通用人工智能（AGI）的关键瓶颈。
现有研究多集中于浅层或模式化推理基准，缺乏对复杂多步逻辑推理能力的全面评估，尤其是在处理大规模关系结构时，模型是否受限于架构（如 token 限制）仍未明晰。
本文聚焦于评估具备深度推理能力的 LLMs（特别是 DeepSeek-R1）在关系推理任务上的表现，探索其潜力与局限。

## Method

*   **核心目标:** 评估大型语言模型在深度关系推理任务上的能力，重点关注 DeepSeek-R1，并与其他模型进行对比。
*   **基准设计:** 提出了两个关系推理基准测试：
    *   **家族树推理（Family Tree Reasoning）**：基于基本关系（如父母、子女）推导复杂家族关系（如姐妹、姑姑、孙子、父系曾姑姑）。
    *   **一般图推理（General Graph Reasoning）**：基于图的边关系推导全局结构属性（如连通性和最短路径）。
*   **数据生成与提示:** 
    *   通过布尔矩阵和邻接矩阵生成逻辑一致的家族树和图数据，并将其转化为自然语言提示。
    *   采用零样本提示（Zero-shot Prompting）方式，不提供任务示例，依赖模型预训练知识和推理能力。
*   **模型评估:** 对比 DeepSeek-R1、DeepSeek-V3 和 GPT-4o 在不同问题规模（n=10, 20, 40）上的表现，每种任务运行100次以确保统计可靠性。
*   **评估指标:** 使用 F1 分数作为主要指标，针对二分类任务计算标准 F1 分数，针对多分类任务（如最短路径）计算宏平均 F1 分数，同时对无效输出（如格式错误或矩阵形状不符）给予零分处理。

## Experiment

*   **有效性:** DeepSeek-R1 在小规模问题（n=10 和 n=20）上显著优于 DeepSeek-V3 和 GPT-4o，F1 分数在复杂任务中最高达 0.98，展现出深度推理能力的明显优势。
*   **局限性:** 当问题规模增大到 n=40 时，所有模型性能急剧下降，包括 DeepSeek-R1，主要由于 token 限制导致推理过程截断或输出不完整，F1 分数在多个任务中降至 0。
*   **设置合理性:** 实验设计全面，涵盖不同任务类型和问题规模，数据生成逻辑严谨，评估指标适合分类任务下的不平衡数据；但零样本提示可能限制模型潜力，未探讨不同提示策略的影响，且未充分考虑 token 限制对大规模问题的冲击。
*   **深入分析:** DeepSeek-R1 的长思维链（Long Chain-of-Thought）是其性能优势的关键，展现出规划、验证和抽象能力，但推理过程偶尔不连贯，提示内部推理机制可能存在不稳定性。

## Further Thoughts

论文提出多模态推理（如结合图像或视频输入）可能增强关系推理能力，这启发我思考：是否可以通过可视化家族树或图结构作为输入，绕过 token 限制，提升模型在复杂任务中的表现？
此外，DeepSeek-R1 推理轨迹的不连贯性提示我们：是否可以通过分析错误推理轨迹，揭示模型认知模式，设计专门的训练目标或分层推理模块来增强推理结构性和可解释性？
进一步思考，是否可以引入外部知识库或逻辑推理工具辅助 LLMs，使其从独立推理者转变为协同推理者，从而突破当前架构限制？