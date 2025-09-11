---
title: "Can AI Make Energy Retrofit Decisions? An Evaluation of Large Language Models"
pubDatetime: 2025-09-08T03:13:47+00:00
slug: "2025-09-llm-energy-retrofit"
type: "arxiv"
id: "2509.06307"
score: 0.4618778239308539
author: "grok-3-latest"
authors: ["Lei Shu", "Dong Zhao"]
tags: ["LLM", "Energy Efficiency", "Decision Making", "Context Sensitivity", "Reasoning"]
institution: ["Michigan State University"]
description: "本文系统评估了大型语言模型（LLMs）在建筑能源改造决策中的潜力，揭示其在技术目标（CO₂ 减排）下的有效性和在社会技术目标（回报期）下的局限，为生成式 AI 在能源领域的应用提供了重要参考。"
---

> **Summary:** 本文系统评估了大型语言模型（LLMs）在建筑能源改造决策中的潜力，揭示其在技术目标（CO₂ 减排）下的有效性和在社会技术目标（回报期）下的局限，为生成式 AI 在能源领域的应用提供了重要参考。 

> **Keywords:** LLM, Energy Efficiency, Decision Making, Context Sensitivity, Reasoning

**Authors:** Lei Shu, Dong Zhao

**Institution(s):** Michigan State University


## Problem Background

建筑能源改造决策的传统方法（如物理建模和数据驱动方法）面临泛化能力差和解释性低的挑战，限制了其在多样化住宅场景中的应用。
作者在智能互联社区（Smart and Connected Communities, S&CCs）的背景下，探索大型语言模型（LLMs）是否能通过处理复杂上下文信息并生成可解释的人类可读建议，克服这些局限，从而支持更有效的能源改造决策。

## Method

*   **研究目标与任务设计:** 评估七个领先的 LLMs（ChatGPT o1, ChatGPT o3, DeepSeek R1, Grok 3, Gemini 2.0, Llama 3.2, Claude 3.7）在建筑能源改造决策中的表现，任务分为两种场景：最大化 CO₂ 减排（技术目标）和最小化投资回报期（社会技术目标）。
*   **数据来源:** 使用 ResStock 2024.2 数据集，选取覆盖美国 49 个州的 400 个住宅样本，包含建筑特性、设备信息、 occupant 行为和能源使用参数，并结合 National Residential Efficiency Measures Database 补充改造成本数据。
*   **提示工程:** 设计结构化提示（prompt），包括 16 个改造方案的概述、角色分配（house retrofit specialist）和房屋特定信息（28 个关键参数），以确保模型理解任务并基于上下文生成推荐。
*   **评估框架:** 从四个维度评估模型表现：
    *   准确性（accuracy）：将 LLMs 推荐的改造方案与 EnergyPlus 模拟基准对比，计算 Top-1、Top-3 和 Top-5 匹配率。
    *   一致性（consistency）：使用 Fleiss’ Kappa 和 Cohen’s Kappa 评估模型间推荐的一致性。
    *   敏感性（sensitivity）：通过随机森林分类器分析模型对输入特征（如位置、建筑特性）的依赖程度。
    *   推理质量（reasoning）：定性分析模型（如 ChatGPT o3 和 DeepSeek R1）的推理逻辑是否符合工程原理。
*   **基准设置:** 使用 EnergyPlus 模拟结果作为基准，计算每个改造方案的 CO₂ 减排量和回报期，用于评估 LLMs 推荐的准确性。

## Experiment

*   **准确性表现:** 在 CO₂ 减排任务中，LLMs 表现较好，Top-1 准确率最高达 54.5%（Gemini 2.0），Top-5 最高达 92.8%（ChatGPT o3），表明模型在技术目标下能接近最优方案；但在回报期任务中，准确率较低，Top-1 最高仅 14.3%（DeepSeek R1），Top-5 最高 52.5%（Gemini 2.0），反映出模型在经济和社会因素上的局限。
*   **一致性结果:** 模型间一致性较低，Fleiss’ Kappa 值多为负值或接近零，表明推荐方案差异较大；高准确率模型（如 ChatGPT o3 和 Gemini 2.0）与其他模型的正确性判断差异更大。
*   **敏感性分析:** LLMs 对地理位置和建筑空间几何高度敏感，但对技术和 occupant 行为特征敏感性较低，与物理基准模型特征重要性有一定一致性，部分模型（如 Grok 3）特征偏好异常导致准确性下降。
*   **推理质量:** ChatGPT o3 和 DeepSeek R1 提供五步推理逻辑（基线假设、包络影响调整、系统能耗计算、设备能耗假设、结果比较），符合工程思维，但推理简化，缺乏上下文依赖的深入理解。
*   **实验设置合理性:** 实验覆盖多样化住宅样本和两种决策场景，评估维度系统，但未探索模型微调或不同提示策略的效果，可能限制结果潜力。

## Further Thoughts

论文揭示了 LLMs 在技术目标下未经微调即可达到高准确率（Top-5 达 92.8%）的潜力，这启发我们思考是否可以通过领域特定数据的微调或检索增强生成（RAG）技术进一步提升模型在社会技术场景中的表现；此外，提示敏感性问题提示未来研究可探索自适应提示生成机制，根据任务复杂度和上下文动态调整提示结构，以增强模型的上下文理解能力。