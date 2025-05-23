---
title: "Toward Effective Reinforcement Learning Fine-Tuning for Medical VQA in Vision-Language Models"
pubDatetime: 2025-05-20T06:12:20+00:00
slug: "2025-05-rl-finetuning-medical-vqa"
type: "arxiv"
id: "2505.13973"
score: 0.6220423976407279
author: "grok-3-latest"
authors: ["Wenhui Zhu", "Xuanzhao Dong", "Xin Li", "Peijie Qiu", "Xiwen Chen", "Abolfazl Razi", "Aris Sotiras", "Yi Su", "Yalin Wang"]
tags: ["LLM", "Reinforcement Learning", "Vision-Language Model", "Medical VQA", "Semantic Alignment", "Reasoning"]
institution: ["Arizona State University", "Clemson University", "Washington University in St. Louis", "Banner Alzheimer’s Institute"]
description: "本文通过系统分析GRPO-based RL微调在医疗VQA中的应用，揭示初始化策略、语义对齐、长链推理奖励和偏差优化的影响，证明RL微调显著提升了多模态大语言模型的准确性和推理质量。"
---

> **Summary:** 本文通过系统分析GRPO-based RL微调在医疗VQA中的应用，揭示初始化策略、语义对齐、长链推理奖励和偏差优化的影响，证明RL微调显著提升了多模态大语言模型的准确性和推理质量。 

> **Keywords:** LLM, Reinforcement Learning, Vision-Language Model, Medical VQA, Semantic Alignment, Reasoning

**Authors:** Wenhui Zhu, Xuanzhao Dong, Xin Li, Peijie Qiu, Xiwen Chen, Abolfazl Razi, Aris Sotiras, Yi Su, Yalin Wang

**Institution(s):** Arizona State University, Clemson University, Washington University in St. Louis, Banner Alzheimer’s Institute


## Problem Background

本文聚焦于医疗视觉问答（Medical VQA）任务中多模态大语言模型（MLLMs）的强化学习（RL）微调问题。
尽管基于RL的微调方法（如Group Relative Policy Optimization, GRPO）在通用领域表现出色，但在医疗场景中直接应用时，模型输出往往难以满足临床准确性和领域对齐的要求。
关键问题是如何通过RL微调提升模型在医疗VQA中的推理质量和回答准确性，同时克服传统监督微调（SFT）在复杂医疗场景中推理能力不足的局限性。

## Method

*   **核心思想:** 利用基于GRPO的强化学习微调方法，通过设计特定奖励机制和优化策略，提升多模态大语言模型在医疗VQA任务中的临床相关性和推理能力。
*   **具体实现:** 
    *   **初始化策略:** 对比从头训练与基于指令微调模型的训练，分析两者在推理探索和领域对齐上的权衡。
    *   **医疗语义对齐奖励:** 引入基于专家LLM（如BioGPT和BioMistral）的奖励机制，通过提示工程评估模型推理是否符合临床逻辑，奖励符合标准的输出（奖励值为1），否则为0。
    *   **长链推理奖励机制:** 测试两种基于长度的奖励设计——Extended Chain Reward (ECR) 直接奖励长输出，Correctness-Weighted Length Reward (CWR) 仅在答案正确时奖励长输出，旨在探索长推理对性能的影响。
    *   **偏差优化 (Dr.GRPO):** 改进GRPO算法，移除标准差归一化和token级平均，采用简单的组均值差作为优势估计，减少模型生成冗长错误回答的倾向。
    *   **对比分析:** 将GRPO方法与多种SFT策略（如LoRA、完整微调、DPO）以及公开模型进行对比，验证RL微调的优越性。
*   **关键点:** 方法不依赖额外的奖励或价值模型，而是通过采样多组响应并计算平均奖励作为基准，简化优化过程，同时通过KL散度正则化控制训练稳定性。

## Experiment

*   **有效性:** GRPO-based RL微调在医疗VQA任务中显著优于传统SFT方法，基础GRPO模型准确性达58.04%，加入语义对齐后提升至59.86%，Dr.GRPO进一步提升至61.09%，而最佳SFT方法仅为52.00%。
*   **具体维度分析:** 指令微调初始化提升了准确性和语言流畅性（困惑度从14.16降至13.28）；语义对齐奖励显著提高相似度评分（从0.21到0.46）；长链推理奖励（ECR）导致准确性下降（至50.17%），CWR略有改善（54.68%）；Dr.GRPO在准确性、困惑度和推理奖励上均有提升。
*   **优越性:** GRPO方法在推理质量和临床相关性上优于SFT，SFT模型常缺乏中间推理步骤，而RL微调能自动探索有意义的推理行为。
*   **实验设置合理性:** 实验基于Qwen2-VL-2B模型，在PMC-VQA数据集子集（10K训练+7K测试）上进行，涵盖多个维度（初始化、奖励设计、偏差优化）和对比模型，设置较为全面；但数据集规模较小，仅用2B参数模型，泛化性可能受限。
*   **开销:** 主要计算开销来自GRPO每步采样8个响应及语义对齐奖励的专家模型评估，但整体仍在可接受范围内（使用4个A100 GPU）。

## Further Thoughts

本文的医疗语义对齐奖励机制启发我在其他领域任务中设计特定领域的奖励函数，利用专家模型或知识库引导输出；长链推理奖励的负面效应提醒我奖励设计需避免形式主义输出，未来可探索动态平衡准确性和推理深度的奖励策略；Dr.GRPO对偏差的优化让我思考是否可以通过无偏优化或混合初始化策略（如冷启动结合指令微调）进一步提升RL微调的稳定性和泛化能力。