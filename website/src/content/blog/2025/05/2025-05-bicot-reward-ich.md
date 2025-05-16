---
title: "Fusing Bidirectional Chains of Thought and Reward Mechanisms A Method for Enhancing Question-Answering Capabilities of Large Language Models for Chinese Intangible Cultural Heritage"
pubDatetime: 2025-05-13T02:05:25+00:00
slug: "2025-05-bicot-reward-ich"
type: "arxiv"
id: "2505.08167"
score: 0.5138455249156262
author: "grok-3-latest"
authors: ["Ruilin Liu", "Zhixiao Zhao", "Jieqiong Li", "Chang Liu", "Dongbo Wang"]
tags: ["LLM", "Reasoning", "Reinforcement Learning", "Domain Adaptation", "Question Answering"]
institution: ["Nanjing Agricultural University"]
description: ""
---

> **Summary:**  

> **Keywords:** LLM, Reasoning, Reinforcement Learning, Domain Adaptation, Question Answering

**Authors:** Ruilin Liu, Zhixiao Zhao, Jieqiong Li, Chang Liu, Dongbo Wang

**Institution(s):** Nanjing Agricultural University


## Problem Background

大型语言模型（LLMs）在领域特定应用（如非物质文化遗产，ICH）中面临灾难性遗忘、知识偏差和错误传承等问题，尤其在复杂问答任务中表现不佳。
本文旨在通过创新训练方法提升模型在ICH领域的问答能力，解决推理能力不足和输出不准确的关键问题，同时增强模型的可解释性和知识激活能力。

## Method

*   **核心思想:** 提出一种结合双向思维链（Bidirectional Chains of Thought, Bi-CoT）和奖励机制的训练方法，基于ICH-Qwen模型，旨在通过多视角推理和细粒度反馈提升问答能力。
*   **双向思维链（Bi-CoT）:** 包括正向推理（从问题到结论）和逆向推理（从结论回溯条件或证据），通过双重推理过程交叉验证答案可靠性，激活模型潜在知识，减少推理错误。例如，对于‘中国皮影戏起源于哪个历史时期？’的问题，正向推理从历史记载推导出‘西汉’，逆向推理则从‘西汉’回溯关键证据（如汉武帝时期的事件），从而强化知识关联。
*   **奖励机制:** 设计一个多维奖励函数，基于格式、关键词匹配和最长公共子序列（Longest Common Subsequence）评估输出质量。奖励函数对正向推理、逆向问题、逆向推理和最终答案分别赋予权重（如最终答案权重为0.4，其他部分为0.2），并通过关键词匹配（每匹配一个得0.25分）鼓励内容准确性，引导模型生成符合人类期望的标准化输出。
*   **训练流程:** 采用强化学习方法（如Proximal Policy Optimization, PPO），通过采样多组输出（每组8个样本），计算相对奖励（relative reward）来优化策略。引入冷启动（cold start）策略，先用高质量数据微调一轮，加速后续训练收敛。同时，使用KL散度约束和概率比裁剪确保训练稳定性。
*   **关键点:** 方法不依赖特定领域预训练知识，注重推理过程而非仅关注最终结果，避免了传统微调中的灾难性遗忘问题，并通过奖励函数提供可解释性强的反馈信号。

## Experiment

*   **有效性:** 在ICH-Qwen模型上，方法在非物质文化遗产问答任务中显著优于基线方法，准确率从0-shot的45.54%提升至65.35%，BLEU-4从8.10提升至12.73，Rouge-L从20.88提升至26.73，相比最佳基线（Bi-CoT Distillation）准确率提升12.87%。
*   **全面性与合理性:** 实验覆盖多个模型（如Internlm3-8B-Instruct、GLM-4-9B-0414、Meta-Llama-3.1-8B-Instruct）和多领域数据集（如金融、Wikidata、StrategyQA），验证了方法的泛化能力。消融实验表明，去掉Bi-CoT或奖励机制后性能下降明显，证明两者结合的必要性。
*   **训练效率:** 冷启动策略使奖励收敛速度从2000步缩短至750步，显著降低训练成本，同时保持最终性能。
*   **局限性:** 方法在开放性问题上的评价仍需改进，当前奖励函数对多答案场景的适应性有限。

## Further Thoughts

双向思维链（Bi-CoT）的正逆向推理结合为复杂推理任务提供了新思路，未来可探索是否能通过动态调整奖励权重，根据任务难度或领域特性进一步优化训练效果；此外，奖励函数的设计启发我们可以在其他领域（如医疗、法律）中引入专家知识作为奖励标准，以提升模型对开放性问题的处理能力。