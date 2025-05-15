---
title: "Learning Like Humans: Advancing LLM Reasoning Capabilities via Adaptive Difficulty Curriculum Learning and Expert-Guided Self-Reformulation"
pubDatetime: 2025-05-13T09:10:48+00:00
slug: "2025-05-adaptive-curriculum-reasoning"
type: "arxiv"
id: "2505.08364"
score: 0.7106482115503522
author: "grok-3-latest"
authors: ["Enci Zhang", "Xingang Yan", "Wei Lin", "Tianxiang Zhang", "Qianchun Lu"]
tags: ["LLM", "Curriculum Learning", "Reinforcement Learning", "Reasoning", "Knowledge Assimilation"]
institution: ["ZTE Corporation, Wired Product Operation Division, Nanjing, China", "Peking University, School of Electronic and Computer Engineering, Shenzhen, China"]
description: "本文提出受人类学习启发的 ADCL 和 EGSR 策略，通过动态难度调整和专家引导自重构显著提升大型语言模型在复杂推理任务中的表现，尤其是在数学推理领域。"
---

> **Summary:** 本文提出受人类学习启发的 ADCL 和 EGSR 策略，通过动态难度调整和专家引导自重构显著提升大型语言模型在复杂推理任务中的表现，尤其是在数学推理领域。 

> **Keywords:** LLM, Curriculum Learning, Reinforcement Learning, Reasoning, Knowledge Assimilation

**Authors:** Enci Zhang, Xingang Yan, Wei Lin, Tianxiang Zhang, Qianchun Lu

**Institution(s):** ZTE Corporation, Wired Product Operation Division, Nanjing, China, Peking University, School of Electronic and Computer Engineering, Shenzhen, China


## Problem Background

大型语言模型（LLMs）在复杂推理任务（如数学推理）中仍面临挑战，尤其是在训练过程中模型对问题难度的动态感知变化（Difficulty Shift）和能力边界受限于预训练知识的问题。
现有的训练范式（如 Zero-RL）依赖静态课程学习和自身探索，难以适应模型能力变化或突破初始知识限制，导致训练效果不佳和推理能力提升受限。

## Method

*   **核心思想:** 借鉴人类学习策略，通过动态调整课程难度和引导模型自重构专家知识，提升 LLMs 的复杂推理能力。
*   **Adaptive Difficulty Curriculum Learning (ADCL):** 
    *   针对难度感知动态变化的问题，ADCL 在训练过程中周期性地根据模型当前状态重新估计即将到来的数据批次难度，并对批次内部样本重新排序，确保课程始终与模型能力对齐。
    *   与传统自适应学习（如 Self-Paced Learning）不同，ADCL 仅重新排序下一个批次而非整个数据集，从而在计算效率和适应性之间取得平衡。
    *   具体步骤包括初始难度评估、迭代训练与动态调整、以及基于模型当前参数的批次内重新排序。
*   **Expert-Guided Self-Reformulation (EGSR):** 
    *   针对能力边界限制问题，EGSR 通过引入专家指导（而非直接模仿），引导模型在自身概念框架内重构专家解决方案，避免 off-policy 数据与模型策略的分布不匹配。
    *   具体实现上，EGSR 在模型探索失败（即所有轨迹奖励为零）时，使用专家解法或答案作为指导，生成更接近模型当前策略的轨迹，并结合强化学习目标（如 GRPO）进行优化。
    *   EGSR 提供两种指导模式：仅使用专家答案（EGSR(a)）或结合专家解法和答案（EGSR(s,a)），后者效果更优。
*   **关键点:** 两种方法均不直接修改模型预训练参数，而是通过训练过程中的动态调整和引导实现能力提升，且可协同使用以获得更大收益。

## Experiment

*   **有效性:** 基于 Qwen2.5-7B 模型的实验表明，ADCL 相比预定义课程学习（PCL）在多个数学推理基准（如 AIME24, AIME25）上显著提升性能，例如 AIME25 从 23.33% 提升至 30.00%；EGSR（尤其是 EGSR(s,a)）相比直接 off-policy 指导方法大幅改进，例如 AIME25 从 16.67% 提升至 30.00%。
*   **组合优势:** ADCL 和 EGSR 组合使用时效果最佳，在 AIME25 上达到 33.33%（比标准 RL 提升 16.66%），在 Minervamath 上提升至 25.74%（比标准 RL 提升 7.73%），显示出协同效应。
*   **实验设置合理性:** 数据集（BaseSet-7K 和 AugSet-10K）经过筛选确保难度分布合理，基准选择（如 MATH500, AIME24/25）覆盖不同难度任务，pass@8 和 pass@32 指标的使用减少了小数据集方差并评估了能力边界，整体设计全面。
*   **局限性:** 实验聚焦数学推理，未涉及其他推理任务，泛化性待验证；计算开销（如 ADCL 的难度重估）未详细分析，可能影响实际应用。

## Further Thoughts

ADCL 的动态难度调整机制可推广至其他领域（如长文本理解或多任务学习），以适应模型能力变化；EGSR 的自重构学习范式为解决 RL 中分布不匹配问题提供了新思路，可应用于对话系统或代码生成；此外，模拟人类认知过程（如引入学习信心或任务优先级）可能进一步优化 AI 训练策略。