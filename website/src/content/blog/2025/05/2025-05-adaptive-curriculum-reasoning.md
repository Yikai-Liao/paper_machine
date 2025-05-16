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
description: "本文提出 ADCL 和 EGSR 两种人类启发的学习策略，通过动态调整课程难度和引导模型自重构专家知识，显著提升大型语言模型在复杂推理任务上的性能。"
---

> **Summary:** 本文提出 ADCL 和 EGSR 两种人类启发的学习策略，通过动态调整课程难度和引导模型自重构专家知识，显著提升大型语言模型在复杂推理任务上的性能。 

> **Keywords:** LLM, Curriculum Learning, Reinforcement Learning, Reasoning, Knowledge Assimilation

**Authors:** Enci Zhang, Xingang Yan, Wei Lin, Tianxiang Zhang, Qianchun Lu

**Institution(s):** ZTE Corporation, Wired Product Operation Division, Nanjing, China, Peking University, School of Electronic and Computer Engineering, Shenzhen, China


## Problem Background

大型语言模型（LLMs）在复杂推理任务（如数学问题求解）中，尽管取得了显著进展，但仍面临一致性解决复杂问题的挑战。
具体而言，模型对问题难度的感知在训练中动态变化（Difficulty Shift 现象），导致静态课程学习效果不佳；同时，现有强化学习方法（如 Zero-RL）依赖自身探索，难以突破预训练知识限制，学习新的推理能力。

## Method

*   **核心思想:** 借鉴人类学习策略，通过动态调整训练课程难度和引导模型内化专家知识，提升大型语言模型的推理能力。
*   **Adaptive Difficulty Curriculum Learning (ADCL):** 
    *   针对 Difficulty Shift 现象，ADCL 定期重新估计即将到来的数据批次难度，根据模型当前状态动态调整课程顺序。
    *   具体步骤包括：初始难度评估、按难度排序数据集并分批、迭代训练中更新模型参数、以及在每个批次后重新估计并排序下一个批次的难度。
    *   相比传统 Self-Paced Learning（SPL），ADCL 仅重新排序下一个批次，计算效率更高，适合 LLM 的后训练阶段。
*   **Expert-Guided Self-Reformulation (EGSR):** 
    *   针对能力边界限制，EGSR 通过专家指导帮助模型突破初始能力，不直接模仿专家轨迹，而是引导模型在自身概念框架内重构专家解决方案。
    *   具体实现：在模型初始探索无奖励（zero-reward）时，利用专家提供的答案或完整解决方案作为指导，生成接近模型当前策略（on-policy）的轨迹，避免 off-policy 数据分布不匹配问题。
    *   通过困惑度（Perplexity, PPL）分析验证，EGSR 生成的轨迹与模型自身分布更接近，训练更稳定。
*   **关键点:** 两种方法均不改变模型架构，ADCL 关注训练数据的动态组织，EGSR 关注知识内化，两者可协同使用以进一步提升性能。

## Experiment

*   **有效性:** 基于 Qwen2.5-7B 模型的实验表明，ADCL 相比预定义课程学习（PCL）在多个数学推理基准上性能提升明显（如 AIME25 从 23.33% 提升至 30.00%）；EGSR（尤其是结合专家解决方案和答案的 EGSR(s,a)）相比直接 off-policy 指导提升显著（如 AIME25 从 16.67% 提升至 30.00%）；两者组合效果最佳（如 AIME25 达到 33.33%，相比 RL 基线提升 16.66%）。
*   **能力扩展:** 通过 pass@32 指标验证，EGSR 显著扩展了模型能力边界（如 AIME25 提升 16.67%），表明其不仅优化现有技能，还帮助模型学习新知识。
*   **合理性与局限:** 实验设置合理，数据集筛选严格，基准覆盖多种难度，超参数标准；但实验仅基于单一模型（Qwen2.5-7B），未验证方法在更大规模模型上的普适性，未来可扩展至不同规模和领域的模型测试。

## Further Thoughts

ADCL 的动态难度调整机制启发我们可以在其他任务中引入类似的自适应数据组织方式，例如根据模型在自然语言理解任务中的实时表现调整训练样本难度；EGSR 的自重构理念可扩展至多轮对话或知识图谱构建中，通过让模型反复用自己的方式重述知识增强理解深度；此外，EGSR 平衡 on-policy 和 off-policy 数据的思路可进一步结合生成对抗网络（GAN）或变分自编码器（VAE），模拟专家知识分布以减少分布不匹配问题。