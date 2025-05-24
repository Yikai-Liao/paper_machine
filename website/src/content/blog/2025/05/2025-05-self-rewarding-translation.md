---
title: "SSR-Zero: Simple Self-Rewarding Reinforcement Learning for Machine Translation"
pubDatetime: 2025-05-22T13:08:25+00:00
slug: "2025-05-self-rewarding-translation"
type: "arxiv"
id: "2505.16637"
score: 0.7742987273654984
author: "grok-3-latest"
authors: ["Wenjie Yang", "Mao Zheng", "Mingyang Song", "Zheng Li"]
tags: ["LLM", "Machine Translation", "Reinforcement Learning", "Self-Rewarding", "Online Training"]
institution: ["Tencent Hunyuan"]
description: "本文提出简单自奖励强化学习框架（SSR），通过模型自评生成奖励信号显著提升机器翻译性能，并在结合外部奖励后达到开源模型的SOTA水平。"
---

> **Summary:** 本文提出简单自奖励强化学习框架（SSR），通过模型自评生成奖励信号显著提升机器翻译性能，并在结合外部奖励后达到开源模型的SOTA水平。 

> **Keywords:** LLM, Machine Translation, Reinforcement Learning, Self-Rewarding, Online Training

**Authors:** Wenjie Yang, Mao Zheng, Mingyang Song, Zheng Li

**Institution(s):** Tencent Hunyuan


## Problem Background

当前大型语言模型（LLMs）在机器翻译（MT）任务中高度依赖外部监督信号（如人工标注参考数据或预训练奖励模型），这些资源获取成本高昂且难以扩展，尤其在低资源语言场景下；
本文提出了一种无需外部监督、完全在线的自我奖励强化学习框架，旨在通过模型自身的判断能力生成奖励信号，提升翻译性能并解决可持续性和扩展性问题。

## Method

*   **核心思想:** 提出一种简单自奖励（Simple Self-Rewarding, SSR）强化学习框架，让模型同时扮演演员（Actor）和评判者（Judge）角色，通过自评生成奖励信号优化翻译性能，无需外部参考数据或奖励模型。
*   **具体实现:** 
    *   基于预训练模型Qwen2.5-7B，设计完全在线的训练流程：模型首先作为演员生成多个候选翻译结果；
    *   随后切换为评判者角色，使用特定提示（Judge Prompt）对每个候选翻译评分（0-100分），评分通过正则表达式提取作为奖励信号；
    *   引入格式奖励（Format Reward）确保输出符合预定格式，综合自奖励和格式奖励形成最终奖励；
    *   使用Group Related Policy Optimization (GRPO)算法，根据奖励信号更新模型参数，迭代训练直至性能收敛。
*   **增强版本:** 在SSR-X-Zero模型中，结合外部奖励模型（如COMET）进一步提升性能，形成自奖励与外部奖励的混合策略。
*   **关键特点:** 训练过程完全在线，仅需少量单语数据（13K示例），无需大规模标注数据，降低了资源依赖。

## Experiment

*   **有效性:** SSR-Zero-7B模型在英文-中文双向翻译任务上，使用仅13K单语数据训练，显著提升性能，ZH→EN方向平均得分从73.97提升至87.37（+18.11%），EN→ZH方向从75.29提升至86.39（+14.74%），优于多个开源MT专用模型（如TowerInstruct-13B）和更大规模通用模型（如Qwen2.5-32B-Instruct）。
*   **优越性:** SSR-X-Zero-7B结合外部奖励（COMET）后，在参数小于72B的开源模型中达到SOTA性能，甚至在EN→ZH方向超越闭源模型GPT-4o（86.39 vs 84.71）。
*   **实验设置:** 实验覆盖WMT23、WMT24和Flores200多个基准数据集，采用XCOMET-XXL和COMETKIWI-XXL评估指标，设置较为全面；但仅限于英文-中文语言对，泛化性待验证。
*   **局限性:** SSR-Zero-7B在训练后期出现格式问题（输出多余引号导致评估分数下降），显示出自奖励机制在长期训练中的稳定性问题；SSR-X-Zero-7B通过外部奖励缓解了这一问题。
*   **开销:** 训练SSR-Zero-7B约17小时，SSR-X-Zero-7B约42小时，使用8个GPU，计算成本相对可控。

## Further Thoughts

自奖励与外部奖励的互补性是一个重要启发，SSR-X-Zero-7B通过结合两者取得最佳性能，提示未来可以在资源有限时先用自奖励快速提升模型，随后引入少量外部监督微调；此外，自奖励机制是否可扩展至其他任务（如文本生成、问答），通过模型自评实现持续改进？是否可以通过少样本提示或链式思维提示提升自评质量，避免初始能力不足导致的局部最优？这些方向值得进一步探索。