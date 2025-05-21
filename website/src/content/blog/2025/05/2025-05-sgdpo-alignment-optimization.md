---
title: "SGDPO: Self-Guided Direct Preference Optimization for Language Model Alignment"
pubDatetime: 2025-05-18T14:19:23+00:00
slug: "2025-05-sgdpo-alignment-optimization"
type: "arxiv"
id: "2505.12435"
score: 0.5016794043670604
author: "grok-3-latest"
authors: ["Wenqiao Zhu", "Ji Liu", "Lulu Wang", "Jun Wu", "Yulun Zhang"]
tags: ["LLM", "Preference Optimization", "Alignment", "Gradient Flow", "Human Feedback"]
institution: ["HiThink Research", "Shanghai Jiao Tong University"]
description: "本文提出自引导直接偏好优化（SGDPO）算法，通过引入‘pilot’项调整梯度流，显著提升大型语言模型与人类偏好的对齐能力并增强训练稳定性。"
---

> **Summary:** 本文提出自引导直接偏好优化（SGDPO）算法，通过引入‘pilot’项调整梯度流，显著提升大型语言模型与人类偏好的对齐能力并增强训练稳定性。 

> **Keywords:** LLM, Preference Optimization, Alignment, Gradient Flow, Human Feedback

**Authors:** Wenqiao Zhu, Ji Liu, Lulu Wang, Jun Wu, Yulun Zhang

**Institution(s):** HiThink Research, Shanghai Jiao Tong University


## Problem Background

大型语言模型（LLMs）在与人类价值观和偏好对齐时面临挑战，传统方法如强化学习从人类反馈（RLHF）依赖复杂奖励模型，而直接偏好优化（DPO）虽简化流程，但存在生成人类偏好响应能力有限、优化结果不稳定及对监督微调（SFT）效果敏感等问题。
这些问题源于DPO在优化过程中梯度流不平衡，难以逃离鞍点，导致模型难以有效提升偏好响应的生成概率。

## Method

*   **核心思想:** 提出自引导直接偏好优化（SGDPO）算法，通过引入一个‘pilot’（引导）项，动态调整优化过程中的梯度流，以平衡偏好和非偏好奖励的更新，提升模型生成人类偏好响应的能力。
*   **具体实现:** 
    *   在目标函数中加入‘pilot’项，通过构造子序列（sub-sequence）来引导奖励更新，子序列从偏好和非偏好响应中随机抽取，长度由超参数r1和r2控制。
    *   子序列构造分为两种模式：Pilot_s（相同随机索引）和Pilot_d（不同随机索引），最终选择Pilot_d以增加学习空间的随机性，防止过拟合。
    *   ‘pilot’项通过调整偏好奖励的梯度更新比例（增大其更新幅度），以及防止非偏好奖励快速下降，实现优化目标G1和G2。
    *   理论分析支持方法设计，证明SGDPO能有效控制梯度流，增强偏好响应的生成概率。
*   **关键特点:** 不依赖绝对奖励值，通过可调参数提供灵活性，与其他方法（如IPO或Cal-DPO）形成区别；同时仅在优化阶段调整，不改变模型结构，保持较低计算开销。

## Experiment

*   **有效性:** SGDPO在MT-Bench基准测试上显著优于DPO，平均得分提升幅度为1.83%至9.19%；在条件基准测试（如GSM8K、MMLU等）中也取得最高平均得分（提升至0.0097），表明其在提升对齐性能方面的有效性。
*   **稳定性:** 训练奖励曲线显示，SGDPO在不同模型配置（Llama-3.1 8B、Qwen-2 7B）下表现出更一致的奖励模式，避免了DPO常见的奖励下降或不稳定现象，验证了梯度流平衡的理论假设。
*   **实验设置合理性:** 实验覆盖多种模型（Instruct和Base配置）和任务类型（开放式如MT-Bench和条件式如GSM8K），使用公开数据集（如UltraFeedback），设置全面；但在某些基准（如AlpacaEval-2上的Qwen-2）未完全超越基线，提示对齐效果可能与任务类型相关。
*   **计算开销:** 引入子序列重采样步骤，增加约0.4%的计算开销，整体影响较小。

## Further Thoughts

SGDPO中‘pilot’项的设计提供了一种动态调整优化方向的思路，启发我们思考是否可以通过引入多模型混合策略或动态调整‘pilot’模型权重，进一步适应不同任务或数据集特性；此外，论文提到的联邦学习适应性也值得探索，特别是在隐私保护和非IID数据场景下，自引导机制可能为分布式对齐提供新路径。