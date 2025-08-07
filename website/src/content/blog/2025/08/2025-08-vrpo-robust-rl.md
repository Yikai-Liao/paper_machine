---
title: "VRPO: Rethinking Value Modeling for Robust RL Training under Noisy Supervision"
pubDatetime: 2025-08-05T04:05:15+00:00
slug: "2025-08-vrpo-robust-rl"
type: "arxiv"
id: "2508.03058"
score: 0.6249124066769347
author: "grok-3-latest"
authors: ["Dingwei Zhu", "Shihan Dou", "Zhiheng Xi", "Senjie Jin", "Guoqiang Zhang", "Jiazheng Zhang", "Junjie Ye", "Mingxu Chai", "Enyu Zhou", "Ming Zhang", "Caishuang Huang", "Yunke Zhang", "Yuran Wang", "Tao Gui"]
tags: ["LLM", "RLHF", "Value Model", "Information Bottleneck", "Semantic Awareness"]
institution: ["Fudan University, College of Computer Science and Artificial Intelligence", "Honor Device Co., Ltd"]
description: "本文提出 VRPO 框架，通过信息瓶颈和语义感知辅助损失增强价值模型的鲁棒性，显著提升了 PPO 在噪声监督下的训练稳定性和泛化能力。"
---

> **Summary:** 本文提出 VRPO 框架，通过信息瓶颈和语义感知辅助损失增强价值模型的鲁棒性，显著提升了 PPO 在噪声监督下的训练稳定性和泛化能力。 

> **Keywords:** LLM, RLHF, Value Model, Information Bottleneck, Semantic Awareness

**Authors:** Dingwei Zhu, Shihan Dou, Zhiheng Xi, Senjie Jin, Guoqiang Zhang, Jiazheng Zhang, Junjie Ye, Mingxu Chai, Enyu Zhou, Ming Zhang, Caishuang Huang, Yunke Zhang, Yuran Wang, Tao Gui

**Institution(s):** Fudan University, College of Computer Science and Artificial Intelligence, Honor Device Co., Ltd


## Problem Background

在现实世界的强化学习（RL）尤其是基于人类反馈的强化学习（RLHF）中，噪声或不完美的奖励监督会导致模型在优势估计时忽略关键语义信息，进而损害策略的稳定性和泛化能力。
传统方法多关注奖励去噪或数据过滤，而忽略了价值模型在策略优化中的潜力，论文提出通过增强价值模型的鲁棒性来吸收不稳定信号，解决噪声监督下的训练不稳定问题。

## Method

*   **核心思想:** 提出 VRPO（Value Model Boosting for Robust Policy Optimization），一个以价值模型为核心的框架，通过信息论和语义感知机制增强 PPO 在噪声监督下的鲁棒性，将价值模型从被动预测器转变为主动噪声调节器。
*   **信息瓶颈机制（Variational Information Bottleneck, IB）:** 从信息论视角优化价值模型，通过变分下界优化一个目标函数，最大化与回报相关的预测信息（I(Z;Y)），同时最小化输入冗余信息（I(X;Z)），以学习紧凑且任务相关的表征，过滤掉无关噪声。具体实现上，使用高斯分布建模潜在表征 Z，并通过 KL 散度约束与标准高斯先验的差异，结合轻量级 MLP 预测回报。
*   **语义感知辅助损失（Semantic-Aware Auxiliary Losses）:** 引入冻结语言模型的熵（entropy）和困惑度（perplexity）作为指导信号，通过辅助损失增强价值模型对关键语义信息的关注。具体方法是，利用冻结语言模型头计算 token 级别的预测分布，针对高不确定性 token 子集（基于熵和困惑度阈值动态选择）施加正则化损失，促进模型内部特征空间与语言语义空间的部分对齐，避免在噪声奖励下偏离语义结构。
*   **实现细节:** 价值模型更新结合 MSE 损失、信息瓶颈 KL 损失和语义正则化损失，策略模型则基于 PPO 的裁剪目标更新，整体训练在采样轨迹和 GAE 优势估计的基础上进行。

## Experiment

*   **有效性:** VRPO 在数学推理、科学问答和多轮对话任务中均显著优于标准 PPO 和 GRPO，尤其在噪声监督下表现稳定。例如，在多轮对话任务中，VRPO 任务完成率（TCR）从 72.1% 提升至 75.9%，最终平均性能达 83.8%，远超 PPO（40.13%）和 GRPO（36.17%）。
*   **稳定性:** VRPO 通过稳定优势估计，避免了 PPO 和 GRPO 在噪声奖励下的性能崩溃，尤其在长度奖励偏见问题上，控制了响应长度（94-95），而 PPO 和 GRPO 响应长度显著膨胀。
*   **实验设置合理性:** 实验覆盖规则奖励和模型奖励两种噪声场景，使用自建 Honor-Dialogue Dataset 和多个公开基准（如 MATH500, GPQA），基于 Qwen 和 Llama 系列模型测试，任务类型和噪声类型设计全面。但未深入探讨超参数敏感性分析，可能存在优化空间。
*   **额外观察:** 解释方差（explained variance）随训练稳步提升，预测误差持续下降，表明价值模型在噪声环境下有效学习并引导策略更新。

## Further Thoughts

价值模型从被动到主动的转变启发了我：是否可以将‘主动调节’思想扩展到奖励模型或策略模型，设计动态加权奖励信号的机制，根据上下文语义重要性调整奖励？此外，信息瓶颈的计算复杂性较高，能否结合高效近似算法降低成本？另外，语义感知损失依赖冻结语言模型，若其存在偏差，是否会引入新噪声，是否可以通过多模型融合或动态更新解决？