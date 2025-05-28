---
title: "Foundations of Top-$k$ Decoding For Language Models"
pubDatetime: 2025-05-25T23:46:34+00:00
slug: "2025-05-topk-decoding-bregman"
type: "arxiv"
id: "2505.19371"
score: 0.7195823953701845
author: "grok-3-latest"
authors: ["Georgy Noarov", "Soham Mallick", "Tao Wang", "Sunay Joshi", "Yan Sun", "Yangxinyu Xie", "Mengxin Yu", "Edgar Dobriban"]
tags: ["LLM", "Decoding", "Sparsity", "Optimization", "Sampling"]
institution: ["University of Pennsylvania"]
description: "本文提出基于稀疏 Bregman 散度最小化的理论框架，解释并泛化了 Top-k 解码方法，通过高效算法和实验验证了新解码策略的潜力。"
---

> **Summary:** 本文提出基于稀疏 Bregman 散度最小化的理论框架，解释并泛化了 Top-k 解码方法，通过高效算法和实验验证了新解码策略的潜力。 

> **Keywords:** LLM, Decoding, Sparsity, Optimization, Sampling

**Authors:** Georgy Noarov, Soham Mallick, Tao Wang, Sunay Joshi, Yan Sun, Yangxinyu Xie, Mengxin Yu, Edgar Dobriban

**Institution(s):** University of Pennsylvania


## Problem Background

大型语言模型（LLMs）在文本生成中广泛使用 Top-k 解码方法，通过截断低概率 token 来提高生成质量，但缺乏理论基础来解释其有效性或指导改进。
论文旨在构建一个理论框架，通过稀疏概率分布恢复的视角，解释 Top-k 解码并提出泛化策略，解决现有方法缺乏理论支持的关键问题。

## Method

*   **核心思想:** 将解码过程建模为一个优化问题，通过最小化 Bregman 散度（一种广义距离度量）并引入 ℓ₀ 正则化诱导稀疏性，恢复一个稀疏的概率分布用于 token 采样。
*   **解码方式:** 提出两种解码方式：原问题解码（Primal Decoding，最小化 Bregman 散度第一参数）和对偶问题解码（Dual Decoding，最小化第二参数），分别对应不同的优化视角和应用场景。
*   **优化策略:** 尽管 ℓ₀ 正则化通常导致组合优化问题，论文证明对于可分离的 Bregman 散度，优化可以通过贪婪选择（选择概率最高的 k 个 token）和二分搜索（快速找到最优 k）高效解决，基于目标函数在 k 上的离散凸性。
*   **具体实例:** 以 α-Bregman 解码为例，基于 Tsallis α-熵生成解码策略，当 α 趋近 1 时等价于 KL 散度（即 Top-k 解码），其他 α 值则生成新策略，如对高概率 token 赋予更多权重或反之。
*   **实现细节:** 解码过程分为选择稀疏模式（sparsity pattern）和重新归一化（renormalization）两步，重新归一化通过求解 Bregman 投影问题实现，确保输出为有效概率分布。

## Experiment

*   **有效性:** 在开放式文本生成任务（WebText 测试集）中，Bregman 解码（α=1.5 和 α=2.0）在困惑度和重复率上与 Top-k 解码表现相当，α=2.0 时困惑度差距最小；在 GSM8K 数学推理任务中，Bregman 解码准确率与 Top-k 相当，尤其在高温度（如 1.5）下性能下降较慢，表现出更好鲁棒性。
*   **实验设置:** 使用 GPT-2 Large 和 LLaMA 3.1 8B 模型，涵盖开放式文本生成和数学推理任务，评估指标包括困惑度差异和重复率差异；实验分为自适应 k（完全评估）和固定 k（部分评估）两种模式。
*   **合理性与局限:** 实验设置较为全面，验证了理论框架的实际可行性，但规模有限，未与其他流行解码方法（如 Top-p 或 Beam Search）广泛对比，且未深入分析计算开销。

## Further Thoughts

论文提出的 Bregman 散度框架启发我们可以在解码中探索更多形式的散度函数，以适应不同任务需求，例如在需要更高多样性的场景中选择强调低概率 token 的散度；此外，自适应 k 的方法提示可以在实际应用中引入动态稀疏度调整机制，根据上下文或用户偏好优化生成质量。