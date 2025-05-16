---
title: "Lost in Transmission: When and Why LLMs Fail to Reason Globally"
pubDatetime: 2025-05-13T00:25:23+00:00
slug: "2025-05-bapo-bandwidth-reasoning"
type: "arxiv"
id: "2505.08140"
score: 0.691159708644794
author: "grok-3-latest"
authors: ["Tobias Schnabel", "Kiran Tomlinson", "Adith Swaminathan", "Jennifer Neville"]
tags: ["LLM", "Attention Mechanism", "Bandwidth Constraint", "Reasoning", "Chain of Thought"]
institution: ["Microsoft Research", "Netflix"]
description: "本文提出 BAPO 模型，通过形式化 LLMs 的有效带宽限制解释其在全局推理任务上的失败，并通过实验验证预测能力，同时揭示思维链降低带宽需求的潜力。"
---

> **Summary:** 本文提出 BAPO 模型，通过形式化 LLMs 的有效带宽限制解释其在全局推理任务上的失败，并通过实验验证预测能力，同时揭示思维链降低带宽需求的潜力。 

> **Keywords:** LLM, Attention Mechanism, Bandwidth Constraint, Reasoning, Chain of Thought

**Authors:** Tobias Schnabel, Kiran Tomlinson, Adith Swaminathan, Jennifer Neville

**Institution(s):** Microsoft Research, Netflix


## Problem Background

大型语言模型（LLMs）在处理需要整合整个输入信息的全局推理问题（如图的可达性、多数投票）时表现不佳。
论文提出，这种失败源于模型内部信息流的‘有效带宽’（effective bandwidth）限制，即 LLMs 无法通过注意力机制在残差流之间准确传递足够的信息，尤其是在因果注意力机制下，早期 token 信息难以有效影响后续 token 的预测。

## Method

*   **核心思想:** 提出有界注意力前缀预言机（Bounded Attention Prefix Oracle, BAPO）模型，用于形式化 LLMs 的内部通信带宽限制，并分析问题对带宽的需求。
*   **模型定义:** BAPO 通过两个参数限制信息流：前缀带宽（prefix bandwidth，限制前缀计算结果传递的位数）和注意力带宽（attention bandwidth，限制可精确关注的前缀 token 数量）。
*   **问题分类:** 将问题分为 BAPO-easy（需要常数带宽即可解决）和 BAPO-hard（需要超常数带宽），通过理论证明分析具体问题的带宽需求，例如证明 Reachability 和 Majority 为 BAPO-hard，而 Index 和 Equality 为 BAPO-easy。
*   **思维链（CoT）作用:** 理论上证明，思维链（Chain of Thought）可以通过将 BAPO-hard 问题分解为一系列 BAPO-easy 步骤来降低带宽需求，甚至使常数带宽的 BAPO 具备图灵完备性。
*   **适用性:** BAPO 模型抽象了 Transformer 的低级细节，专注于信息流需求，提供了一个广泛适用的分析框架。

## Experiment

*   **有效性:** 实验验证了 BAPO 模型的预测能力，在 BAPO-easy 任务（如 Index、Equality）上，GPT-4o、Claude 3.5 Sonnet 和 Gemini 1.5 Pro 等 LLMs 表现良好，准确率较高且随输入规模变化不大；而在 BAPO-hard 任务（如 Reachability、Majority）上，准确率显著下降，尤其在输入规模 n=200 时接近随机猜测。
*   **全面性:** 实验设置覆盖了多个模型家族、不同输入规模（n=6 到 200）、多种任务类型（包括理论问题和真实世界任务如酒店评论情感判断和代码变量追踪），数据生成过程设计合理，确保无明显捷径。
*   **CoT 效果:** 引入思维链（CoT）后，BAPO-hard 任务的性能有所提升，尤其在小规模输入时，但在大规模输入下仍存在显著下降，表明 CoT 并非完全解决带宽限制的方案。
*   **结论:** 实验结果支持了有效带宽限制的假设，BAPO 模型成功预测了 LLMs 的失败模式。

## Further Thoughts

论文提出的‘有效带宽’概念启发了我思考 LLMs 的设计权衡：低带宽可能有助于泛化能力，而高带宽需求可能与精确推理冲突，是否可以通过动态调整带宽或设计混合架构（如结合外部工具或记忆增强）来优化这一权衡？此外，CoT 将复杂问题分解为低带宽步骤的理论结果提示我们，可以在训练时引入低带宽推理目标，或通过特定提示策略增强模型的分步推理能力。