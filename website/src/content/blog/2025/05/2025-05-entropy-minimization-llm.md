---
title: "The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning"
pubDatetime: 2025-05-21T05:39:11+00:00
slug: "2025-05-entropy-minimization-llm"
type: "arxiv"
id: "2505.15134"
score: 0.7936242106437946
author: "grok-3-latest"
authors: ["Shivam Agarwal", "Zimin Zhang", "Lifan Yuan", "Jiawei Han", "Hao Peng"]
tags: ["LLM", "Entropy Minimization", "Reasoning", "Post-Training", "Inference Scaling"]
institution: ["University of Illinois Urbana-Champaign"]
description: "本文通过熵最小化提出三种无监督方法（EM-FT, EM-RL, EM-INF），显著提升大型语言模型在复杂推理任务上的性能，无需标注数据且在推理时效率更高。"
---

> **Summary:** 本文通过熵最小化提出三种无监督方法（EM-FT, EM-RL, EM-INF），显著提升大型语言模型在复杂推理任务上的性能，无需标注数据且在推理时效率更高。 

> **Keywords:** LLM, Entropy Minimization, Reasoning, Post-Training, Inference Scaling

**Authors:** Shivam Agarwal, Zimin Zhang, Lifan Yuan, Jiawei Han, Hao Peng

**Institution(s):** University of Illinois Urbana-Champaign


## Problem Background

大型语言模型（LLM）在预训练阶段已具备强大的推理能力，但这些能力在复杂任务（如数学、物理、编码）上未被充分利用；传统方法依赖标注数据进行监督微调或强化学习，成本高且适用性有限。
本文旨在探索是否可以通过熵最小化这一无监督方法，在不依赖标注数据的情况下显著提升 LLM 的推理性能。

## Method

*   **核心思想:** 通过熵最小化（Entropy Minimization, EM）减少模型输出的不确定性，强化其高置信度输出，从而提升推理任务性能，假设模型在高置信度时更可能正确。
*   **具体方法:**
    *   **EM-FT（无监督微调）:** 直接最小化 token 级别的熵，类似于监督微调，但使用模型自身生成的未标注输出，通过减少熵来增强模型对自信输出的偏好。
    *   **EM-RL（强化学习）:** 使用负熵作为唯一的奖励信号进行强化学习，分为序列级熵（trajectory-level，适合探索有限推理路径）和 token 级熵（token-level，适合每步确定性的复杂推理），通过政策梯度优化模型输出分布。
    *   **EM-INF（推理时调整）:** 在推理阶段通过梯度下降优化 logits 以最小化 token 级熵，无需更新模型参数，设置熵阈值防止过度优化，同时使用采样解码选择下一个 token。
*   **关键点:** 三种方法均无需标注数据，EM-FT 和 EM-RL 适用于后训练阶段，EM-INF 适用于推理时动态优化，且计算开销相对较低。

## Experiment

*   **有效性:** EM-FT 在数学和编码任务上平均提升 8%，EM-RL 提升 11%，在 Qwen-7B 上与依赖标注数据的 GRPO 和 RLOO 性能相当（例如 EM-RL-token 在 AMC 上从 31.3% 提升至 57.8%）；EM-INF 在复杂任务（如 SciCode）上使 Qwen-32B 准确率提升 6%（主问题从 4.6% 到 10.7%），超越 GPT-4o。
*   **优越性:** EM-FT 和 EM-RL 在无监督条件下提供了与有监督方法相似的性能提升；EM-INF 比自一致性和迭代精炼效率高 3 倍，且在高不确定性任务上表现更优。
*   **局限性:** 方法在个体价值推理任务（IndVal）上无效，因模型置信度与正确性不相关；在 Llama-3.1-8B 上效果不如 Qwen-2.5，表明依赖预训练模型能力。
*   **实验设置:** 涵盖多种任务（数学、编码、科学推理）和模型（7B 到 32B），对比了有监督与无监督方法的性能和计算成本（FLOPs），设置较为全面，但未探讨跨语言或文化背景的泛化性。

## Further Thoughts

熵最小化作为无监督优化工具的成功，启发我们探索其他信息论指标（如互信息）来进一步优化 LLM 输出；EM-INF 的推理时动态调整策略提示可以在推理阶段引入上下文自适应熵阈值或用户反馈机制；方法对预训练能力的依赖性表明未来 LLM 设计应更关注预训练阶段的推理能力培养，而非仅追求数据规模扩展。