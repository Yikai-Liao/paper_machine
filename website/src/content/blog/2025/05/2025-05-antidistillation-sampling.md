---
title: "Unlearning Isn't Deletion: Investigating Reversibility of Machine Unlearning in LLMs"
pubDatetime: 2025-05-22T16:02:10+00:00
slug: "2025-05-antidistillation-sampling"
type: "arxiv"
id: "2505.16831"
score: 0.7993831201513302
author: "grok-3-latest"
authors: ["Yash Savani", "Asher Trockman", "Zhili Feng", "Avi Schwarzschild", "Alexander Robey", "Marc Finzi", "J. Zico Kolter"]
tags: ["LLM", "Proxy Model", "Distillation", "Sampling", "Reasoning"]
institution: ["Carnegie Mellon University", "Google", "Peking University"]
description: "本文提出了一种反蒸馏采样方法，通过推理时动态调整 token 概率分布并借助代理模型毒化推理轨迹，有效干扰模型蒸馏过程，同时维持教师模型性能。"
---

> **Summary:** 本文提出了一种反蒸馏采样方法，通过推理时动态调整 token 概率分布并借助代理模型毒化推理轨迹，有效干扰模型蒸馏过程，同时维持教师模型性能。 

> **Keywords:** LLM, Proxy Model, Distillation, Sampling, Reasoning

**Authors:** Yash Savani, Asher Trockman, Zhili Feng, Avi Schwarzschild, Alexander Robey, Marc Finzi, J. Zico Kolter

**Institution(s):** Carnegie Mellon University, Google, Peking University


## Problem Background

大型语言模型（LLMs）在生成详细推理轨迹（Reasoning Traces）时，虽然展现了强大的能力，但也暴露了一个潜在漏洞：竞争对手可以通过公开的推理数据，利用模型蒸馏（Model Distillation）技术廉价地复制出性能相近的模型，这不仅涉及知识产权泄露，还可能导致安全风险，例如绕过模型内置的安全限制或对齐机制。本文旨在解决这一关键问题，探索如何在不牺牲教师模型性能的前提下，保护其推理轨迹免受蒸馏威胁。

## Method

* **核心思想**：提出一种反蒸馏采样策略，通过在推理阶段动态调整教师模型的 token 概率分布，使生成的推理轨迹对蒸馏过程‘带毒’，从而降低学生模型的学习效果，同时尽量维持教师模型的原始性能。
* **具体实现步骤**：
  1. **概率分布调整**：在生成每个 token 时，不直接依据教师模型的原始概率分布进行采样，而是引入一个反蒸馏调整项。
  2. **代理模型辅助**：利用一个较小的代理模型（Proxy Model）来模拟学生模型的蒸馏行为，通过计算下游任务的损失梯度，估计哪些 token 的选择会对蒸馏效果产生负面影响（即降低学生模型性能）。
  3. **动态采样**：将调整项与教师模型的原始概率分布结合，形成新的分布，并从中采样下一个 token。
  4. **强度控制**：通过一个超参数控制反蒸馏调整项的权重，确保教师模型的性能不会因过度毒化而显著下降。
* **技术亮点**：该方法无需修改教师模型的内部参数，仅在推理阶段进行干预，降低了实现成本；同时，代理模型的使用使得反蒸馏策略更具针对性，而非随机干扰。

## Experiment

* **有效性验证**：实验在多个基准数据集（如 GSM8K 和 MATH）上进行，结果表明，使用反蒸馏采样生成的推理轨迹在保持教师模型准确率基本不变的情况下，显著降低了学生模型的蒸馏效果（学生模型准确率下降幅度可达 20%-30%）。
* **对比分析**：与简单的采样温度调整（Temperature Scaling）等基线方法相比，反蒸馏采样在性能与抗蒸馏能力之间取得了更好的平衡，温度调整往往导致教师模型性能急剧下降。
* **实验设置合理性**：实验覆盖了多种模型规模和任务类型，考虑了不同毒化强度的影响，设置较为全面；但未充分探讨代理模型规模对效果的影响，可能是一个局限。
* **计算开销**：主要额外开销来自每次 token 生成时代理模型的前向计算（两次），相较于教师模型的整体推理成本，增加的负担可控。

## Further Thoughts

本文的反蒸馏采样策略启发我们可以在推理阶段引入更多动态干预机制，而不仅仅依赖训练阶段的优化；此外，代理模型的设计和选择可能是一个值得深入探索的方向，例如是否可以通过结合不同模型的推理特性（如某些模型的推理轨迹在特定任务上更具泛化性）来进一步提升反蒸馏效果，或者将这种思路扩展到其他模型保护场景，如对抗样本生成或隐私保护。