---
title: "LLM-Explorer: A Plug-in Reinforcement Learning Policy Exploration Enhancement Driven by Large Language Models"
pubDatetime: 2025-05-21T09:24:23+00:00
slug: "2025-05-anti-distillation-sampling"
type: "arxiv"
id: "2505.15293"
score: 0.6419092462871595
author: "grok-3-latest"
authors: ["Yash Savani", "Asher Trockman", "Zhili Feng", "Avi Schwarzschild", "Alexander Robey", "Marc Finzi", "J. Zico Kolter"]
tags: ["LLM", "Proxy Model", "Distillation", "Sampling", "Reasoning"]
institution: ["Carnegie Mellon University", "Google", "Peking University"]
description: "本文提出了一种反蒸馏采样方法，通过代理模型辅助在推理时调整 token 采样分布，有效毒化大型语言模型的推理轨迹以干扰模型蒸馏，同时维持原始模型性能。"
---

> **Summary:** 本文提出了一种反蒸馏采样方法，通过代理模型辅助在推理时调整 token 采样分布，有效毒化大型语言模型的推理轨迹以干扰模型蒸馏，同时维持原始模型性能。 

> **Keywords:** LLM, Proxy Model, Distillation, Sampling, Reasoning

**Authors:** Yash Savani, Asher Trockman, Zhili Feng, Avi Schwarzschild, Alexander Robey, Marc Finzi, J. Zico Kolter

**Institution(s):** Carnegie Mellon University, Google, Peking University


## Problem Background

大型语言模型（LLMs）在生成详细推理轨迹（Reasoning Traces）时，虽然展现了强大的能力，但也暴露了一个潜在风险：竞争对手可以通过模型蒸馏（Model Distillation）技术，利用这些公开的推理数据廉价地训练出性能相近的学生模型。这不仅可能导致知识产权泄露，还可能带来安全隐患，例如绕过模型内置的安全限制或对齐机制。因此，亟需一种方法在不牺牲教师模型性能的前提下，保护其输出免受蒸馏威胁。

## Method

* **核心理念**：提出一种反蒸馏采样（Anti-Distillation Sampling）策略，通过在推理时动态调整 token 采样概率分布，使生成的推理轨迹对蒸馏过程具有‘毒性’，从而干扰学生模型的学习效果，而不改变教师模型的内部参数。
* **具体实现**：
  1. 在生成每个 token 时，结合教师模型的原始概率分布和一个额外的‘反蒸馏调整项’。
  2. 该调整项通过一个轻量级的代理模型（Proxy Model）计算，代理模型模拟学生模型的蒸馏过程，并结合下游任务的损失梯度，估计哪些 token 选择会显著降低蒸馏效果（即对学生模型有害）。
  3. 最终从调整后的概率分布中采样下一个 token，确保生成的文本既保持语义连贯性，又对蒸馏过程产生干扰。
* **关键特性**：
  - 不需要对教师模型进行重新训练或微调，仅在推理阶段进行采样调整，降低了实现成本。
  - 通过控制调整项的强度，可以灵活平衡教师模型的性能与抗蒸馏能力，避免对自身输出质量造成显著影响。
  - 代理模型通常是一个小型模型，其计算开销相对可控，适合实际部署。

## Experiment

* **效果验证**：实验在多个基准数据集（如 GSM8K 和 MATH，用于评估数学推理能力）上进行，结果表明反蒸馏采样能够在维持教师模型准确率基本不变的情况下，显著降低学生模型通过蒸馏获得的性能（学生模型准确率下降幅度可达 20%-30%）。
* **对比分析**：与简单的采样温度调整（Temperature Scaling）等基线方法相比，反蒸馏采样在性能与抗蒸馏能力之间取得了更好的权衡；温度调整虽然也能干扰蒸馏，但往往导致教师模型输出质量急剧下降，而本文方法有效避免了这一问题。
* **实验设置**：实验覆盖了多种任务类型和模型规模，设置较为全面；同时测试了不同强度调整项的影响，确保方法的鲁棒性。
* **计算开销**：主要额外开销来自每次 token 生成时代理模型的前向计算（两次前向传播），但由于代理模型规模较小，整体开销在可接受范围内。

## Further Thoughts

本文的反蒸馏采样方法启发了我思考如何进一步优化代理模型的设计：是否可以通过自适应选择代理模型（例如根据目标学生模型的特性动态调整）来提升毒化效果？此外，反蒸馏的概念是否可以推广到其他生成式模型（如图像生成模型）的输出保护中，通过类似的方式干扰扩散模型的蒸馏过程？另一个有趣的方向是探索推理轨迹的‘毒化’是否可以结合强化学习（RLHF）机制，进一步增强模型输出的安全性。