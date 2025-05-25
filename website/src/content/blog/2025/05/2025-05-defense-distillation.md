---
title: "O$^2$-Searcher: A Searching-based Agent Model for Open-Domain Open-Ended Question Answering"
pubDatetime: 2025-05-22T12:17:13+00:00
slug: "2025-05-defense-distillation"
type: "arxiv"
id: "2505.16582"
score: 0.6999807422679857
author: "grok-3-latest"
authors: ["Yash Savani", "Asher Trockman", "Zhili Feng", "Avi Schwarzschild", "Alexander Robey", "Marc Finzi", "J. Zico Kolter"]
tags: ["LLM", "Proxy Model", "Distillation", "Sampling", "Reasoning"]
institution: ["Carnegie Mellon University", "Google", "Peking University"]
description: "本文提出了一种反蒸馏采样方法，通过推理时动态调整token概率分布以毒化推理轨迹，有效干扰模型蒸馏过程，同时维持大型语言模型的原始性能。"
---

> **Summary:** 本文提出了一种反蒸馏采样方法，通过推理时动态调整token概率分布以毒化推理轨迹，有效干扰模型蒸馏过程，同时维持大型语言模型的原始性能。 

> **Keywords:** LLM, Proxy Model, Distillation, Sampling, Reasoning

**Authors:** Yash Savani, Asher Trockman, Zhili Feng, Avi Schwarzschild, Alexander Robey, Marc Finzi, J. Zico Kolter

**Institution(s):** Carnegie Mellon University, Google, Peking University


## Problem Background

大型语言模型（LLM）在生成详细推理轨迹（Reasoning Traces）时，虽然展现了强大的性能，但也暴露了一个潜在漏洞：竞争对手可以通过公开的推理数据，利用模型蒸馏（Model Distillation）技术以较低成本复制出类似性能的模型，这不仅涉及知识产权泄露，还可能导致安全风险（如绕过模型内置的安全限制）。本文旨在解决这一关键问题，探索如何在不牺牲模型性能的前提下，保护模型免受未经授权的蒸馏。

## Method

* **核心思想**：提出一种创新的采样策略，称为‘反蒸馏采样’，通过在推理时动态调整token生成概率分布，‘毒化’推理轨迹，从而干扰下游的模型蒸馏过程，同时尽量维持原始模型（教师模型）的性能。
* **具体实现**：
  1. 在生成每个token时，不直接依赖教师模型的原始概率分布，而是引入一个调整项，用于评估哪些token选择会对蒸馏过程造成‘有害’影响（即降低学生模型的性能）。
  2. 该调整项通过一个轻量级的代理模型（Proxy Model）计算，结合下游任务的损失梯度，估计token选择的潜在影响。
  3. 最终基于调整后的概率分布进行采样，确保生成的文本既保留教师模型的推理能力，又对蒸馏过程构成干扰。
* **关键特点**：
  - 不需要修改教师模型的内部参数，仅在推理阶段调整采样策略，降低了实施成本。
  - 通过控制‘毒化’强度（如调整项的权重），在性能与防御效果之间取得平衡。
  - 代理模型通常是一个小型模型，计算开销相对可控。

## Experiment

* **有效性验证**：实验结果表明，在多个基准数据集（如GSM8K和MATH）上，使用反蒸馏采样生成的推理文本能够在保持教师模型准确率基本不变的情况下，显著降低学生模型通过蒸馏获得的性能（准确率下降明显）。
* **对比分析**：与简单的采样温度调整（Temperature Scaling）相比，反蒸馏采样在性能-防御权衡上表现更优，温度调整往往导致教师模型性能急剧下降，而本文方法能更好地维持原始性能。
* **实验设置合理性**：实验覆盖了多种任务和数据集，充分验证了方法的泛化性；同时，实验还探讨了不同毒化强度的影响，提供了全面的参数分析。
* **计算开销**：主要额外开销来自每次token生成时代理模型的前向计算，但由于代理模型规模较小，整体开销在可接受范围内。

## Further Thoughts

本文的反蒸馏采样策略启发我们思考：是否可以将类似的‘毒化’机制扩展到其他生成模型（如图像生成模型）中，以保护其生成内容不被轻易复制？此外，是否可以通过多模型协作或动态代理模型选择，进一步提升防御效果，例如针对不同类型的蒸馏攻击自适应调整毒化策略？另一个有趣的方向是探索推理轨迹的‘毒化’是否会对模型的可解释性研究产生副作用，这可能是未来研究的一个重要课题。