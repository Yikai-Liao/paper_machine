---
title: "Structured Agent Distillation for Large Language Model"
pubDatetime: 2025-05-20T02:01:55+00:00
slug: "2025-05-anti-distillation-sampling"
type: "arxiv"
id: "2505.13820"
score: 0.8060393923452811
author: "grok-3-latest"
authors: ["Yash Savani", "Asher Trockman", "Zhili Feng", "Avi Schwarzschild", "Alexander Robey", "Marc Finzi", "J. Zico Kolter"]
tags: ["LLM", "Proxy Model", "Distillation", "Sampling", "Reasoning"]
institution: ["Carnegie Mellon University", "Google", "Peking University"]
description: "本文提出了一种反蒸馏采样方法，通过推理时动态调整 token 概率分布，利用代理模型生成对蒸馏有害的推理轨迹，有效保护大型语言模型的知识产权，同时维持其性能。"
---

> **Summary:** 本文提出了一种反蒸馏采样方法，通过推理时动态调整 token 概率分布，利用代理模型生成对蒸馏有害的推理轨迹，有效保护大型语言模型的知识产权，同时维持其性能。 

> **Keywords:** LLM, Proxy Model, Distillation, Sampling, Reasoning

**Authors:** Yash Savani, Asher Trockman, Zhili Feng, Avi Schwarzschild, Alexander Robey, Marc Finzi, J. Zico Kolter

**Institution(s):** Carnegie Mellon University, Google, Peking University


## Problem Background

大型语言模型（LLMs）在生成详细推理过程（Reasoning Traces）时，虽然展现了强大的能力，但也暴露了一个潜在风险：竞争对手可以通过公开的推理数据，利用模型蒸馏（Model Distillation）技术，以较低成本复制出性能相近的模型。这不仅可能导致知识产权的泄露，还可能带来安全隐患，例如通过蒸馏绕过模型内置的安全限制或对齐机制。论文的出发点在于如何保护大型语言模型的推理输出，防止其被轻易蒸馏，同时不影响模型自身的性能。

## Method

* **核心思想**：提出一种反蒸馏采样策略（Anti-Distillation Sampling），通过在推理时动态调整 token 采样概率分布，使生成的推理轨迹对蒸馏过程具有‘毒性’，从而降低学生模型通过蒸馏学习到的性能，而不影响教师模型（即原始大模型）的输出质量。
* **具体实现步骤**：
  1. **概率调整机制**：在生成每个 token 时，不直接依据教师模型的原始概率分布采样，而是引入一个调整项。这个调整项旨在识别并优先选择那些对蒸馏过程有害的 token（即会导致学生模型性能下降的 token）。
  2. **代理模型辅助**：使用一个小型代理模型（Proxy Model）来模拟学生模型的蒸馏过程，通过计算下游任务的损失梯度，估计哪些 token 选择会干扰蒸馏效果。
  3. **采样过程**：将代理模型的梯度信息融入教师模型的概率分布，形成一个新的分布，并从中采样下一个 token。
  4. **强度控制**：通过一个超参数控制反蒸馏调整的强度，确保教师模型的输出质量（如准确率或流畅性）不被显著影响。
* **技术特点**：该方法无需修改教师模型的内部参数，仅在推理阶段调整采样策略，计算开销主要来自代理模型的前向推理。这种设计既灵活又高效，适用于多种大型语言模型。

## Experiment

* **有效性验证**：实验在多个基准数据集（如 GSM8K 和 MATH）上进行，结果表明，使用反蒸馏采样生成的推理文本，在保持教师模型准确率基本不变的情况下，显著降低了学生模型通过蒸馏获得的性能（学生模型准确率下降幅度可达 20%-30%）。
* **对比分析**：与简单的采样温度调整（Temperature Scaling）等基线方法相比，反蒸馏采样在性能与抗蒸馏能力之间取得了更好的平衡。温度调整虽然也能干扰蒸馏，但会导致教师模型输出质量急剧下降，而本文方法避免了这一问题。
* **实验设置合理性**：实验覆盖了多个任务类型和数据集，评估指标包括教师模型的准确率、学生模型的蒸馏后性能以及计算开销，设置较为全面。不过，论文未深入探讨方法在不同规模模型上的适应性，可能存在一定的局限性。
* **计算开销**：主要额外开销来自每次 token 生成时代理模型的两次前向计算，相较于教师模型的整体推理成本，增加的负担较小。

## Further Thoughts

本文的反蒸馏采样策略启发我们可以在推理阶段通过动态调整输出分布来实现多种目标，而不仅仅是抗蒸馏。例如，可以利用类似机制，通过代理模型引导采样过程，增强模型在特定任务上的表现（如提高推理的逻辑性或多样性）。此外，代理模型的选择和设计也值得进一步探索：是否可以利用不同类型的模型（如不同架构或训练数据的模型）作为代理，以适应不同的蒸馏威胁？或者是否可以通过多代理模型协作，进一步提升抗蒸馏效果？这种基于推理时调整的思路，可能为模型安全性和定制化输出开辟新的研究方向。