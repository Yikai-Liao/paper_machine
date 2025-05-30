---
title: "Train with Perturbation, Infer after Merging: A Two-Stage Framework for Continual Learning"
pubDatetime: 2025-05-28T14:14:19+00:00
slug: "2025-05-anti-distillation-sampling"
type: "arxiv"
id: "2505.22389"
score: 0.8664344495883617
author: "grok-3-latest"
authors: ["Yash Savani", "Asher Trockman", "Zhili Feng", "Avi Schwarzschild", "Alexander Robey", "Marc Finzi", "J. Zico Kolter"]
tags: ["LLM", "Proxy Model", "Distillation", "Sampling", "Reasoning"]
institution: ["Carnegie Mellon University", "Google", "Peking University"]
description: "本文提出反蒸馏采样方法，通过代理模型辅助在推理时动态调整 token 采样分布，有效毒化大型语言模型的推理轨迹以干扰模型蒸馏，同时维持原始模型性能。"
---

> **Summary:** 本文提出反蒸馏采样方法，通过代理模型辅助在推理时动态调整 token 采样分布，有效毒化大型语言模型的推理轨迹以干扰模型蒸馏，同时维持原始模型性能。 

> **Keywords:** LLM, Proxy Model, Distillation, Sampling, Reasoning

**Authors:** Yash Savani, Asher Trockman, Zhili Feng, Avi Schwarzschild, Alexander Robey, Marc Finzi, J. Zico Kolter

**Institution(s):** Carnegie Mellon University, Google, Peking University


## Problem Background

大型语言模型（LLMs）在生成详细推理过程（Reasoning Traces）时，虽然展现了强大的能力，但也暴露了一个潜在风险：竞争对手可以通过公开的推理数据，利用模型蒸馏（Model Distillation）技术以较低成本复制出性能相近的模型。这不仅可能导致知识产权的泄露，还可能带来安全隐患，例如通过蒸馏绕过模型内置的安全限制或对齐机制。论文的出发点在于如何保护大型语言模型的推理输出，防止其被轻易复制，同时确保原始模型的性能不受影响。

## Method

* **核心思想**：提出一种创新的采样策略，称为反蒸馏采样（Anti-Distillation Sampling），旨在通过在推理过程中动态调整 token 采样分布，使生成的推理轨迹对蒸馏过程‘有毒’，从而降低学生模型通过蒸馏学习到的性能，而不影响教师模型自身的输出质量。
* **具体实现步骤**：
  1. 在生成每个 token 时，基于教师模型的原始概率分布，引入一个额外的调整项，用于干扰蒸馏效果。
  2. 该调整项通过一个轻量级的代理模型（Proxy Model）计算，代理模型模拟学生模型的蒸馏学习过程，并结合下游任务的损失梯度，估计哪些 token 选择会显著降低蒸馏效果。
  3. 将调整项与原始概率分布结合，形成新的采样分布，从中选择下一个 token。
* **关键特点**：
  - 不需要修改教师模型的内部参数，仅在推理阶段通过采样策略实现干扰，保持了模型的原始性能。
  - 引入了毒化强度的可调参数，允许在抗蒸馏效果和输出质量之间进行权衡。
  - 代理模型通常是一个较小的模型，计算开销相对可控，但其选择和训练对最终效果至关重要。

## Experiment

* **有效性验证**：实验在多个基准数据集（如 GSM8K 和 MATH）上进行，结果表明反蒸馏采样能够在维持教师模型准确率基本不变的情况下，显著降低学生模型通过蒸馏获得的性能（学生模型准确率下降幅度可达 20%-30%，具体取决于毒化强度）。
* **对比分析**：与简单的采样温度调整（temperature scaling）等基线方法相比，反蒸馏采样在性能-抗蒸馏能力的权衡上表现更优，温度调整往往导致教师模型输出质量急剧下降，而本文方法能更好地保持输出一致性。
* **实验设置合理性**：实验覆盖了不同规模的教师模型和学生模型，测试了多种下游任务，确保了方法的通用性；同时，作者还探讨了不同毒化强度对结果的影响，提供了详细的消融研究。
* **计算开销**：主要额外开销来自每次 token 生成时代理模型的前向计算（通常需要两次前向传播），相较于教师模型的整体推理成本，增加的计算负担在可接受范围内。

## Further Thoughts

论文中通过代理模型动态调整采样分布的思路非常具有启发性。未来或许可以探索更复杂的代理模型设计，例如结合多模型集成或跨模型推理数据的特性差异，进一步提升抗蒸馏效果。此外，这种‘毒化’输出的思想是否可以扩展到其他领域，如对抗性攻击或数据隐私保护，通过在输出中嵌入难以察觉的噪声，干扰未经授权的使用？另外，是否可以通过强化学习（RLHF）进一步优化采样策略，使其在不同任务和模型间自适应调整毒化强度？