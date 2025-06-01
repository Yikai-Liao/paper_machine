---
title: "EVOREFUSE: Evolutionary Prompt Optimization for Evaluation and Mitigation of LLM Over-Refusal to Pseudo-Malicious Instructions"
pubDatetime: 2025-05-29T14:26:46+00:00
slug: "2025-05-antidistillation-sampling"
type: "arxiv"
id: "2505.23473"
score: 0.6131207540803825
author: "grok-3-latest"
authors: ["Yash Savani", "Asher Trockman", "Zhili Feng", "Avi Schwarzschild", "Alexander Robey", "Marc Finzi", "J. Zico Kolter"]
tags: ["LLM", "Proxy Model", "Distillation", "Sampling", "Reasoning"]
institution: ["Carnegie Mellon University", "Google", "Peking University"]
description: "本文提出了一种反蒸馏采样方法，通过代理模型辅助调整token生成概率分布，成功毒化大型语言模型的推理轨迹以干扰模型蒸馏，同时维持原始模型性能。"
---

> **Summary:** 本文提出了一种反蒸馏采样方法，通过代理模型辅助调整token生成概率分布，成功毒化大型语言模型的推理轨迹以干扰模型蒸馏，同时维持原始模型性能。 

> **Keywords:** LLM, Proxy Model, Distillation, Sampling, Reasoning

**Authors:** Yash Savani, Asher Trockman, Zhili Feng, Avi Schwarzschild, Alexander Robey, Marc Finzi, J. Zico Kolter

**Institution(s):** Carnegie Mellon University, Google, Peking University


## Problem Background

大型语言模型（LLMs）在生成详细推理过程（Reasoning Traces）时，虽然展现了强大的能力，但也暴露了一个潜在风险：竞争对手可以通过公开的推理数据，利用模型蒸馏（Model Distillation）技术廉价地复制出性能相近的模型。这不仅可能导致知识产权泄露，还可能带来安全隐患，例如绕过模型内置的安全限制或对齐机制。

## Method

* **核心思想**：提出一种创新策略，在不改变原始模型（教师模型）参数或显著影响其性能的前提下，通过调整推理时的采样过程，使生成的推理轨迹对蒸馏过程具有‘毒性’，从而降低学生模型通过蒸馏学习到的性能。
* **具体实现**：
  * 在生成每个token时，不直接依据教师模型的概率分布采样，而是引入一个‘反蒸馏’调整项，改变概率分布。
  * 该调整项通过一个轻量级的代理模型（Proxy Model）计算，代理模型用于模拟学生模型的蒸馏过程，并结合下游任务的损失梯度，估计哪些token的选择会对蒸馏效果造成负面影响。
  * 最终基于调整后的概率分布进行采样，确保生成的文本既保持教师模型的输出质量，又对蒸馏过程产生干扰。
* **关键特性**：
  * 不需要对教师模型进行重新训练或微调，仅在推理阶段进行干预，降低了计算成本。
  * 通过控制调整项的强度，可以在性能维持和抗蒸馏效果之间取得平衡。
* **技术细节**：代理模型通常是一个较小的预训练模型，其训练目标是近似学生模型的学习行为；调整项的计算可能涉及梯度估计或启发式规则。

## Experiment

* **有效性验证**：实验在多个基准数据集（如GSM8K和MATH，用于评估数学推理能力）上进行，结果表明，使用反蒸馏采样生成的推理轨迹在保持教师模型准确率基本不变的情况下，显著降低了学生模型通过蒸馏获得的性能（准确率下降幅度可达20%-30%）。
* **对比分析**：与简单的采样策略（如提高采样温度）相比，反蒸馏采样在性能-抗蒸馏能力的权衡上表现更优；温度调整往往导致教师模型输出质量急剧下降，而本文方法更为精细。
* **实验设置合理性**：实验覆盖了不同规模的教师模型和学生模型，测试了多种下游任务，确保了方法的普适性；同时，实验还探讨了代理模型规模对效果的影响，验证了方法的鲁棒性。
* **计算开销**：主要额外开销来自每次token生成时代理模型的前向计算，但由于代理模型较小，整体开销可控。

## Further Thoughts

本文提出的代理模型辅助采样策略启发了我思考：是否可以进一步探索不同类型的辅助模型（如基于规则的模型或不同架构的神经网络）来优化概率调整过程？此外，推理数据的特性对蒸馏效果的影响也值得深入研究，例如某些模型的推理轨迹可能天然具有更高的泛化性，能否利用这一特性设计更高效的抗蒸馏机制？最后，这种推理时干预的思路是否可以扩展到其他领域，如隐私保护或模型效率提升？