---
title: "Transformer Copilot: Learning from The Mistake Log in LLM Fine-tuning"
pubDatetime: 2025-05-22T06:00:45+00:00
slug: "2025-05-antidistillation-sampling"
type: "arxiv"
id: "2505.16270"
score: 0.8139729179944515
author: "grok-3-latest"
authors: ["Yash Savani", "Asher Trockman", "Zhili Feng", "Avi Schwarzschild", "Alexander Robey", "Marc Finzi", "J. Zico Kolter"]
tags: ["LLM", "Proxy Model", "Distillation", "Sampling", "Reasoning"]
institution: ["Carnegie Mellon University", "Google", "Peking University"]
description: "本文提出了一种反蒸馏采样方法，通过代理模型辅助在推理时调整token概率分布，毒化大型语言模型的推理轨迹以干扰模型蒸馏，同时维持原始模型性能。"
---

> **Summary:** 本文提出了一种反蒸馏采样方法，通过代理模型辅助在推理时调整token概率分布，毒化大型语言模型的推理轨迹以干扰模型蒸馏，同时维持原始模型性能。 

> **Keywords:** LLM, Proxy Model, Distillation, Sampling, Reasoning

**Authors:** Yash Savani, Asher Trockman, Zhili Feng, Avi Schwarzschild, Alexander Robey, Marc Finzi, J. Zico Kolter

**Institution(s):** Carnegie Mellon University, Google, Peking University


## Problem Background

大型语言模型（LLMs）在生成详细推理过程（Reasoning Traces）时展现了强大的能力，但这也成为了一个潜在的漏洞。
竞争对手可以通过公开的推理过程，利用模型蒸馏（Model Distillation）技术以较低成本复制出性能相近的模型，从而导致知识产权泄露和安全风险（如绕过模型内置的安全限制）。
本文旨在解决这一关键问题：如何在不牺牲原始模型性能的前提下，保护其推理过程不被轻易用于蒸馏，进而提高模型的安全性和知识产权保护。

## Method

*   **核心思想:** 提出一种反蒸馏采样策略，通过在推理时动态调整token生成概率分布，使生成的推理过程对蒸馏过程具有‘毒性’，从而干扰竞争对手的模型蒸馏效果，同时尽可能维持原始模型（教师模型）的性能。
*   **具体实现步骤:** 
    1. 在生成每个token时，不直接依据教师模型的概率分布采样，而是引入一个反蒸馏调整项。
    2. 该调整项通过一个轻量级的代理模型（Proxy Model）计算，代理模型用于模拟学生模型的蒸馏行为，并结合下游任务的损失梯度，估计哪些token选择会降低蒸馏效果（即对蒸馏有害）。
    3. 将调整项与教师模型的原始概率分布结合，形成一个新的概率分布，并从中采样下一个token。
    4. 通过一个可调参数控制反蒸馏的强度，确保对教师模型自身性能的影响最小化。
*   **技术特点:** 该方法无需修改教师模型的内部参数，仅在推理阶段调整采样策略，具有较低的实现成本。此外，代理模型通常是一个小型模型，计算开销可控。
*   **创新点:** 相比传统的采样策略（如温度调整），反蒸馏采样针对性地优化了抗蒸馏能力，而非简单增加随机性，从而在性能和安全性之间取得了更好的平衡。

## Experiment

*   **有效性验证:** 实验在多个基准数据集（如GSM8K和MATH）上进行，结果表明，使用反蒸馏采样生成的推理文本在保持教师模型准确率基本不变的情况下，显著降低了学生模型通过蒸馏获得的性能（学生模型准确率下降明显）。
*   **对比分析:** 与简单的采样温度调整方法相比，反蒸馏采样在性能-抗蒸馏能力的权衡上表现更优，温度调整往往导致教师模型性能急剧下降，而本文方法有效避免了这一问题。
*   **实验设置合理性:** 实验涵盖了多个任务和数据集，设置了多种蒸馏场景和学生模型类型，较为全面地验证了方法的鲁棒性；同时，作者提供了消融研究，分析了反蒸馏强度参数对结果的影响，增强了结论的可信度。
*   **计算开销:** 主要额外开销来自于每次token生成时代理模型的两次前向计算，但由于代理模型规模较小，整体开销在可接受范围内。

## Further Thoughts

本文的反蒸馏采样策略启发我们可以在推理阶段通过动态调整概率分布来实现多种目标，而不仅仅是抗蒸馏。例如，可以设计类似的策略来增强模型生成的多样性，或者针对特定任务优化输出质量。此外，代理模型的选择和设计也值得进一步探索：是否可以通过自适应选择不同类型的代理模型（不仅仅是小模型，也可能是特定领域预训练模型）来进一步提升反蒸馏效果或适应不同场景？另外，这种方法是否可以推广到其他生成式模型（如图像生成模型）中，以保护其生成内容不被轻易复制？