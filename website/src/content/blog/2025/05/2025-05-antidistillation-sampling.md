---
title: "LIMOPro: Reasoning Refinement for Efficient and Effective Test-time Scaling"
pubDatetime: 2025-05-25T15:17:57+00:00
slug: "2025-05-antidistillation-sampling"
type: "arxiv"
id: "2505.19187"
score: 0.716795822203863
author: "grok-3-latest"
authors: ["Yash Savani", "Asher Trockman", "Zhili Feng", "Avi Schwarzschild", "Alexander Robey", "Marc Finzi", "J. Zico Kolter"]
tags: ["LLM", "Proxy Model", "Distillation", "Sampling", "Reasoning"]
institution: ["Carnegie Mellon University", "Google", "Peking University"]
description: "本文提出了一种反蒸馏采样方法，通过代理模型辅助调整推理时token采样分布，有效毒化大型语言模型的推理轨迹以干扰模型蒸馏，同时维持原始模型性能。"
---

> **Summary:** 本文提出了一种反蒸馏采样方法，通过代理模型辅助调整推理时token采样分布，有效毒化大型语言模型的推理轨迹以干扰模型蒸馏，同时维持原始模型性能。 

> **Keywords:** LLM, Proxy Model, Distillation, Sampling, Reasoning

**Authors:** Yash Savani, Asher Trockman, Zhili Feng, Avi Schwarzschild, Alexander Robey, Marc Finzi, J. Zico Kolter

**Institution(s):** Carnegie Mellon University, Google, Peking University


## Problem Background

大型语言模型（LLM）在生成详细推理轨迹（Reasoning Traces）时，虽然展现了强大的能力，但也暴露了一个潜在风险：竞争对手可以通过公开的推理数据进行模型蒸馏（Model Distillation），以低成本复制出性能相近的模型。这不仅威胁到模型的知识产权，还可能导致安全漏洞，例如绕过内置的安全限制或对齐机制。

## Method

*   **核心思想:** 提出一种反蒸馏采样策略，在不改变原始模型（教师模型）参数或性能的前提下，通过调整推理时的采样过程，使生成的推理轨迹对蒸馏过程具有‘毒性’，从而降低学生模型的学习效果。
*   **具体实现:** 
    1. 在生成每个token时，除了依赖教师模型自身的概率分布，还引入一个反蒸馏调整项。
    2. 该调整项通过一个轻量级的代理模型（Proxy Model）计算，代理模型用于模拟学生模型的蒸馏行为，并结合下游任务的损失梯度，估计哪些token选择会显著降低蒸馏效果。
    3. 基于调整后的概率分布进行采样，确保生成的文本既保持教师模型的性能，又对蒸馏过程产生干扰。
*   **关键特性:** 该方法无需对教师模型进行重新训练或微调，仅在推理阶段（Test Time）进行采样调整，同时通过一个可调参数控制‘毒化’强度，避免对教师模型自身输出质量的影响。
*   **技术细节:** 代理模型通常是一个较小的预训练模型，其训练目标是近似学生模型在蒸馏任务上的行为；采样过程可能结合了温度调节和梯度引导，确保平衡性能与抗蒸馏能力。

## Experiment

*   **有效性验证:** 实验在多个标准数据集（如GSM8K和MATH）上进行，结果表明，使用反蒸馏采样生成的推理轨迹在保持教师模型准确率（与基线相比下降不到1%）的同时，显著降低了学生模型的蒸馏效果（准确率下降超过20%）。
*   **对比分析:** 与简单的采样温度提升（会导致教师模型性能显著下降）相比，反蒸馏采样在性能与抗蒸馏能力之间取得了更好的权衡。
*   **实验设置合理性:** 实验覆盖了多种教师模型（如LLaMA、GPT系列）和学生模型（不同规模），并在多个任务（如数学推理、问答）上验证了方法的普适性；同时，实验还探讨了不同毒化强度的影响，设置较为全面。
*   **计算开销:** 主要额外开销来自每次token生成时代理模型的前向推理（两次计算），但由于代理模型规模较小，整体开销可控。

## Further Thoughts

论文中通过代理模型动态调整采样分布的思路非常具有启发性。未来或许可以探索更复杂的代理模型设计，例如使用多模型集成或自适应调整代理模型权重，以应对不同类型的蒸馏攻击。此外，这种‘毒化’推理轨迹的思路是否可以推广到其他生成式模型（如图像生成模型）或多模态模型中，值得进一步研究。另一个有趣的方向是，是否可以通过强化学习（RLHF）进一步优化采样策略，使其在抗蒸馏的同时提升推理质量。