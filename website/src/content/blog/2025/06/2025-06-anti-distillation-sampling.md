---
title: "Angles Don't Lie: Unlocking Training-Efficient RL Through the Model's Own Signals"
pubDatetime: 2025-06-02T21:40:38+00:00
slug: "2025-06-anti-distillation-sampling"
type: "arxiv"
id: "2506.02281"
score: 0.737260232653202
author: "grok-3-latest"
authors: ["Yash Savani", "Asher Trockman", "Zhili Feng", "Avi Schwarzschild", "Alexander Robey", "Marc Finzi", "J. Zico Kolter"]
tags: ["LLM", "Proxy Model", "Distillation", "Sampling", "Reasoning"]
institution: ["Carnegie Mellon University", "Google", "Peking University"]
description: "本文提出反蒸馏采样方法，通过代理模型辅助在推理时调整 token 分布，成功毒化大型语言模型的推理轨迹以干扰模型蒸馏，同时维持原始性能。"
---

> **Summary:** 本文提出反蒸馏采样方法，通过代理模型辅助在推理时调整 token 分布，成功毒化大型语言模型的推理轨迹以干扰模型蒸馏，同时维持原始性能。 

> **Keywords:** LLM, Proxy Model, Distillation, Sampling, Reasoning

**Authors:** Yash Savani, Asher Trockman, Zhili Feng, Avi Schwarzschild, Alexander Robey, Marc Finzi, J. Zico Kolter

**Institution(s):** Carnegie Mellon University, Google, Peking University


## Problem Background

大型语言模型（LLMs）在生成详细推理过程（Reasoning Traces）时，虽然展现了强大的能力，但也暴露了一个潜在风险：竞争对手可以通过公开的推理数据，利用模型蒸馏（Model Distillation）技术以较低成本复制出性能相近的模型。这不仅可能导致知识产权的泄露，还可能带来安全隐患，例如通过蒸馏绕过模型内置的安全限制或对齐机制。论文的出发点在于如何保护大型语言模型的推理输出，防止其被恶意利用，同时不影响模型自身的性能。

## Method

* **核心思想**：提出一种创新的采样策略，称为‘反蒸馏采样’（Anti-Distillation Sampling），旨在通过在推理过程中动态调整 token 采样分布，‘毒化’生成的推理轨迹，从而干扰基于这些数据的模型蒸馏效果，同时尽可能维持原始模型（教师模型）的性能。
* **具体实现步骤**：
  1. 在生成每个 token 时，不直接依据教师模型的概率分布采样，而是引入一个调整项，形成新的概率分布。
  2. 该调整项通过一个轻量级的代理模型（Proxy Model）计算，代理模型用于模拟学生模型在下游任务上的表现，并通过损失梯度估计哪些 token 选择会显著降低蒸馏效果（即对蒸馏‘有害’）。
  3. 将调整项与教师模型的原始概率分布结合，控制‘毒化’强度，确保生成的文本既干扰蒸馏，又不显著损害教师模型的输出质量。
* **技术细节**：该方法不修改教师模型的内部参数，仅在推理阶段（Test-Time）进行采样调整，因此对原始模型无侵入性。此外，毒化强度的可调参数允许在性能和抗蒸馏能力之间进行权衡。
* **创新点**：相比传统的防御方法（如简单增加采样随机性或提高温度），该方法通过代理模型的辅助实现了更精准的干扰，且对教师模型性能的影响更小。

## Experiment

* **有效性验证**：实验在多个基准数据集（如 GSM8K 和 MATH）上进行，结果表明反蒸馏采样在保持教师模型准确率基本不变的情况下，显著降低了学生模型通过蒸馏获得的性能（学生模型准确率下降幅度可达 20%-30%）。
* **对比分析**：与基线方法（如直接提高采样温度）相比，反蒸馏采样在性能-抗蒸馏能力的权衡上表现更优，温度提高会导致教师模型输出质量急剧下降，而本文方法则实现了更平滑的控制。
* **实验设置**：实验覆盖了多种模型规模和任务类型，设置较为全面，包含了不同强度的毒化参数测试，验证了方法的鲁棒性；但未充分探讨代理模型选择对结果的影响，可能存在优化空间。
* **计算开销**：主要额外开销来自每次 token 生成时代理模型的两次前向计算，但由于代理模型规模较小，整体开销可控。

## Further Thoughts

论文中通过代理模型动态调整采样分布的思路非常具有启发性。未来或许可以探索更复杂的代理模型设计，例如利用多模型集成或跨领域数据训练代理模型，以进一步提升对不同类型蒸馏攻击的防御效果。此外，这种‘毒化’推理轨迹的思路是否可以反向应用，即通过设计特定采样策略增强推理数据的可蒸馏性，帮助小模型学习？这可能为资源受限场景下的模型训练提供新思路。