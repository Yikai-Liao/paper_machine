---
title: "LEGO-Compiler: Enhancing Neural Compilation Through Translation Composability"
pubDatetime: 2025-05-26T07:07:54+00:00
slug: "2025-05-antidistillation-sampling"
type: "arxiv"
id: "2505.20356"
score: 0.6836388511541023
author: "grok-3-latest"
authors: ["Yash Savani", "Asher Trockman", "Zhili Feng", "Avi Schwarzschild", "Alexander Robey", "Marc Finzi", "J. Zico Kolter"]
tags: ["LLM", "Proxy Model", "Distillation", "Sampling", "Reasoning"]
institution: ["Carnegie Mellon University", "Google", "Peking University"]
description: "本文提出了一种反蒸馏采样方法，通过代理模型辅助动态调整Token采样分布，毒化大型语言模型的推理轨迹以干扰模型蒸馏，同时维持原始性能，有效提升了模型知识产权保护能力。"
---

> **Summary:** 本文提出了一种反蒸馏采样方法，通过代理模型辅助动态调整Token采样分布，毒化大型语言模型的推理轨迹以干扰模型蒸馏，同时维持原始性能，有效提升了模型知识产权保护能力。 

> **Keywords:** LLM, Proxy Model, Distillation, Sampling, Reasoning

**Authors:** Yash Savani, Asher Trockman, Zhili Feng, Avi Schwarzschild, Alexander Robey, Marc Finzi, J. Zico Kolter

**Institution(s):** Carnegie Mellon University, Google, Peking University


## Problem Background

大型语言模型（LLM）在生成详细推理轨迹（Reasoning Traces）时，虽然展现了强大的能力，但也暴露了一个潜在漏洞：竞争对手可以通过公开的推理数据，利用模型蒸馏（Distillation）技术低成本地复制出性能相近的模型，这不仅涉及知识产权泄露，还可能导致安全风险，例如绕过模型内置的安全限制。

## Method

* **核心思想**：提出一种创新策略，在不影响原始模型（教师模型）性能的前提下，通过‘毒化’生成的推理轨迹，干扰下游的模型蒸馏过程，使其难以成功复制模型能力。
* **具体实现**：
  * 在推理阶段，调整Token采样策略：除了基于教师模型自身的概率分布，还引入一个‘反蒸馏’调整项。
  * 该调整项通过一个轻量级的代理模型（Proxy Model）计算，结合下游任务的损失梯度，评估哪些Token选择会对蒸馏过程产生负面影响（即降低学生模型性能）。
  * 最终基于调整后的概率分布进行采样，生成带有‘毒性’的推理轨迹。
* **技术细节**：
  * 代理模型是一个小型模型，用于近似估计蒸馏效果，计算成本较低。
  * 通过控制调整项的强度，确保教师模型的输出质量不受显著影响。
  * 该方法无需修改教师模型的权重，仅在推理时动态调整采样过程，具有较高的实用性。
* **创新点**：相比传统的采样调整方法（如提高采样温度），这种方法在性能和抗蒸馏能力之间取得了更好的平衡。

## Experiment

* **有效性验证**：实验在多个基准数据集（如GSM8K和MATH）上进行，结果表明，使用反蒸馏采样生成的推理轨迹，在保持教师模型准确率基本不变的情况下，显著降低了学生模型通过蒸馏获得的性能（准确率下降明显）。
* **对比分析**：与简单的采样温度调整方法相比，反蒸馏采样在性能-抗蒸馏能力的权衡上表现更优，温度调整往往导致教师模型输出质量急剧下降。
* **实验设置**：实验覆盖了不同规模的教师模型和学生模型，设置较为全面，但未充分探讨方法在非语言任务（如图像生成模型）上的适用性。
* **计算开销**：主要额外开销来自每次Token生成时代理模型的前向计算，但由于代理模型较小，整体成本可控。

## Further Thoughts

论文提出的‘毒化’推理轨迹以干扰蒸馏的思想具有广泛的启发性：是否可以将类似机制应用于其他领域，例如通过动态调整数据分布来保护用户隐私，或在对抗性攻击中引入‘伪信息’以误导攻击者？此外，不同模型的推理数据在蒸馏中的表现差异（如某些模型的轨迹更难被蒸馏）或许可以作为设计更高效保护策略的切入点。