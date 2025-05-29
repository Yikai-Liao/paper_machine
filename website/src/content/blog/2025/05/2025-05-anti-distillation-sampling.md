---
title: "Scaling over Scaling: Exploring Test-Time Scaling Pareto in Large Reasoning Models"
pubDatetime: 2025-05-26T20:58:45+00:00
slug: "2025-05-anti-distillation-sampling"
type: "arxiv"
id: "2505.20522"
score: 0.8154984753477666
author: "grok-3-latest"
authors: ["Yash Savani", "Asher Trockman", "Zhili Feng", "Avi Schwarzschild", "Alexander Robey", "Marc Finzi", "J. Zico Kolter"]
tags: ["LLM", "Proxy Model", "Distillation", "Sampling", "Reasoning"]
institution: ["Carnegie Mellon University", "Google", "Peking University"]
description: "本文提出了一种反蒸馏采样方法，通过代理模型辅助动态调整Token概率分布，成功毒化大型语言模型的推理轨迹以干扰模型蒸馏，同时维持原始模型性能。"
---

> **Summary:** 本文提出了一种反蒸馏采样方法，通过代理模型辅助动态调整Token概率分布，成功毒化大型语言模型的推理轨迹以干扰模型蒸馏，同时维持原始模型性能。 

> **Keywords:** LLM, Proxy Model, Distillation, Sampling, Reasoning

**Authors:** Yash Savani, Asher Trockman, Zhili Feng, Avi Schwarzschild, Alexander Robey, Marc Finzi, J. Zico Kolter

**Institution(s):** Carnegie Mellon University, Google, Peking University


## Problem Background

大型语言模型（LLM）在生成详细推理轨迹（Reasoning Traces）时，虽然展现了强大的能力，但也暴露了一个潜在漏洞：竞争对手可以通过模型蒸馏（Model Distillation）技术，利用这些公开数据廉价复制出性能相近的模型，从而导致知识产权泄露和安全风险（如绕过安全限制或生成有害内容）。
论文的目标是开发一种方法，保护教师模型的输出数据，使其难以被用于蒸馏，同时确保教师模型自身的性能不受显著影响。

## Method

* **核心思想：** 提出一种反蒸馏采样策略（Anti-Distillation Sampling），在推理时动态调整Token生成概率分布，使生成的推理轨迹对蒸馏过程‘带毒’，从而降低学生模型的学习效果，而不影响教师模型的准确性。
* **具体实现：**
  1. **概率调整机制：** 在生成每个Token时，除了依赖教师模型自身的概率分布外，引入一个调整项，用于偏向选择那些对蒸馏有害的Token。
  2. **代理模型辅助：** 使用一个小型代理模型（Proxy Model）模拟学生模型的蒸馏过程，通过计算下游任务的损失梯度，估计哪些Token选择会降低蒸馏效果。
  3. **采样过程：** 将调整项与教师模型的原始概率分布结合，形成新的分布，并从中采样下一个Token。
  4. **强度控制：** 通过一个超参数控制‘毒化’的强度，确保教师模型的输出质量不会显著下降。
* **技术优势：** 该方法无需修改教师模型的内部参数，仅在推理阶段进行干预，降低了实现成本；同时，代理模型的使用使得调整过程更加精准和可控。

## Experiment

* **有效性验证：** 实验在多个基准数据集（如GSM8K、MATH）上进行，结果表明，使用反蒸馏采样生成的推理轨迹，在保持教师模型准确率基本不变的情况下，显著降低了学生模型的蒸馏效果（准确率下降幅度达20%-30%）。
* **对比分析：** 与其他方法（如直接提高采样温度）相比，反蒸馏采样在性能与防护效果之间取得了更好的平衡；提高采样温度会导致教师模型输出质量急剧下降，而本文方法的影响微乎其微。
* **实验设置：** 实验设计较为全面，涵盖了不同规模的教师模型和学生模型，以及多种下游任务；此外，还进行了消融研究，验证了代理模型和调整项的重要性。
* **计算开销：** 主要额外开销来自每次Token生成时代理模型的前向计算，但由于代理模型规模较小，整体延迟仍在可接受范围内。

## Further Thoughts

本文的代理模型设计启发了我思考是否可以引入多层次代理模型，针对不同类型的蒸馏攻击（如基于不同数据集或不同模型架构）进行定制化防御；此外，这种动态调整概率分布的思路是否可以应用于其他生成任务（如图像生成模型的输出保护），以防止类似的知识产权泄露？另一个有趣的方向是探索攻击者可能采取的反制措施，例如通过数据清洗或更复杂的蒸馏技术绕过‘毒化’输出，是否需要进一步设计对抗性防御机制？