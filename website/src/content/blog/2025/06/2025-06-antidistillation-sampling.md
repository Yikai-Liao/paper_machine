---
title: "Scalable In-Context Q-Learning"
pubDatetime: 2025-06-02T04:21:56+00:00
slug: "2025-06-antidistillation-sampling"
type: "arxiv"
id: "2506.01299"
score: 0.6757728776395603
author: "grok-3-latest"
authors: ["Yash Savani", "Asher Trockman", "Zhili Feng", "Avi Schwarzschild", "Alexander Robey", "Marc Finzi", "J. Zico Kolter"]
tags: ["LLM", "Proxy Model", "Distillation", "Sampling", "Reasoning"]
institution: ["Carnegie Mellon University", "Google", "Peking University"]
description: "本文提出了一种反蒸馏采样策略，通过代理模型辅助在推理时调整token概率分布，有效毒化大型语言模型的推理轨迹以干扰模型蒸馏，同时维持原始模型性能。"
---

> **Summary:** 本文提出了一种反蒸馏采样策略，通过代理模型辅助在推理时调整token概率分布，有效毒化大型语言模型的推理轨迹以干扰模型蒸馏，同时维持原始模型性能。 

> **Keywords:** LLM, Proxy Model, Distillation, Sampling, Reasoning

**Authors:** Yash Savani, Asher Trockman, Zhili Feng, Avi Schwarzschild, Alexander Robey, Marc Finzi, J. Zico Kolter

**Institution(s):** Carnegie Mellon University, Google, Peking University


## Problem Background

大型语言模型（LLM）在生成详细推理轨迹（Reasoning Traces）时，虽然展现了强大的能力，但这些公开的推理数据也成为了一个潜在的漏洞。
竞争对手可以通过模型蒸馏（Distillation）技术，利用这些推理轨迹低成本地复制出性能相近的模型，从而导致知识产权泄露，甚至可能绕过安全限制，带来潜在的安全风险。
因此，如何在不影响模型性能的前提下，保护推理轨迹免受蒸馏攻击，成为一个亟待解决的关键问题。

## Method

*   **核心思想:** 提出一种反蒸馏采样策略，通过在推理时动态调整概率分布，使生成的推理轨迹对蒸馏过程具有‘毒性’，从而干扰学生模型的学习效果，同时尽量维持教师模型的原始性能。
*   **具体实现步骤:** 
    1. 在生成每个token时，不直接依据教师模型的原始概率分布进行采样，而是引入一个反蒸馏调整项。
    2. 该调整项通过一个轻量级的代理模型（Proxy Model）来辅助计算，具体是基于下游任务的损失梯度，估计哪些token的选择会对蒸馏过程造成更大的干扰（即降低学生模型的性能）。
    3. 将调整项与教师模型的原始概率分布结合，形成一个新的分布，并从中采样下一个token。
    4. 通过一个可调参数控制‘毒化’的强度，确保教师模型自身的输出质量不会显著下降。
*   **技术亮点:** 该方法无需修改教师模型的内部参数，仅在推理阶段通过采样策略实现反蒸馏效果，具有较低的实施成本和较高的灵活性。
*   **创新点:** 相比传统的防御方法（如简单增加采样温度导致性能下降），该方法通过代理模型的辅助实现了更精细的概率调整，达到了性能与防御效果的更好平衡。

## Experiment

*   **有效性验证:** 实验结果表明，在多个基准数据集（如GSM8K和MATH）上，使用反蒸馏采样策略生成的推理轨迹能够显著降低学生模型通过蒸馏获得的准确率（例如下降超过20%），而教师模型自身的任务性能仅受到轻微影响（准确率下降小于2%）。
*   **对比分析:** 与其他baseline方法（如直接提高采样温度）相比，反蒸馏采样在性能-防御权衡上表现更优，教师模型的输出质量下降幅度更小，同时对蒸馏的干扰效果更强。
*   **实验设置合理性:** 实验覆盖了多种模型规模（从小型到大型语言模型）和多个任务类型（数学推理、逻辑推理等），设置较为全面，但未涉及跨语言任务或多模态任务的测试，可能存在一定的局限性。
*   **计算开销:** 该方法的额外开销主要来自每次token生成时对代理模型的两次前向计算，但由于代理模型规模较小，整体开销可控。

## Further Thoughts

论文中通过代理模型动态调整采样分布的思路非常具有启发性，未来可以探索是否能利用不同模型的推理轨迹特性来进一步优化反蒸馏策略。例如，某些模型的推理数据可能在特定任务或特定模型家族中表现出更强的蒸馏效果，是否可以针对性地设计不同的代理模型或调整策略？此外，是否可以将反蒸馏思想扩展到其他生成任务（如图像生成或多模态生成）中，以保护更广泛的生成模型输出？