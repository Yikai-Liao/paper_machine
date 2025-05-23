---
title: "Polar Sparsity: High Throughput Batched LLM Inferencing with Scalable Contextual Sparsity"
pubDatetime: 2025-05-20T20:15:42+00:00
slug: "2025-05-antidistillation-sampling"
type: "arxiv"
id: "2505.14884"
score: 0.7514157847915494
author: "grok-3-latest"
authors: ["Yash Savani", "Asher Trockman", "Zhili Feng", "Avi Schwarzschild", "Alexander Robey", "Marc Finzi", "J. Zico Kolter"]
tags: ["LLM", "Proxy Model", "Distillation", "Sampling", "Reasoning"]
institution: ["Carnegie Mellon University", "Google", "Peking University"]
description: "本文提出了一种反蒸馏采样方法，通过代理模型辅助在推理时调整 token 采样分布，有效毒化大型语言模型的推理轨迹以干扰模型蒸馏，同时维持原始模型性能。"
---

> **Summary:** 本文提出了一种反蒸馏采样方法，通过代理模型辅助在推理时调整 token 采样分布，有效毒化大型语言模型的推理轨迹以干扰模型蒸馏，同时维持原始模型性能。 

> **Keywords:** LLM, Proxy Model, Distillation, Sampling, Reasoning

**Authors:** Yash Savani, Asher Trockman, Zhili Feng, Avi Schwarzschild, Alexander Robey, Marc Finzi, J. Zico Kolter

**Institution(s):** Carnegie Mellon University, Google, Peking University


## Problem Background

大型语言模型（LLMs）在生成详细推理过程（Reasoning Traces）时，虽然展现了强大的能力，但也暴露了一个潜在的漏洞。
竞争对手可以通过公开的推理数据，利用模型蒸馏（Model Distillation）技术，以较低成本复制出性能相近的模型，这不仅涉及知识产权泄露，还可能导致安全风险，例如绕过模型内置的安全限制或对齐机制。
本文旨在解决这一关键问题：如何在不影响原始模型性能的前提下，防止其推理过程被用于有效的模型蒸馏。

## Method

*   **核心思想:** 提出一种反蒸馏采样策略，通过在推理时动态调整 token 采样分布，使生成的推理过程对蒸馏过程具有‘毒性’，从而降低学生模型的学习效果，同时尽量维持教师模型的性能。
*   **具体实现:** 
    *   在生成每个 token 时，不直接依据教师模型的概率分布采样，而是引入一个反蒸馏调整项。
    *   该调整项通过一个轻量级的代理模型（Proxy Model）计算，代理模型用于模拟学生模型的蒸馏过程，并结合下游任务的损失梯度，估计哪些 token 选择会显著降低蒸馏效果。
    *   调整后的概率分布会倾向于选择那些对蒸馏有害的 token，同时通过一个可调参数控制毒化强度，避免对教师模型自身性能造成过大影响。
*   **技术细节:** 
    *   反蒸馏采样不修改教师模型的内部参数，仅在推理阶段对采样过程进行干预，因此对原始模型无侵入性。
    *   代理模型通常是一个较小的模型，其计算开销相对可控，但需要针对目标任务进行一定的预训练或微调。
*   **创新点:** 这种方法区别于传统的防御机制（如简单增加采样随机性），通过有针对性的概率调整，实现了性能与抗蒸馏能力的更好平衡。

## Experiment

*   **有效性验证:** 实验在多个基准数据集（如 GSM8K 和 MATH）上进行，结果表明，使用反蒸馏采样生成的推理文本，在保持教师模型准确率基本不变的情况下，显著降低了学生模型通过蒸馏获得的性能（学生模型准确率下降幅度可达 20%-30%）。
*   **对比分析:** 与基线方法（如直接提高采样温度）相比，反蒸馏采样在性能-抗蒸馏能力的权衡上表现更优，基线方法往往导致教师模型性能急剧下降。
*   **实验设置:** 实验覆盖了不同规模的教师模型和学生模型，数据集选择具有代表性，包含数学推理等复杂任务，评价指标包括准确率和蒸馏效果（学生模型性能）。
*   **计算开销:** 主要额外开销来自每次 token 生成时对代理模型的两次前向计算，但由于代理模型较小，整体开销在可接受范围内。
*   **局限性:** 实验未充分探讨反蒸馏采样对不同类型下游任务的普适性，可能在某些任务上效果有所差异。

## Further Thoughts

反蒸馏采样的核心在于利用代理模型动态调整概率分布，这启发我们是否可以进一步探索多代理模型协作机制，例如针对不同类型的学生模型或任务设计不同的代理模型，以实现更精准的毒化效果。
此外，是否可以将这种动态调整的思想应用于其他领域，如数据生成或对抗性防御，进一步提升模型的安全性？
另一个值得思考的方向是，推理数据的特性对蒸馏效果的影响是否可以被量化，从而设计更通用的防御策略。