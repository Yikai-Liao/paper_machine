---
title: "Stable Forgetting: Bounded Parameter-Efficient Unlearning in LLMs"
pubDatetime: 2025-09-29T01:30:15+00:00
slug: "2025-09-bounded-unlearning-llm"
type: "arxiv"
id: "2509.24166"
score: 0.7249807147352817
author: "grok-3-latest"
authors: ["Arpit Garg", "Hemanth Saratchandran", "Ravi Garg", "Simon Lucey"]
tags: ["LLM", "Machine Unlearning", "Parameter Efficiency", "Gradient Stability", "Fine-Tuning"]
institution: ["Australian Institute for Machine Learning (AIML), The University of Adelaide"]
description: "本文提出有界参数高效遗忘方法，通过对MLP前馈层LoRA适配器应用有界函数，稳定梯度差异优化过程，显著提升大型语言模型遗忘效果并保持保留性能。"
---

> **Summary:** 本文提出有界参数高效遗忘方法，通过对MLP前馈层LoRA适配器应用有界函数，稳定梯度差异优化过程，显著提升大型语言模型遗忘效果并保持保留性能。 

> **Keywords:** LLM, Machine Unlearning, Parameter Efficiency, Gradient Stability, Fine-Tuning

**Authors:** Arpit Garg, Hemanth Saratchandran, Ravi Garg, Simon Lucey

**Institution(s):** Australian Institute for Machine Learning (AIML), The University of Adelaide


## Problem Background

大型语言模型（LLMs）在预训练过程中吸收了大量数据，其中可能包含敏感、版权或个人身份信息，亟需通过机器遗忘（Machine Unlearning）移除特定数据影响以满足隐私和安全需求。
然而，现有遗忘方法（如梯度差异方法）在结合交叉熵损失时，因梯度上升导致权重和梯度无界增长，造成训练不稳定，进而影响遗忘效果和对保留数据的性能。

## Method

*   **核心思想:** 提出‘Bounded Parameter-Efficient Unlearning’方法，通过对MLP前馈层的LoRA适配器应用有界函数，限制权重和梯度动态，稳定遗忘过程中的训练。
*   **理论依据:** 作者通过数学分析证明，梯度上升（用于遗忘数据）会导致MLP前馈层权重和梯度的无界增长，引发优化不稳定；引入有界函数（如tanh或sine）可有效约束这种增长。
*   **具体实现:** 基于LoRA参数高效微调框架，仅对MLP前馈层的低秩适配器矩阵（AB^T）应用有界非线性变换（如sin(ωAB^T)），而不修改注意力层或预训练权重；通过选择合适的频率参数ω（如100），增强适配器的表达能力，同时保持参数效率。
*   **优势与细节:** 该方法避免了预训练知识的丢失，仅需少量额外计算（主要是sine函数评估），并通过有界性确保梯度上升过程中的稳定性；实验中主要采用sine函数，因其高频特性可提升有效秩，提供更好的遗忘-保留平衡。

## Experiment

*   **遗忘效果显著:** 在TOFU基准上，使用Phi-1.5B模型（rank-4 LoRA），方法取得遗忘质量（Forget Quality, FQ）9.43e-01，比最强基线LoKU（1.28e-04）高出三个数量级；在TDEC基准上，提取损失（EL10）降至0.3，优于所有基线，显示出强大的隐私保护能力。
*   **保留性能稳定:** 模型实用性（Model Utility, MU）在TOFU上保持为0.52，与原始模型相当；在TDEC上推理准确性达52.1，优于基线，表明遗忘未损害保留数据表现。
*   **全面性与合理性:** 实验覆盖GPT-Neo、Phi、LLaMA等多种架构，参数规模从125M到8B，LoRA rank从4到32，遗忘比例从1%到10%，验证了方法的鲁棒性；图表数据（如Fig. 2）显示标准LoRA梯度爆炸至10^5，而本方法保持在10^1-10^2，证实了理论预测。
*   **局限性与扩展性:** 在超大规模模型（如LLaMA-3.1-70B）上遗忘质量略降（0.42-0.48），可能因参数冗余增加，提示需进一步优化；计算开销增加（如sine评估），但对遗忘效果的提升具有合理性。

## Further Thoughts

有界参数化思路不仅适用于遗忘任务，还可能推广至对抗训练或持续学习中，通过约束权重动态缓解灾难性遗忘问题；此外，sine函数频率参数（ω）的动态调整或自适应设计可能进一步优化遗忘与保留的平衡，值得探索；最后，分析注意力层的梯度行为并设计针对性稳定机制，或可提升整体遗忘框架的性能。