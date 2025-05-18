---
title: "Multi-Token Prediction Needs Registers"
pubDatetime: 2025-05-15T17:25:03+00:00
slug: "2025-05-mutor-multi-token"
type: "arxiv"
id: "2505.10518"
score: 0.5541906897743336
author: "grok-3-latest"
authors: ["Anastasios Gerontopoulos", "Spyros Gidaris", "Nikos Komodakis"]
tags: ["LLM", "Multi-Token Prediction", "Pre-Training", "Fine-Tuning", "Planning"]
institution: ["Archimedes, Athena Research Center", "valeo.ai", "University of Crete", "IACM-Forth"]
description: "本文提出 `MuToR`，通过插入可学习的寄存器词实现多词预测，以极小参数开销显著提升语言模型和图像生成模型在复杂生成任务中的性能，同时保持与现有预训练模型的兼容性。"
---

> **Summary:** 本文提出 `MuToR`，通过插入可学习的寄存器词实现多词预测，以极小参数开销显著提升语言模型和图像生成模型在复杂生成任务中的性能，同时保持与现有预训练模型的兼容性。 

> **Keywords:** LLM, Multi-Token Prediction, Pre-Training, Fine-Tuning, Planning

**Authors:** Anastasios Gerontopoulos, Spyros Gidaris, Nikos Komodakis

**Institution(s):** Archimedes, Athena Research Center, valeo.ai, University of Crete, IACM-Forth


## Problem Background

当前大型语言模型（LLMs）主要依赖自回归的下一词预测（Next-Token Prediction）训练目标，这种方法在处理需要长距离依赖或规划的任务时表现不佳，容易导致模型学习捷径（Shortcut Learning），忽略长期模式。本文旨在通过改进训练目标，增强模型对未来多步预测的能力，从而提升其在复杂生成任务中的性能，尤其是在微调和预训练场景中。

## Method

* **核心思想**：提出 `MuToR`（Multi-Token Prediction with Registers），通过在输入序列中插入可学习的寄存器词（Register Tokens），让模型在训练时预测未来多个位置的目标词，同时不改变核心架构，确保与现有预训练模型兼容。
* **具体实现**：
  * **寄存器词插入**：在常规词之间插入寄存器词，每个寄存器词分配一个随机采样的偏移量（Offset），用于预测未来特定位置的词。
  * **注意力掩码设计**：通过定制注意力掩码，确保常规词不受寄存器词影响，寄存器词只能关注前面的常规词，从而在推理时可以丢弃寄存器词，保持推理速度不变。
  * **位置编码调整**：寄存器词的位置编码设置为目标预测位置的前一个位置，模拟标准下一词预测模式，适用于现代位置编码方案（如 RoPE）。
  * **训练目标优化**：结合标准下一词预测损失和寄存器词的多词预测损失，通过加权求和进行联合优化，权重参数可调。
  * **领域适配**：在图像生成任务中，扩展为二维偏移量预测，考虑空间依赖性，增强训练信号。
* **关键优势**：相比之前多词预测方法（如添加额外预测头），`MuToR` 引入极少参数，预测范围可扩展，训练成本与预测范围无关，且在推理时无额外开销。

## Experiment

* **有效性**：在语言建模任务（如数学推理 GSM8K, MATH500 和摘要生成 SAMSum）中，`MuToR` 显著优于标准下一词预测和多头预测基线，例如在 GSM8K 上，Gemma 2B 模型准确率从 38.87% 提升至 42.10%。
* **跨领域表现**：在图像生成任务（ImageNet）中，`MuToR` 的二维变体（MuToR-2D）在 FID 和 IS 指标上表现优异，FID 从 7.71 降至 6.57（100K 步），且收敛更快。
* **参数高效性**：结合参数高效微调（PEFT）方法 LoRA，`MuToR` 在资源受限场景下仍保持显著提升，甚至部分超越全参数微调。
* **合理性与全面性**：实验覆盖监督微调、PEFT 和预训练场景，控制训练计算量进行公平对比，消融实验验证了最大预测范围和寄存器数量等超参数的影响，整体设置合理且结果稳健。
* **特殊任务**：在合成数据任务（星图路径查找）中，`MuToR` 克服了下一词预测的捷径学习问题，解决率大幅提升。

## Further Thoughts

通过插入可学习的辅助词（Register Tokens）增强训练信号的思路非常具有启发性，是否可以扩展到其他任务（如多模态学习或复杂推理）中，通过类似辅助结构提升模型能力？此外，寄存器词位置的优化（目前为均匀或随机分布）可以通过自适应策略（如基于任务语义或不确定性）进一步提升效率，减少计算开销。