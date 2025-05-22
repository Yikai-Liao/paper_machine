---
title: "FlashThink: An Early Exit Method For Efficient Reasoning"
pubDatetime: 2025-05-20T05:28:21+00:00
slug: "2025-05-flashthink-early-exit"
type: "arxiv"
id: "2505.13949"
score: 0.8474416211678248
author: "grok-3-latest"
authors: ["Guochao Jiang", "Guofeng Quan", "Zepeng Ding", "Ziqin Luo", "Dixuan Wang", "Zheng Hu"]
tags: ["LLM", "Early Exit", "Reasoning Efficiency", "Verification Model", "Inference Optimization"]
institution: ["Fudan University"]
description: "本文提出 FlashThink 方法，通过验证模型动态判断推理过程的提前退出时机，显著提升大型语言模型的推理效率，同时保持准确率。"
---

> **Summary:** 本文提出 FlashThink 方法，通过验证模型动态判断推理过程的提前退出时机，显著提升大型语言模型的推理效率，同时保持准确率。 

> **Keywords:** LLM, Early Exit, Reasoning Efficiency, Verification Model, Inference Optimization

**Authors:** Guochao Jiang, Guofeng Quan, Zepeng Ding, Ziqin Luo, Dixuan Wang, Zheng Hu

**Institution(s):** Fudan University


## Problem Background

大型语言模型（LLMs）在推理任务中表现出色，但经常对简单问题生成冗长的推理内容，导致计算资源浪费和推理时间延长。
论文指出，模型在推理过程早期往往已具备得出正确答案的能力，因此可以通过提前退出推理阶段来提升效率，解决计算开销过大的关键问题。

## Method

*   **核心思想:** 在不修改原始推理模型参数或自回归生成范式的前提下，通过一个外部验证模型动态判断推理过程是否可以提前退出，从而减少不必要的推理内容生成。
*   **具体实现:** 
    *   使用预定义的分隔符（Delimiter Tokens）将推理内容分割成多个小块（Chunks）。
    *   在生成每个小块后，调用验证模型（Verification Model）评估当前推理内容是否足以得出正确答案。
    *   若验证模型判断可以退出，则直接生成最终答案；否则继续生成下一个推理小块。
*   **优化策略（FT[2]）:** 对验证模型进行微调，使其适应特定推理模型和输入数据分布，进一步提升准确率和效率。
*   **优势:** 该方法避免了昂贵的模型重新训练成本，仅通过推理阶段的动态调整实现效率提升，具有较高的实用性。

## Experiment

*   **有效性:** FlashThink 方法在保持甚至略微提升模型准确率的同时，显著缩短了推理内容长度。例如，DeepSeek-R1 的平均准确率从 87.00 提升至 87.15，QwQ-32B 从 83.56 提升至 83.87；效率方面，DeepSeek-R1 和 QwQ-32B 的推理内容长度平均减少了 77.04% 和 77.47%。
*   **数据集全面性:** 实验覆盖了四个难度各异的基准数据集（GSM8K, MATH, GPQA Diamond, DROP），涵盖数学推理和知识推理任务，设置合理。
*   **模型多样性:** 测试了多个推理模型（DeepSeek-R1, QwQ-32B 等）及不同验证模型，验证了方法的普适性，其中 Qwen2.5-7B-Instruct 被证明为较优验证模型。
*   **优化效果:** 微调后的 FT[2] 方法进一步提升了性能，例如 QwQ-32B 在 GPQA Diamond 数据集上的准确率从 58.08 提升至 62.73，效率从 65.32% 提升至 75.64%。
*   **结论:** 实验数据表明，FlashThink 在效率提升方面效果显著，对准确率的负面影响极小，实验设计全面合理。

## Further Thoughts

FlashThink 的‘动态提前退出’概念启发我们可以在多步计算任务中引入中间状态评估机制，不仅限于语言模型推理，还可能应用于图像生成或决策任务；此外，验证模型的模块化设计和微调策略（FT[2]）提示我们可以通过辅助模型优化主模型性能，未来或许可以探索非生成式模型（如 MLP 或 Encoder-only Transformers）作为验证模型，以进一步降低计算成本。