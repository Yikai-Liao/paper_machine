---
title: "Shadow-FT: Tuning Instruct via Base"
pubDatetime: 2025-05-19T05:16:21+00:00
slug: "2025-05-shadow-ft-tuning"
type: "arxiv"
id: "2505.12716"
score: 0.7952482843245375
author: "grok-3-latest"
authors: ["Taiqiang Wu", "Runming Yang", "Jiayi Li", "Pengfei Hu", "Ngai Wong", "Yujiu Yang"]
tags: ["LLM", "Fine-Tuning", "Weight Similarity", "Instruction Tuning", "Parameter Efficiency"]
institution: ["The University of Hong Kong", "Tsinghua University", "Tencent"]
description: "本文提出 Shadow-FT 框架，通过在 BASE 模型上微调并将权重更新嫁接到 INSTRUCT 模型，显著提升了 INSTRUCT 模型在下游任务上的性能，且不增加额外训练成本。"
---

> **Summary:** 本文提出 Shadow-FT 框架，通过在 BASE 模型上微调并将权重更新嫁接到 INSTRUCT 模型，显著提升了 INSTRUCT 模型在下游任务上的性能，且不增加额外训练成本。 

> **Keywords:** LLM, Fine-Tuning, Weight Similarity, Instruction Tuning, Parameter Efficiency

**Authors:** Taiqiang Wu, Runming Yang, Jiayi Li, Pengfei Hu, Ngai Wong, Yujiu Yang

**Institution(s):** The University of Hong Kong, Tsinghua University, Tencent


## Problem Background

大型语言模型（LLMs）在预训练后通常需要进一步微调以适应特定任务，但直接对指令微调后的模型（INSTRUCT 模型）进行微调往往效果有限，甚至导致性能退化。
研究发现，预训练模型（BASE 模型）与 INSTRUCT 模型的权重高度相似（例如 Llama-3.1-8B 的相对差距 σ 仅为 0.016），这启发了一种新思路：利用 BASE 模型间接提升 INSTRUCT 模型的微调效果，解决直接微调效果不佳的问题。

## Method

*   **核心思想:** 不直接微调 INSTRUCT 模型，而是先对对应的 BASE 模型进行微调，然后将学习到的权重更新直接嫁接到 INSTRUCT 模型上，利用两者权重的高度相似性和结构一致性。
*   **具体步骤:** 
    *   使用常规微调方法（如全参数微调或参数高效微调 LoRA）对 BASE 模型进行训练，获取权重更新 ΔW = W_B+ - W_B，其中 W_B 是原始 BASE 模型权重，W_B+ 是微调后权重。
    *   将 ΔW 直接加到 INSTRUCT 模型的权重上，即 W_I+ = W_I + ΔW，其中 W_I 是原始 INSTRUCT 模型权重，W_I+ 是更新后的权重。
    *   由于 BASE 和 INSTRUCT 模型结构相同，嫁接操作无需额外调整即可完成。
*   **优势:** 该方法不引入额外参数，训练成本与传统微调相当，但避免了直接微调 INSTRUCT 模型可能受到的指令跟随能力干扰，利用 BASE 模型更‘纯粹’的学习能力来提升效果。

## Experiment

*   **有效性:** Shadow-FT 在多个主流 LLM 系列（如 Qwen 3、Llama 3）上显著优于传统微调方法，例如在 Qwen-3-4B 上平均得分从传统微调的 66.2 提升至 69.6，比未微调的 INSTRUCT 模型高 1.6 分。
*   **全面性:** 实验覆盖了 19 个基准数据集（包括数学、编程和推理任务），测试了不同模型规模（1B 到 32B）和微调策略（全参数和 LoRA），结果一致显示 Shadow-FT 的优越性；此外，方法还成功扩展到多模态模型（MLLMs）和直接偏好优化（DPO）。
*   **合理性:** 数据表明传统微调常导致性能下降（如 Qwen-3-4B 在 Math-7 上从 73.8 降至 71.2），而 Shadow-FT 避免下降并实现提升（如达到 75.9），验证了 BASE 模型更适合学习新知识的假设。
*   **成本:** Shadow-FT 不增加额外训练开销，仅需在 BASE 模型上微调一次并嫁接权重更新，计算成本与传统方法相当。

## Further Thoughts

Shadow-FT 揭示了模型权重相似性作为性能提升‘桥梁’的潜力，这种思路可扩展到其他权重相似的模型对，例如利用特定领域预训练模型作为‘影子’提升通用模型性能；此外，是否可以通过构建近似‘影子’模型（例如通过权重剪枝或蒸馏）来解决无 BASE 模型的 INSTRUCT 模型微调问题，值得进一步探索。