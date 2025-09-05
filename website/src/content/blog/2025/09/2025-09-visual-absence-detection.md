---
title: "Unveiling the Response of Large Vision-Language Models to Visually Absent Tokens"
pubDatetime: 2025-09-03T05:17:25+00:00
slug: "2025-09-visual-absence-detection"
type: "arxiv"
id: "2509.03025"
score: 0.7757511402706194
author: "grok-3-latest"
authors: ["Sohee Kim", "Soohyun Ryu", "Joonhyung Park", "Eunho Yang"]
tags: ["LVLM", "Hallucination", "Cross-Modal Alignment", "Neuron Activation", "Visual Grounding"]
institution: ["KAIST AI", "AITRICS"]
description: "本文揭示了大型视觉-语言模型中存在视觉缺失感知神经元，并基于此开发轻量级检测器和改进策略，有效减少了因视觉缺失词导致的幻觉问题。"
---

> **Summary:** 本文揭示了大型视觉-语言模型中存在视觉缺失感知神经元，并基于此开发轻量级检测器和改进策略，有效减少了因视觉缺失词导致的幻觉问题。 

> **Keywords:** LVLM, Hallucination, Cross-Modal Alignment, Neuron Activation, Visual Grounding

**Authors:** Sohee Kim, Soohyun Ryu, Joonhyung Park, Eunho Yang

**Institution(s):** KAIST AI, AITRICS


## Problem Background

大型视觉-语言模型（LVLMs）在处理视觉与文本输入时，常常错误地将缺乏视觉证据的文本输入（即视觉缺失词）视为图像中存在的内容，导致生成不准确的响应。
这种问题源于模型在跨模态对齐学习中的偏差，作者旨在解决如何让模型识别输入文本是否在图像中有视觉依据，并改进输出以避免幻觉问题。

## Method

*   **核心思想:** 利用模型内部前馈网络（FFN）的激活模式，识别输入文本是否在图像中有视觉依据，并基于此改进模型输出。
*   **具体实现:** 
    *   **发现视觉缺失感知神经元（VA Neurons）:** 通过对比视觉存在和缺失词的 FFN 激活模式，作者发现一组特定神经元在处理视觉缺失词时表现出显著不同的激活值，称之为视觉缺失感知神经元。
    *   **评分系统筛选神经元:** 设计一个基于 Bhattacharyya 系数的评分系统（S_VA），量化每个 FFN 神经元对视觉缺失的敏感度，筛选出高敏感度的 VA 神经元。
    *   **构建视觉缺失检测器（VA Detector）:** 利用 VA 神经元的激活值作为特征，训练一个轻量级线性分类器，判断输入或生成的词是否在图像中有视觉依据。训练数据来自作者构建的 VA-QA 数据集，包含视觉存在和缺失的对比样本。
    *   **输出改进策略:** 在二元问答任务中，若检测到视觉缺失词，则将答案调整为“No”；在开放式生成任务中，若生成的词被检测为视觉缺失，则回退到前一步并选择概率次高的词，确保输出更贴合图像内容。
*   **关键特点:** 该方法不依赖外部知识或模型重训练，仅利用内部激活信号，计算开销低，且适用于多种 LVLMs，具有较强的通用性。

## Experiment

*   **有效性:** 在二元问答任务中，方法显著提高了对包含视觉缺失词的问题的正确拒绝率（即“No”准确率），例如在 VA-QA 数据集上，LLaVA-v1.5 的“No”准确率从 48.0% 提升至 77.5%，整体准确率从 71.6% 提升至 83.5%。在开放式生成任务中，幻觉内容比例降低（CHAIR 指标下降），如 mPLUG-Owl2 的句级幻觉指标从 66.8 降至 57.2。
*   **全面性与合理性:** 实验覆盖多个数据集（VA-QA、POPE、R-Bench、SEED-Bench、CHAIR 等），包括领域内和领域外测试，验证了方法的泛化能力。测试了不同模型规模（7B 到 32B）和解码策略（如 beam search），结果一致表明改进效果。数据集设计合理，如 VA-QA 通过对比图像对确保视觉缺失词的明确性。
*   **权衡与开销:** 方法在提高“No”准确率的同时，部分情况下“Yes”准确率略有下降，反映模型更保守，但整体准确率提升。计算开销低，VA 检测器仅需额外计算激活值和分类，对性能影响小。

## Further Thoughts

论文通过模型内部神经元激活模式检测跨模态对齐问题的思路非常启发性，是否可以进一步挖掘其他类型的激活模式，用于解决 LVLMs 在属性推理或时间理解等任务中的幻觉问题？此外，若将 VA 检测器的训练数据扩展到包含情感或抽象概念的多样化数据集，是否能提升其检测能力？这种自省式方法是否也可应用于其他多模态模型（如语音-文本模型）以检测跨模态不一致性？