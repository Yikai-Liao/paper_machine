---
title: "Let LLMs Break Free from Overthinking via Self-Braking Tuning"
pubDatetime: 2025-05-20T16:53:40+00:00
slug: "2025-05-self-braking-tuning"
type: "arxiv"
id: "2505.14604"
score: 0.7322826442743937
author: "grok-3-latest"
authors: ["Haoran Zhao", "Yuchen Yan", "Yongliang Shen", "Haolei Xu", "Wenqi Zhang", "Kaitao Song", "Jian Shao", "Weiming Lu", "Jun Xiao", "Yueting Zhuang"]
tags: ["LLM", "Reasoning", "Efficiency", "Self-Regulation", "Supervised Fine-Tuning"]
institution: ["Zhejiang University", "Tianjin University", "Microsoft Research Asia"]
description: "本文提出自刹车调优（SBT）框架，通过数据构建和自然语言提示，使大型推理模型自主终止冗余推理，在数学任务上减少30%-60% token消耗，同时保持准确率。"
---

> **Summary:** 本文提出自刹车调优（SBT）框架，通过数据构建和自然语言提示，使大型推理模型自主终止冗余推理，在数学任务上减少30%-60% token消耗，同时保持准确率。 

> **Keywords:** LLM, Reasoning, Efficiency, Self-Regulation, Supervised Fine-Tuning

**Authors:** Haoran Zhao, Yuchen Yan, Yongliang Shen, Haolei Xu, Wenqi Zhang, Kaitao Song, Jian Shao, Weiming Lu, Jun Xiao, Yueting Zhuang

**Institution(s):** Zhejiang University, Tianjin University, Microsoft Research Asia


## Problem Background

大型推理模型（LRMs）通过生成详细的多步推理轨迹在数学和逻辑任务中表现出色，但往往伴随过度思考（Overthinking），导致推理过程冗长、计算成本高、延迟增加，并可能掩盖核心解决方案，限制了实际部署中的应用效率。
现有方法多依赖外部干预（如强化学习、监督微调或推理时动态调整），而论文提出一个核心问题：能否让模型自主识别并终止冗余推理，从而内生地提高效率？

## Method

*   **核心思想：** 提出自刹车调优（Self-Braking Tuning, SBT）框架，通过数据构建和训练策略，使模型内生地学习在推理过程中适时终止冗余思考，避免外部控制机制的复杂性。
*   **过度思考识别：** 设计两个互补指标量化推理轨迹中的冗余部分：
    *   **推理效率比（Reasoning Efficiency Ratio）：** 衡量首次正确答案前的推理步骤占比，值越低表示冗余越多。
    *   **过度思考标记比（Overthinking Marker Ratio）：** 基于特定语言标记（如‘Wait’, ‘Check’）的出现频率，评估语言层面的冗余倾向。
    *   两者结合为加权‘过度思考分数（Overthink Score）’，用于定位推理终止点。
*   **数据构建策略：** 提出两种方法构建自适应推理长度的数据集：
    *   **SBT-Exact（精确截断）：** 统一保留基础解（Foundation Solution）和一个演化解（Evolution Solution），并掩码后续少量冗余内容，确保结构一致性。
    *   **SBT-Dynamic（动态截断）：** 逐步分析推理轨迹，根据过度思考分数和预设阈值动态决定终止点，适应不同问题复杂性。
*   **自调节刹车机制：** 引入两种训练辅助手段：
    *   **冗余思考掩码：** 保留部分冗余推理内容但不参与损失计算，让模型暴露于冗余模式而不强化它们。
    *   **自然语言刹车提示：** 在终止点插入自反性语句（如‘我已经多次得到相同答案，是时候结束了’），增强模型对推理状态的元认知感知。
*   **实现细节：** 基于高质量数据集OpenR1-Math构建训练数据，通过监督微调（Supervised Fine-Tuning）训练模型，推理时无需额外干预即可自主终止。

## Experiment

*   **有效性：** 在多个数学基准数据集（GSM8K, MATH500, AIME, AMC23）上，SBT显著减少了token消耗（30%-60%），同时保持了接近基线的准确率。例如，Qwen2.5-Math-7B-Instruct上，SBT-E减少30.7% token，准确率仅下降2.65%；Llama-3.1-8B上，SBT-E减少62.8% token，保留94.1%准确率。
*   **模型差异：** 通用模型（如Llama）在大规模时效率提升更明显（1B模型减少54.2%，8B减少62.8%）；数学专用模型（如Qwen）在小规模时提升更显著（1.5B减少48.9%，7B减少30.7%），反映预训练目标对SBT效果的影响。
*   **策略对比：** SBT-E在token减少上更激进（平均48.3%），但准确率略降；SBT-D更平衡，尤其在困难任务（如MATH500）上表现优异，Llama-3.1-8B准确率提升2.62%同时减少58.7% token。
*   **实验设置合理性：** 实验覆盖不同难度任务和模型架构，评估指标包括准确率和token消耗，推理时采用vLLM生成多样本取平均，硬件环境一致（NVIDIA A100 GPU）。但数据集规模（92K样本）较小，可能限制泛化性，且未充分探讨极端复杂任务中的过早终止风险。

## Further Thoughts

SBT框架通过内生自调节机制启发我们，是否可以通过更复杂的元认知提示或多阶段训练，进一步增强模型对推理深度的感知能力？
自然语言刹车提示优于特殊token的效果表明，语义上下文理解能力可被更好利用，未来是否可以设计个性化刹车提示，基于任务类型或用户需求动态调整终止行为？
此外，SBT目前聚焦数学推理，若扩展到开放性或多模态任务，过度思考的定义和识别需重新设计，是否可以通过跨领域迁移学习解决这一问题？