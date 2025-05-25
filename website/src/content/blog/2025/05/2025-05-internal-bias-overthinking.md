---
title: "Internal Bias in Reasoning Models leads to Overthinking"
pubDatetime: 2025-05-22T09:35:52+00:00
slug: "2025-05-internal-bias-overthinking"
type: "arxiv"
id: "2505.16448"
score: 0.7424996906152026
author: "grok-3-latest"
authors: ["Renfei Dang", "Shujian Huang", "Jiajun Chen"]
tags: ["LLM", "Reasoning", "Internal Bias", "Attention Mechanism", "Overthinking"]
institution: ["National Key Laboratory for Novel Software Technology, Nanjing University"]
description: "本文揭示了推理模型中内部偏见（Internal Bias）是导致过度思考的关键原因，并通过 MASK 方法有效减少推理长度 31%-53% 并提升复杂任务准确率。"
---

> **Summary:** 本文揭示了推理模型中内部偏见（Internal Bias）是导致过度思考的关键原因，并通过 MASK 方法有效减少推理长度 31%-53% 并提升复杂任务准确率。 

> **Keywords:** LLM, Reasoning, Internal Bias, Attention Mechanism, Overthinking

**Authors:** Renfei Dang, Shujian Huang, Jiajun Chen

**Institution(s):** National Key Laboratory for Novel Software Technology, Nanjing University


## Problem Background

当前长推理模型（如 o1/R1 类型）在复杂任务中表现出强大的自发反思和纠错能力，但常因冗余推理导致过度思考（Overthinking），浪费计算资源。
作者提出，模型在接触问题时会立即形成初步猜测（内部偏见，Internal Bias），当这一猜测与推理结果冲突时，模型倾向于进行额外反思，从而引发过度思考行为。
研究这一问题的意义在于揭示模型行为背后的机制，为优化推理效率和减少资源浪费提供理论依据。

## Method

*   **核心思想：** 识别并量化推理模型中的内部偏见（Internal Bias），分析其对过度思考的影响，并通过干预手段减少其负面效应。
*   **具体步骤：**
    *   **内部偏见测量：** 设计‘直接答案’（Direct Answer）模板，强制模型在不推理的情况下立即输出答案，捕捉其初步猜测；通过多重采样（16 次），结合不同提示模板和解码温度，近似内部偏见的概率分布。
    *   **偏差程度量化：** 对于数值型任务，采用平均绝对误差（MAE）衡量直接答案与最终推理结果的差距；对于分类任务，使用不一致率（Inconsistency Rate）评估偏差程度。
    *   **注意力机制分析：** 通过可视化和统计分析注意力分数，发现模型在决定是否进一步反思时，过度关注输入问题部分，从而引入内部偏见。
    *   **MASK 方法干预：** 在模型首次得出答案后，通过修改注意力掩码（Attention Mask）屏蔽输入问题部分，强制模型仅依赖自身推理步骤进行后续决策，减少内部偏见的影响。
*   **创新点：** 方法结合了行为分析、注意力机制解释和干预实验，从现象到机制再到解决方案，形成了完整的研究逻辑。

## Experiment

*   **有效性：** 实验在多个模型（DeepSeek-R1, QwQ-32B, R1-Distill-Qwen-14B）、数据集（CharCount, KnowLogic, AIME 2024/2025）和语言（英文和中文）上展开，结果显示内部偏见普遍存在，直接答案准确率远低于推理结果；高偏差组推理长度比低偏差组长 17.2% 至 42.1%，表明内部偏见显著导致过度思考。
*   **MASK 方法效果：** MASK 方法将推理长度减少了 31% 至 53%（如 AIME 2024 减少 53.5%）；在复杂任务上准确率提升 2.1% 至 10%（如 AIME 2025 从 36.7% 提升至 46.7%），但在简单任务上准确率略降，可能是 MASK 对短推理链的干扰。
*   **合理性与全面性：** 实验设计考虑了问题复杂性（通过首轮推理长度评估），排除了复杂性而非内部偏见导致长推理的可能性；数据一致性高，标准偏差虽大，但 MASK 后稳定性提升；实验覆盖了不同模型家族、规模和任务类型，设置较为全面。

## Further Thoughts

内部偏见作为模型行为的驱动因素，这一概念可扩展至决策模型或对话系统，探索初步猜测对后续行为的影响；注意力机制的过度关注问题提示可以通过动态调整注意力分布优化效率，例如在推理后期降低对输入的关注权重；MASK 方法启发是否可在训练阶段引入注意力约束机制，而非仅依赖解码时干预；此外，内部偏见的形成可能与预训练数据分布有关，未来可研究通过数据清洗或训练目标调整减少这种偏见。