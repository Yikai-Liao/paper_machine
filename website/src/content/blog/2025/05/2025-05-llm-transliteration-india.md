---
title: "Beyond Specialization: Benchmarking LLMs for Transliteration of Indian Languages"
pubDatetime: 2025-05-26T11:35:51+00:00
slug: "2025-05-llm-transliteration-india"
type: "arxiv"
id: "2505.19851"
score: 0.627941971140452
author: "grok-3-latest"
authors: ["Gulfarogh Azam", "Mohd Sadique", "Saif Ali", "Mohammad Nadeem", "Erik Cambria", "Shahab Saquib Sohail", "Mohammad Sultan Alam"]
tags: ["LLM", "Transliteration", "Fine-Tuning", "Multilingual NLP", "Benchmarking"]
institution: ["Aligarh Muslim University", "Nanyang Technological University", "VIT Bhopal University"]
description: "本文通过对大型语言模型（LLMs）在印度语言音译任务上的系统性评估，证明了通用模型（尤其是微调后的 GPT-4o）在大多数情况下可以媲美甚至超越专用模型 IndicXlit，展现了通用模型在领域特定任务中的潜力。"
---

> **Summary:** 本文通过对大型语言模型（LLMs）在印度语言音译任务上的系统性评估，证明了通用模型（尤其是微调后的 GPT-4o）在大多数情况下可以媲美甚至超越专用模型 IndicXlit，展现了通用模型在领域特定任务中的潜力。 

> **Keywords:** LLM, Transliteration, Fine-Tuning, Multilingual NLP, Benchmarking

**Authors:** Gulfarogh Azam, Mohd Sadique, Saif Ali, Mohammad Nadeem, Erik Cambria, Shahab Saquib Sohail, Mohammad Sultan Alam

**Institution(s):** Aligarh Muslim University, Nanyang Technological University, VIT Bhopal University


## Problem Background

印度作为一个语言和文字系统极为多样的国家，音译（Transliteration）在多语言自然语言处理中至关重要，尤其是在信息检索、机器翻译和跨语言知识整合等应用中。
论文的出发点是探索大型语言模型（LLMs）是否能在未经专门训练的情况下，胜任从罗马化文本到印度本地脚本的音译任务，并与专门设计的音译模型（如 IndicXlit）进行性能对比。
关键问题在于：通用型 LLMs 是否具备足够的跨语言知识来处理音译，尤其是在处理未见过词汇或稀有词时？此外，通过微调（fine-tuning）是否能进一步提升其性能？

## Method

*   **模型选择与对比**：论文选择了多个大型语言模型（LLMs），包括 GPT-4o、GPT-4.5、GPT-4.1、Gemma-3-27B-it 和 Mistral-Large，并与专门为印度语言音译设计的 IndicXlit 模型进行对比。IndicXlit 是一个基于 Transformer 的编码器-解码器模型，在 Aksharantar 数据集上训练，支持 21 种语言和 12 种脚本。
*   **数据集准备**：实验使用了两个基准数据集——Dakshina 和 Aksharantar，覆盖 10 种印度语言（包括孟加拉语、印地语、泰米尔语等），测试集分为多个子集，如高频词汇（AK-Freq）、外来命名实体（AK-NEF）和印度本地命名实体（AK-NEI），总计约 12.3 万个词对，确保了语言和任务类型的多样性。
*   **提示设计**：为 LLMs 设计了结构化的输入提示（Prompt），如 GPT 系列模型采用 ChatML 格式，包含系统角色和用户指令，明确要求模型将罗马化文本转换为目标语言的本地脚本；其他模型如 Mistral-Large 和 Gemma-3 使用简化的指令引导提示，确保模型理解音译任务。
*   **微调策略**：对 GPT-4o 进行了微调，使用了来自 Dakshina 和 Aksharantar 数据集的约 98.2 万个训练词对和 9 万个验证词对，覆盖 10 种语言，采用 OpenAI API 服务进行微调，学习率倍数设为 1.5，其余参数保持默认，训练损失和验证损失显示出平稳收敛。
*   **评估指标**：采用 Top-1 Accuracy（预测最可能的输出与实际数据完全匹配的百分比）和 Character Error Rate (CER，基于 Levenshtein 距离的字符级错误率) 作为性能评估标准，确保结果在不同模型间具有可比性。
*   **鲁棒性测试**：通过引入噪声输入（如拼写错误和大小写变化）测试模型在真实场景下的泛化能力，使用了每个语言 100 个常见词的小规模测试集，手动添加合理噪声。

## Experiment

*   **性能对比**：实验结果表明，GPT 系列模型（尤其是 GPT-4.5）在大多数语言和数据集子集上优于专用模型 IndicXlit，例如在泰米尔语（AK-Freq 子集）上，GPT-4.5 的 Top-1 Accuracy 为 78.48%，CER 为 0.041，而 IndicXlit 分别为 69.75% 和 0.052；Gemma-3 和 Mistral-Large 表现稍逊，通常低于 IndicXlit。
*   **微调效果**：微调后的 GPT-4o 在 9 种语言的平均准确率上显著优于 IndicXlit，尤其在命名实体数据集（如 AK-NEI）上表现突出，例如在印地语（AK-NEI）上，GPT-4o-Fine-Tuned 的准确率为 71.10%，CER 为 0.072，而 IndicXlit 仅为 60.94% 和 0.103。
*   **鲁棒性测试**：在噪声输入条件下，微调后的 GPT-4o 表现出最佳泛化能力，Top-1 Accuracy 和 CER 指标均优于其他模型，例如在孟加拉语上，GPT-4o-Fine-Tuned 的准确率为 27%，CER 为 0.282，而 IndicXlit 分别为 18% 和 0.331。
*   **实验设置合理性**：实验覆盖了 10 种语言和多个数据集子集，设置较为全面，评估指标（Top-1 Accuracy 和 CER）能够有效反映音译性能；此外，错误分析揭示了模型在字形、脚本和语义错误上的分布差异，为结果提供了更深层次的解释。
*   **局限性**：实验未涉及句子级音译和反向音译（从本地脚本到罗马化），且仅覆盖 10 种语言，未包括更多印度语言，部分模型（如 GPT-4.5）因技术限制未能进行微调。

## Further Thoughts

论文揭示了通用型 LLMs 在领域特定任务（如音译）上的巨大潜力，尤其通过微调可以显著提升性能，这启发了我思考是否可以在其他多语言任务（如情感分析、文本分类）中采用类似策略，利用通用模型加微调的方式替代专用模型，以降低开发成本；此外，论文对噪声输入的鲁棒性测试让我意识到，未来可以设计更复杂的噪声类型（如语境噪声或多模态输入噪声），进一步验证 LLMs 在真实场景下的适应性；另一个有趣的方向是探索 LLMs 在句子级音译中的表现，是否能捕捉上下文依赖的音译规则。