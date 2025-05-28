---
title: "Beyond Specialization: Benchmarking LLMs for Transliteration of Indian Languages"
pubDatetime: 2025-05-26T11:35:51+00:00
slug: "2025-05-llm-transliteration-indian"
type: "arxiv"
id: "2505.19851"
score: 0.627941971140452
author: "grok-3-latest"
authors: ["Gulfarogh Azam", "Mohd Sadique", "Saif Ali", "Mohammad Nadeem", "Erik Cambria", "Shahab Saquib Sohail", "Mohammad Sultan Alam"]
tags: ["LLM", "Transliteration", "Fine-Tuning", "Multilingual NLP", "Benchmarking"]
institution: ["Aligarh Muslim University", "Nanyang Technological University", "VIT Bhopal University"]
description: "本文通过系统性评估证明了通用大型语言模型（LLMs）在印度语言音译任务上的潜力，尤其在微调后可超越专门模型，同时揭示了其鲁棒性和错误模式特性。"
---

> **Summary:** 本文通过系统性评估证明了通用大型语言模型（LLMs）在印度语言音译任务上的潜力，尤其在微调后可超越专门模型，同时揭示了其鲁棒性和错误模式特性。 

> **Keywords:** LLM, Transliteration, Fine-Tuning, Multilingual NLP, Benchmarking

**Authors:** Gulfarogh Azam, Mohd Sadique, Saif Ali, Mohammad Nadeem, Erik Cambria, Shahab Saquib Sohail, Mohammad Sultan Alam

**Institution(s):** Aligarh Muslim University, Nanyang Technological University, VIT Bhopal University


## Problem Background

印度作为一个拥有22种官方语言和多种书写系统的多语言国家，音译（将文本从一种脚本转换为另一种脚本）在自然语言处理中至关重要。
传统上，音译依赖专门模型（如IndicXlit），但大型语言模型（LLMs）的快速发展引发了研究兴趣：通用LLMs是否能在未经特定任务训练的情况下胜任音译任务，甚至通过微调进一步提升性能？
关键问题在于通用模型与专门模型在处理印度语言多样性和复杂性（如稀有词汇、跨脚本映射）时的表现对比，以及LLMs的局限性和潜在优势。

## Method

*   **核心思想:** 系统性评估通用大型语言模型（LLMs）与专门音译模型IndicXlit在印度语言音译任务上的性能，探索通用模型是否能替代专门模型。
*   **数据集与任务设置:** 使用两个基准数据集Dakshina和Aksharantar，覆盖10种印度语言（包括印地语、泰米尔语、乌尔都语等），测试集分为高频词汇（AK-Freq）、外国命名实体（AK-NEF）和印度命名实体（AK-NEI）等子集，确保任务多样性。
*   **模型选择与对比:** 选择了多个通用LLMs（GPT-4o、GPT-4.5、GPT-4.1、Gemma-3-27B-it、Mistral-Large）与专门模型IndicXlit进行对比，评估其在音译任务上的表现。
*   **提示设计:** 为LLMs设计特定提示（如ChatML格式），以引导模型理解音译任务并生成准确输出，例如将罗马字母输入转换为本土脚本。
*   **微调实验:** 对GPT-4o进行微调，使用近百万对音译词对数据（来自Dakshina和Aksharantar），以观察微调是否能显著提升性能。
*   **评估指标:** 使用Top-1 Accuracy（预测最可能结果与实际结果匹配的百分比）和Character Error Rate (CER，字符级错误率) 评估模型性能。
*   **错误分析与鲁棒性测试:** 分析模型输出的错误类型（图形错误、脚本错误、语义错误），并通过引入噪声输入（如拼写错误、大小写变化）测试模型在现实场景中的鲁棒性。
*   **关键点:** 方法不仅关注性能指标，还深入探讨模型的局限性和适应性，提供了多维度的评估视角。

## Experiment

*   **有效性:** 实验结果表明，GPT家族模型（尤其是GPT-4.5）在大多数语言和数据集类别上优于专门模型IndicXlit，例如GPT-4.5在泰米尔语AK-Freq子集上的Top-1 Accuracy为78.48%，而IndicXlit为69.75%。
*   **微调提升:** 微调后的GPT-4o在几乎所有语言上表现更优，尤其在命名实体数据集上，例如在印地语AK-NEI数据集上准确率为71.10%，远超IndicXlit的60.94%。
*   **鲁棒性测试:** 在噪声输入条件下，微调GPT-4o表现出更强的泛化能力，Top-1 Accuracy和CER均优于其他模型，表明其对不完美输入的适应性更强。
*   **实验设置合理性:** 实验覆盖10种语言和多个数据集子集，设置较为全面，考虑了语言多样性和词汇类型，但未涉及句子级音译和反向音译（本土脚本到罗马字母），存在一定局限性。
*   **对比分析:** 尽管IndicXlit在某些特定类别（如泰卢固语AK-Freq）上表现更好，但整体来看，LLMs（尤其是微调后）在音译任务上展现出显著潜力。

## Further Thoughts

论文揭示了通用LLMs通过微调可以快速适应特定任务（如音译）的潜力，这启发我们思考：未来是否可以通过少量任务特定数据和通用基础模型，替代为每个任务开发独立模型的传统方式？
此外，错误类型（图形、脚本、语义）的细致分类和噪声测试为模型优化提供了新思路，例如针对特定错误类型调整提示设计或训练数据分布，可能进一步提升模型在现实场景中的表现。