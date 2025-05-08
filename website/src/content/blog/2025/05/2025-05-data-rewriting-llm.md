---
title: "Rewriting Pre-Training Data Boosts LLM Performance in Math and Code"
pubDatetime: 2025-05-05T07:38:43+00:00
slug: "2025-05-data-rewriting-llm"
type: "arxiv"
id: "2505.02881"
score: 0.8052080118391378
author: "grok-3-latest"
authors: ["Kazuki Fujii", "Yukito Tajima", "Sakae Mizuki", "Hinari Shimada", "Taihei Shiotani", "Koshiro Saito", "Masanari Ohi", "Masaki Kawamura", "Taishi Nakamura", "Takumi Okamoto", "Shigeki Ishida", "Kakeru Hattori", "Youmi Ma", "Hiroya Takamura", "Rio Yokota", "Naoaki Okazaki"]
tags: ["LLM", "Pre-Training", "Data Quality", "Code Generation", "Mathematical Reasoning"]
institution: ["Institute of Science Tokyo, Department of Computer Science", "National Institute of Advanced Industrial Science and Technology", "Institute of Science Tokyo, Institute of Integrated Research, Supercomputing Research Center"]
description: "本文通过系统性重写预训练数据，构建 SwallowCode 和 SwallowMath 数据集，显著提升了大型语言模型在代码生成和数学推理任务上的性能，提出了一种创新的‘改造并保留’数据处理范式。"
---

> **Summary:** 本文通过系统性重写预训练数据，构建 SwallowCode 和 SwallowMath 数据集，显著提升了大型语言模型在代码生成和数学推理任务上的性能，提出了一种创新的‘改造并保留’数据处理范式。 

> **Keywords:** LLM, Pre-Training, Data Quality, Code Generation, Mathematical Reasoning

**Authors:** Kazuki Fujii, Yukito Tajima, Sakae Mizuki, Hinari Shimada, Taihei Shiotani, Koshiro Saito, Masanari Ohi, Masaki Kawamura, Taishi Nakamura, Takumi Okamoto, Shigeki Ishida, Kakeru Hattori, Youmi Ma, Hiroya Takamura, Rio Yokota, Naoaki Okazaki

**Institution(s):** Institute of Science Tokyo, Department of Computer Science, National Institute of Advanced Industrial Science and Technology, Institute of Science Tokyo, Institute of Integrated Research, Supercomputing Research Center


## Problem Background

大型语言模型（LLMs）在程序合成和数学推理方面的性能受限于预训练数据的质量，现有公开数据集（如 The-Stack-v1/v2 和 Finemath-4+）常包含噪声、冗余和风格不一致的内容，通过规则过滤或模型评分虽有改进，但仍不足以支持高效的模型学习。
作者提出通过系统性重写预训练数据，消除这些问题，提升数据质量和模型性能。

## Method

*   **核心思想:** 提出一种‘改造并保留’（transform-and-retain）的策略，利用大型语言模型（LLM）对低质量预训练数据进行系统性重写，而非传统排除性过滤，最大化数据利用率，针对代码和数学领域分别构建高质量数据集。
*   **SwallowCode（代码数据集）构建方法:** 基于 The-Stack-v2 数据集，设计四阶段流水线：
    *   **语法验证:** 使用 Python 的 compile() 函数过滤无效代码，减少约10.6%的样本量，确保代码语法正确。
    *   **基于 pylint 的风格过滤:** 应用 pylint 工具，设置质量阈值（评分≥7.0），并通过自定义启发式算法惩罚过于冗长的注释，进一步减少约34.3%的样本量，提升代码结构质量。
    *   **风格引导代码重写（SGCR）:** 使用 Llama-3.3-70B-Instruct 模型，根据 Google Python Style Guide 的10项准则（如变量命名、类型注解、错误处理）重写代码，提升风格一致性和可读性。
    *   **自包含优化重写（SCOR）:** 在 SGCR 基础上进一步优化语义内容，确保代码自包含（解决外部依赖）、优化算法效率（如将低效算法替换为动态规划），并将琐碎代码转化为有教学价值的示例。
*   **SwallowMath（数学数据集）构建方法:** 基于 Finemath-4+ 数据集，设计 LLM 驱动的重写流程：
    *   使用 Llama-3.3-70B-Instruct 模型去除冗余内容（如网页头尾、元数据），恢复缺失上下文。
    *   重格式化解答为简洁且逐步的解释，确保内容逻辑清晰，适合数学推理任务。
*   **关键特点:** 重写过程不丢弃低质量数据，而是通过 LLM 深度改造提升其价值；流水线设计对编程语言具有通用性（仅需语法检查和 linter 工具）；实验中控制了训练预算（50 billion tokens）和模型架构（Llama-3.1-8B），确保结果可比性。

## Experiment

*   **有效性:** SwallowCode 在 HumanEval 和 HumanEval+ 基准上，相比 Stack-Edu 数据集，pass@1 分别提升了 +17.0 和 +17.7，显著优于其他公开代码数据集（如 The-Stack-v1/v2）；SwallowMath 在 GSM8K 和 MATH 基准上，相比原始 Finemath-4+，准确率分别提升了 +12.4 和 +7.6，显示出重写策略在代码和数学任务上的显著效果。
*   **消融研究:** 每个流水线阶段（语法过滤、风格过滤、SGCR、SCOR）均逐步贡献性能提升，其中重写阶段（SGCR 和 SCOR）贡献最大，例如 SGCR 提升 HumanEval 得分超9点，SCOR 进一步提升超5点。
*   **实验设置合理性:** 实验在 50 billion token 预算下进行，基于 Llama-3.1-8B 持续预训练，控制了模型架构和数据混合比例；通过 Jaccard 相似度检查避免测试集泄露，确保结果可靠性；消融研究覆盖了每个处理阶段的影响，设计全面。
*   **局限性:** 实验仅限于 Python 代码，未验证其他编程语言的效果；训练预算固定为 50B tokens，未探索更大规模训练的影响；重写依赖 Llama-3.3-70B-Instruct，可能引入该模型的偏见。

## Further Thoughts

‘改造并保留’的理念非常具有启发性，传统数据处理多通过过滤丢弃低质量样本，而本文利用 LLM 重写数据，将低质量内容转化为高质量资源，这种思路可推广至其他领域，如自然语言文本清洗、对话数据优化或多模态数据处理；此外，流水线对编程语言的通用性（仅需语法检查和 linter 工具）启发我们思考如何将类似方法应用于其他结构化数据（如 SQL 查询、配置文件）或非结构化数据的质量提升；另一个值得探索的方向是重写过程中是否可以引入多模型协作或领域专家反馈，进一步减少模型偏见并提升重写质量。