---
title: "Pre-training Large Memory Language Models with Internal and External Knowledge"
pubDatetime: 2025-05-21T19:26:03+00:00
slug: "2025-05-large-memory-lm"
type: "arxiv"
id: "2505.15962"
score: 0.859059367204009
author: "grok-3-latest"
authors: ["Linxi Zhao", "Sofian Zalouk", "Christian K. Belardi", "Justin Lovelace", "Jin Peng Zhou", "Kilian Q. Weinberger", "Yoav Artzi", "Jennifer J. Sun"]
tags: ["LLM", "External Knowledge", "Pre-Training", "Knowledge Retrieval", "Factuality"]
institution: ["Cornell University"]
description: "本文提出 Large Memory Language Models (LMLM)，通过将事实知识外部化存储到数据库并在预训练和推理时动态查询，显著提升了事实准确性和知识可编辑性，同时保持了语言能力。"
---

> **Summary:** 本文提出 Large Memory Language Models (LMLM)，通过将事实知识外部化存储到数据库并在预训练和推理时动态查询，显著提升了事实准确性和知识可编辑性，同时保持了语言能力。 

> **Keywords:** LLM, External Knowledge, Pre-Training, Knowledge Retrieval, Factuality

**Authors:** Linxi Zhao, Sofian Zalouk, Christian K. Belardi, Justin Lovelace, Jin Peng Zhou, Kilian Q. Weinberger, Yoav Artzi, Jennifer J. Sun

**Institution(s):** Cornell University


## Problem Background

大型语言模型（LLMs）将事实知识和语言能力紧密耦合在模型参数中，导致事实知识难以更新、遗忘或验证，容易产生幻觉或过时信息，限制了模型在事实准确性和可控性方面的表现。
论文旨在通过将事实知识从模型参数中解耦并存储到外部数据库，解决这些问题，提高参数效率和知识的可编辑性。

## Method

*   **核心思想:** 提出一种新的语言模型架构——Large Memory Language Models (LMLM)，通过将事实知识外部化存储到数据库中，减少模型参数对事实的记忆负担，学会在需要时查询外部知识。
*   **数据准备:** 使用一个小型语言模型（Annotator）从预训练语料中提取实体级事实三元组（entity, relation, value），并构建一个紧凑的外部数据库，同时在原始文本中插入查询调用（lookup calls），以便模型学习何时依赖外部知识。
*   **预训练过程:** 采用标准的下一词预测任务，但在计算损失时屏蔽从数据库返回的事实值（masked from loss），以阻止模型将这些事实记忆到参数中，鼓励模型学习基于查询结果生成文本。
*   **推理过程:** 模型在生成文本时，遇到特定标记会触发数据库查询，通过模糊匹配（fuzzy matching）检索相关事实值，并将其融入生成内容中，减少对内部参数的依赖。
*   **技术细节:** 数据库检索基于句嵌入空间的余弦相似度（使用 ALL-MiniLM-L6-V2 模型），设置相似度阈值为 0.6 以确保检索质量；同时，模型词汇表中增加了四个特殊标记，用于格式化查询和返回值的交互。

## Experiment

*   **有效性:** LMLM 在事实准确性基准（如 FactScore 和 T-REx）上显著优于同规模标准模型，例如 LMLM-382M 的 FactScore 比标准模型提高 17.9%，接近更大规模模型（如 LLaMA2-7B）的表现。
*   **训练效率:** 在预训练过程中，LMLM 表现出更低的验证困惑度（perplexity），表明外部化知识提高了训练效率。
*   **通用能力:** 在自然语言理解（NLU）任务上，LMLM 与标准模型表现相当，证明知识外部化未损害语言能力。
*   **知识编辑:** 在 TOFU 机器遗忘基准测试中，LMLM 通过简单删除数据库条目实现即时遗忘，且不影响模型通用能力，优于现有方法（如 NPO）。
*   **实验设置合理性:** 实验涵盖了语言建模、事实准确性和知识编辑等多维度评估，与同规模及更大规模模型进行了对比；但由于计算资源限制，模型和数据集规模较小，可能影响结果的普适性。

## Further Thoughts

LMLM 的知识外部化思想启发我们进一步探索如何将外部数据库与动态更新的知识图谱或实时数据源结合，以适应快速变化的事实信息；此外，是否可以将这一方法扩展到非实体级知识（如长篇文本或复杂推理知识）的外部化存储和检索，也是一个值得研究的方向，可能为构建更灵活、可控的语言模型提供新路径。