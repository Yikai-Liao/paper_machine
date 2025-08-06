---
title: "Refine-n-Judge: Curating High-Quality Preference Chains for LLM-Fine-Tuning"
pubDatetime: 2025-08-03T01:56:03+00:00
slug: "2025-08-refine-judge-curation"
type: "arxiv"
id: "2508.01543"
score: 0.8112419396095323
author: "grok-3-latest"
authors: ["Unknown"]
tags: ["LLM", "Data Curation", "Refinement", "Judgment", "Fine-Tuning"]
institution: ["Unknown"]
description: "本文提出 Refine-n-Judge 框架，通过 LLM 的自优化和判断能力自动化生成高质量偏好数据集，显著提升微调后模型性能。"
---

> **Summary:** 本文提出 Refine-n-Judge 框架，通过 LLM 的自优化和判断能力自动化生成高质量偏好数据集，显著提升微调后模型性能。 

> **Keywords:** LLM, Data Curation, Refinement, Judgment, Fine-Tuning

**Authors:** Unknown

**Institution(s):** Unknown


## Problem Background

大型语言模型（LLMs）的性能高度依赖于高质量训练数据，而传统人工反馈方法成本高、扩展性差且存在噪声问题；论文旨在解决数据集质量提升的自动化和可扩展性挑战，避免单纯迭代优化可能导致的质量下降问题。

## Method

* **核心思想**：提出 Refine-n-Judge 框架，通过结合 LLM 的自优化（Refinement）和判断（Judgment）能力，自动化生成高质量偏好数据集，用于模型微调。
* **具体实现**：
  * **数据集整理阶段**：从初始回答（来自公开数据集或 LLM 生成）开始，进入迭代循环。优化步骤中，LLM 基于一组标准（如准确性、完整性、清晰度、简洁性和相关性）生成反馈并改进回答；判断步骤中，LLM 评估新回答是否优于前一个，若是则继续迭代，若否则终止（最多设置 10 次迭代上限）。
  * **微调阶段**：利用生成的按质量排序的回答链（preference chains），选取最高质量回答与查询配对，通过监督微调（SFT）训练 LLM。
  * **技术细节**：使用结构化提示（包括反馈、优化和判断三个提示）确保过程系统性；通过随机化回答位置减少判断中的位置偏差；同一 LLM 可同时担任优化和判断角色，也可使用不同模型。
* **关键创新**：将优化与判断结合，避免无意义的迭代，确保每次改进均经过验证，同时无需人工干预即可实现数据集质量提升。

## Experiment

* **有效性**：Refine-n-Judge 在多个数据集（如 Acronym、TruthfulQA、UltraChat）上显著优于基线方法（如零样本生成和拒绝采样），GPT-4 评判胜率高达 89%；微调后的 Llama 3.1-8B 和 Llama 3.3-70B 模型在 AlpacaEval 和 MT-Bench 等基准上性能提升明显，例如 Llama 3.1-8B 的 AlpacaEval 胜率从 79.3% 提升至 84.8%。
* **鲁棒性**：针对噪声数据（如不准确、冗长、误导性回答）的测试表明框架能有效纠正低质量输入，例如在 TruthfulQA 数据集上真实性评分提升 92%，UltraChat 上冗长度减少 52%。
* **合理性与局限**：实验设置全面，覆盖多种任务和数据集，并通过多 LLM 评委（GPT-4、Claude 等）验证结果一致性；但判断一致性在迭代后期下降（从 100% 降至 50%），可能影响终止决策可靠性；此外，计算成本较高，每次迭代需多次调用 LLM。

## Further Thoughts

Refine-n-Judge 框架中优化与判断结合的思路启发了我，是否可以将这种机制扩展到多模型协作场景，例如不同 LLM 分别担任优化和判断角色以减少偏差；此外，针对后期判断一致性下降的问题，可引入多评委投票或动态调整终止阈值；更进一步，这种迭代优化-判断循环是否适用于其他领域（如图像生成模型）也值得探索，关键在于设计有效的判断机制。