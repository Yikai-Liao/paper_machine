---
title: "LookAlike: Consistent Distractor Generation in Math MCQs"
pubDatetime: 2025-05-03T19:18:06+00:00
slug: "2025-05-lookalike-distractor-gen"
type: "arxiv"
id: "2505.01903"
score: 0.6806483166986361
author: "grok-3-latest"
authors: ["Nisarg Parikh", "Nigel Fernandez", "Alexander Scarlatos", "Simon Woodhead", "Andrew Lan"]
tags: ["LLM", "Preference Optimization", "Synthetic Data", "Consistency", "Education"]
institution: ["University of Massachusetts Amherst", "Eedi"]
description: "本文提出LOOK A LIKE方法，通过合成偏好对挖掘和交替优化策略，显著提高了数学多选题中干扰项和错误描述生成的一致性，超越了现有最先进方法。"
---

> **Summary:** 本文提出LOOK A LIKE方法，通过合成偏好对挖掘和交替优化策略，显著提高了数学多选题中干扰项和错误描述生成的一致性，超越了现有最先进方法。 

> **Keywords:** LLM, Preference Optimization, Synthetic Data, Consistency, Education

**Authors:** Nisarg Parikh, Nigel Fernandez, Alexander Scarlatos, Simon Woodhead, Andrew Lan

**Institution(s):** University of Massachusetts Amherst, Eedi


## Problem Background

在数学多选题（MCQs）的教育评估中，干扰项（distractors）应反映学生常见错误以帮助识别误解，但现有大型语言模型（LLMs）在生成干扰项时，常常无法确保与输入错误描述的一致性，限制了自动化生成在教育中的应用。
论文旨在解决如何提高错误描述与干扰项生成之间的一致性问题，同时提升错误描述本身的生成质量。

## Method

*   **核心思想:** 通过偏好优化（Preference Optimization）提高数学多选题中干扰项和错误描述生成的一致性，利用模型自身的生成不一致性作为训练信号。
*   **具体实现:** 提出了名为‘LOOK A LIKE’的方法，包含两项创新：
    *   **合成偏好对挖掘（Synthetic Preference Pairs Mining）:** 针对每个问题和错误描述，模型过生成（overgenerate）多个干扰项或错误描述，将与真实值精确匹配的作为‘偏好’样本，不匹配的作为‘非偏好’样本，用于直接偏好优化（DPO）训练。这种方式避免了昂贵的人工标注，实现了可扩展的偏好数据构建。
    *   **交替优化策略（Alternating Optimization）:** 为解决DPO训练不稳定性问题，提出在训练中交替使用监督微调（SFT）和DPO目标，分别按批次（per-batch）和按轮次（per-epoch）切换，确保模型既学习一致性偏好，又保持对真实数据的拟合能力，避免质量下降。
*   **关键点:** 方法不依赖外部标注，利用模型自身弱点动态生成训练数据，同时通过交替优化平衡学习目标，适用于干扰项和错误描述的双任务生成。

## Experiment

*   **有效性:** 在包含1434个数学多选题的真实数据集上，LOOK A LIKE 在干扰项生成准确率达到51.6%，错误生成准确率达到57.2%，分别比现有最先进方法DiVERT（45.6%和47.7%）提高了约6%和9.5%，提升显著。
*   **优越性:** 相比使用人工标注偏好数据的DPO-GT，LOOK A LIKE 表现相当，验证了合成偏好对的有效性；交替优化策略优于其他DPO正则化方法（如RPO和DPOP），提升了训练稳定性。
*   **实验设置:** 数据集覆盖10-13岁学生数学题目，采用5折交叉验证，评估指标包括干扰项的精确匹配（Exact Match）和错误描述的LLM-as-a-Judge（GPT-4o-mini判断数学等价性），设置全面合理，但局限于中学数学，未验证跨学科泛化性。
*   **额外验证:** 定性分析和人类评估进一步确认了生成错误的教学一致性，LOOK A LIKE 生成的错误更具体且与干扰项相关。

## Further Thoughts

论文中利用模型自身生成不一致性作为合成偏好对的训练信号，这一自监督方式极具启发性，不仅降低了人工标注成本，还能动态适应模型弱点，可推广至其他需要一致性或指令跟随的领域，如对话系统或代码生成；此外，交替优化策略揭示了在偏好优化中平衡真实数据拟合与偏好学习的重要性，值得在多目标优化任务中进一步探索。