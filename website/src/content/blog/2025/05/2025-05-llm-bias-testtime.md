---
title: "Think Again! The Effect of Test-Time Compute on Preferences, Opinions, and Beliefs of Large Language Models"
pubDatetime: 2025-05-26T07:41:21+00:00
slug: "2025-05-llm-bias-testtime"
type: "arxiv"
id: "2505.19621"
score: 0.7520816238744367
author: "grok-3-latest"
authors: ["George Kour", "Itay Nakash", "Ateret Anaby-Tavor", "Michal Shmueli-Scheuer"]
tags: ["LLM", "Bias Evaluation", "Test Time Compute", "Reasoning", "Neutrality"]
institution: ["IBM Research AI"]
description: "本文提出 POBs 基准数据集评估大型语言模型的主观倾向，发现模型常表现出进步主义-集体主义偏见，且测试时计算对改善中立性和一致性效果有限，而新模型版本偏见更强。"
---

> **Summary:** 本文提出 POBs 基准数据集评估大型语言模型的主观倾向，发现模型常表现出进步主义-集体主义偏见，且测试时计算对改善中立性和一致性效果有限，而新模型版本偏见更强。 

> **Keywords:** LLM, Bias Evaluation, Test Time Compute, Reasoning, Neutrality

**Authors:** George Kour, Itay Nakash, Ateret Anaby-Tavor, Michal Shmueli-Scheuer

**Institution(s):** IBM Research AI


## Problem Background

随着大型语言模型（LLMs）广泛融入人类生活并影响决策，其在主观领域（如社会、文化、伦理和个人偏好）中的倾向性（preferences, opinions, beliefs）可能通过微妙偏见塑造用户观点和行为，尤其在商业应用中，模型应保持中立或反映特定价值观；论文旨在评估 LLMs 是否表现出主观倾向、倾向强度如何，以及测试时计算是否能减轻这些偏见。

## Method

* **基准数据集设计：** 构建了 POBs（Preference, Opinion, and Belief Survey）基准数据集，包含 20 个主观话题，分为极性话题（polar topics，如‘AI 谨慎 vs. 乐观’）和非极性话题（non-polar topics，如职业偏好），每个话题包含 12-38 个 Likert 量表问题，用于揭示模型偏见和一致性。
* **提示策略：** 采用三种提示方式评估模型行为：
  * **直接提示（Direct）：** 直接要求模型选择 Likert 量表选项，输出答案。
  * **推理提示（Reasoning）：** 要求模型在回答前提供推理过程（使用 <think> 标签），然后给出最终答案。
  * **自我反思提示（Self-Reflection）：** 在推理基础上，要求模型回顾并重新考虑其初始回答（使用 <rethink> 标签），最终输出修正答案。
* **评估指标：** 引入三个关键指标量化模型表现：
  * **可靠性（Reliability）：** 通过多次重复同一问题，计算答案极性（polarity）的平均归一化绝对差异，评估模型输出的稳定性。
  * **非中立性指数（Non-Neutrality Index, NNI）：** 平均问题答案的绝对极性值，量化模型在争议话题上的立场强度（越高表示越不中立）。
  * **话题一致性指数（Topical Consistency Index, TCI）：** 计算同一极性话题内各问题平均极性的标准差（取反），评估模型在同一话题内的立场一致性（越高表示越一致）。
* **实验对象：** 选择了 10 个主流开源和闭源 LLMs（如 GPT-4o、LLaMA 3.3、DeepSeek 等），并对比同一厂商的新旧模型版本，分析其行为和偏见演变。

## Experiment

* **有效性：** 实验表明大多数 LLMs 在争议话题上表现出较强的立场（高 NNI），倾向于进步主义-集体主义观点；增加测试时计算（推理和自我反思）对改善中立性（降低 NNI）和一致性（提高 TCI）的效果有限，变化幅度较小。
* **模型差异：** 新版本模型（如 LLaMA 3.3 对比 LLaMA 3.2）表现出更强的偏见（更高 NNI）和更低的一致性（更低 TCI），表明新模型更倾向于特定观点且内部矛盾增加；较大模型（如 LLaMA 3.3-70B）在直接提示下可靠性较高，但推理和反思时可靠性下降。
* **实验设置合理性：** 实验覆盖多种模型、提示策略和话题类型，指标设计科学（NNI 和 TCI 有效量化偏见和一致性），数据集中包含中立和拒绝选项以评估模型回避倾向；但存在局限性，如缺乏人类基准对比、仅限于英文数据集、提示策略可能影响结果，降低了普适性。
* **显著性：** 测试时计算对中立性和一致性的改善不显著（表 2 显示 NNI 和 TCI 变化有限），而新模型偏见增强的趋势显著（图 2 和图 5 显示新模型在进步主义-集体主义象限更集中），揭示了一个值得关注的盲点。

## Further Thoughts

论文揭示模型在自我报告偏见时往往低估自身倾向（图 5），这启发是否可以通过多轮交互或复杂自我评估机制提高模型对自身偏见的认知；此外，测试时计算对中立性改善有限，是否意味着需要在训练阶段（如预训练或后训练）引入更强的中立性约束？另一个想法是针对不同文化背景定制 POBs 数据集，探索模型在跨文化语境下的偏见表现差异。