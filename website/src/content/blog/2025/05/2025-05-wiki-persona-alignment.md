---
title: "WikiPersonas: What Can We Learn From Personalized Alignment to Famous People?"
pubDatetime: 2025-05-19T15:39:48+00:00
slug: "2025-05-wiki-persona-alignment"
type: "arxiv"
id: "2505.13257"
score: 0.6735799934037471
author: "grok-3-latest"
authors: ["Zilu Tang", "Afra Feyza Akyürek", "Ekin Akyürek", "Derry Wijaya"]
tags: ["LLM", "Personalization", "Alignment", "Preference Inference", "Multi-Task Learning"]
institution: ["Boston University", "MIT", "Monash University Indonesia"]
description: "本文通过WikiPersona数据集和多任务模型结合推断前缀的方法，显著提升了大型语言模型的个性化对齐能力，并提出推理时移除前缀以缓解对齐税。"
---

> **Summary:** 本文通过WikiPersona数据集和多任务模型结合推断前缀的方法，显著提升了大型语言模型的个性化对齐能力，并提出推理时移除前缀以缓解对齐税。 

> **Keywords:** LLM, Personalization, Alignment, Preference Inference, Multi-Task Learning

**Authors:** Zilu Tang, Afra Feyza Akyürek, Ekin Akyürek, Derry Wijaya

**Institution(s):** Boston University, MIT, Monash University Indonesia


## Problem Background

当前大型语言模型（LLMs）的偏好对齐主要基于平均人类偏好，忽略了个体间的多样性和矛盾性，导致模型无法充分适应个性化需求。
论文通过引入 WikiPersona 数据集，基于真实名人的公开信息，探索如何在无先验用户知识的情况下实现细粒度的个性化对齐，解决现有方法在捕捉复杂个体偏好上的不足。

## Method

*   **核心思想:** 提出了一种个性化对齐框架，通过构建基于真实名人的数据集（WikiPersona）和多种对齐策略，使模型能够适应个体化偏好，尤其是在偏好冲突的主题上。
*   **数据集构建:** 
    *   **名人选择:** 基于11个偏好轴（如政治、饮食）选择50位名人，确保偏好多样性和矛盾性，使用GPT-4生成名人相关的个性化问题（personal）和轴相关问题（divergent）。
    *   **响应生成:** 使用基线模型（Zephyr-7B-beta）通过链式思维（Chain-of-Thought, CoT）提示生成多样化响应，结合聚类和奖励模型过滤，确保响应内容多样且质量均衡。
    *   **偏好标注:** 利用GPT-4作为‘个性化评判者’（LLM-as-personal-judge），通过三轮成对比较为每个问题标注偏好响应。
*   **对齐策略:** 
    *   **提示（Prompting）:** 在不改变模型参数的情况下，通过前缀（如名人姓名、少样本示例）条件化模型输出，适用于快速适应但受限于上下文长度和泛化能力。
    *   **个人模型（Personal Model, PM）:** 为每个名人单独微调一个LoRA适配器，使用直接偏好优化（DPO）损失，适合数据充足场景，但训练成本高且泛化能力差。
    *   **多任务模型（Multitask Model, MT）:** 在所有名人数据上微调单一适配器，使用推断的个性化前缀（如 persona_gpt4）区分个体偏好，结合DPO损失优化，既高效又能泛化到未训练名人。
*   **关键点:** 强调前缀质量对个性化效果的影响，提出在推理时移除前缀以缓解对齐税，同时通过5折交叉验证评估泛化能力。

## Experiment

*   **有效性:** 多任务模型（MT）结合高质量前缀（如 persona_gpt4）在训练和未训练名人上均显著提升个性化准确率，尤其在偏好冲突的 divergent 问题上，胜率较基线模型（Zephyr）提升约20%。
*   **对比分析:** 提示方法效果有限，尤其在小型模型上；个人模型（PM）在训练名人上表现优异，但在未训练名人上泛化能力几乎为零；多任务模型在性能和效率间取得最佳平衡。
*   **对齐税:** 个性化对齐导致推理任务性能下降（最高达10%），但安全性与事实性有所提升；移除前缀可有效缓解对齐税。
*   **实验设置合理性:** 实验覆盖多个模型（Zephyr等）、数据集子集（D_small 和 D_all）、任务类型（安全性、推理、事实性），并通过5折交叉验证和GPT-4评判验证结果，设置全面且合理。
*   **开销:** 数据集生成成本约500美元（主要用于GPT-4 API调用），响应生成耗费11个GPU天，显示出一定计算需求。

## Further Thoughts

论文中关于推断个性化前缀（inferred persona prefix）的想法非常具有启发性，未来可以通过动态更新前缀内容（例如基于用户实时交互数据）进一步提升个性化效果；此外，针对对齐税问题，可以探索多阶段训练或混合前缀策略，在保持个性化能力的同时优化模型的通用任务表现。