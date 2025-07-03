---
title: "MMReason: An Open-Ended Multi-Modal Multi-Step Reasoning Benchmark for MLLMs Toward AGI"
pubDatetime: 2025-06-30T07:14:38+00:00
slug: "2025-06-mmreason-benchmark"
type: "arxiv"
id: "2506.23563"
score: 0.4912334120876286
author: "grok-3-latest"
authors: ["Huanjin Yao", "Jiaxing Huang", "Yawen Qiu", "Michael K. Chen", "Wenzheng Liu", "Wei Zhang", "Wenjie Zeng", "Xikun Zhang", "Jingyi Zhang", "YuXin Song", "Wenhao Wu", "Dacheng Tao"]
tags: ["LLM", "Multi-Modal", "Reasoning", "Benchmark", "Evaluation"]
institution: ["Nanyang Technological University", "Tsinghua University", "Baidu Inc.", "University of California", "University of Science and Technology of China"]
description: "本文提出 MMReason 基准测试，通过开放式问题、数据过滤和双重评估机制，精准且全面地评估多模态大语言模型的长链推理能力。"
---

> **Summary:** 本文提出 MMReason 基准测试，通过开放式问题、数据过滤和双重评估机制，精准且全面地评估多模态大语言模型的长链推理能力。 

> **Keywords:** LLM, Multi-Modal, Reasoning, Benchmark, Evaluation

**Authors:** Huanjin Yao, Jiaxing Huang, Yawen Qiu, Michael K. Chen, Wenzheng Liu, Wei Zhang, Wenjie Zeng, Xikun Zhang, Jingyi Zhang, YuXin Song, Wenhao Wu, Dacheng Tao

**Institution(s):** Nanyang Technological University, Tsinghua University, Baidu Inc., University of California, University of Science and Technology of China


## Problem Background

多模态大语言模型（MLLMs）在迈向通用人工智能（AGI）的过程中，推理能力是关键，但现有基准测试在评估长链推理能力时存在不足：
1. 问题缺乏难度和多样性，难以挑战现代先进模型；
2. 多选题格式容易导致猜测（Guessability）和记忆（Memorization）问题，模型可能通过捷径而非推理得出答案；
3. 缺乏对中间推理步骤的评估，仅关注最终答案，忽略推理过程的完整性。
因此，作者提出 MMReason 基准测试，旨在更精准、全面地评估 MLLMs 的长链推理能力。

## Method

* **数据收集与多样性构建**：从现有基准测试中筛选问题，并从互联网上收集新的高难度问题，覆盖数学、商业、科学、工程、社会科学和健康等六个学科，包含从预科到大学、从基础到竞赛级别的多种难度，确保问题多样性和挑战性。
* **问题格式化与猜测消除**：将所有问题重新格式化为开放式问题，避免多选题格式导致的猜测问题，确保模型必须通过推理而非选择得出答案。
* **数据过滤与记忆消除**：设计多模型投票机制（Multi-Model Voting），使用多个强大 MLLMs（如 GPT-4o、Qwen2.5-VL 等）进行过滤，仅输入文本部分（不含图像），若模型能正确回答则剔除该问题，认为其可能被记忆或视觉相关性低，最终保留 1384 个鲁棒且视觉相关的问题。
* **中间步骤评估机制**：为问题标注详细的逐步解决方案，设计基于参考的三元评分机制（正确、不可验证、错误），使用 GPT-4o 提取模型响应的关键步骤并评分，确保中间推理步骤的评估可靠性。
* **最终答案评估策略**：鼓励模型生成多步推理响应，而非仅输出最终答案，通过 GPT-4o 提取最终答案并判断正确性，避免简单记忆或猜测的影响。

## Experiment

* **挑战性与有效性**：MMReason 对主流 MLLMs 构成显著挑战，最高性能的 GPT-4o 最终答案推理准确率仅为 25.7%，表明其适合评估长链推理能力；开源模型如 Qwen-2.5-VL-72B 达到 24.7%，接近闭源模型水平。
* **全面性与学科分析**：实验覆盖 13 个模型（3 个闭源，10 个开源），从学科维度（6 个学科）和数据来源维度（现有基准 vs 新数据）进行分析，新收集数据难度最高（如 GPT-4o 准确率仅 21.8%）；工程领域普遍表现较差（如 MiniCPM-V-2.6 仅 1%），反映领域知识挑战。
* **过滤效果**：多模型投票过滤后，文本输入准确率显著下降（如 GPT-4o 从 13.4% 降至 0.78%），多模态相关性率提升至 97.0%，证明了消除记忆和增强视觉相关性的有效性。
* **合理性与不足**：实验设置全面，但中间步骤评分具体数值披露不足，依赖 GPT-4o 评分可能引入偏差；三元评分机制一定程度上缓解了这一问题，设计合理。

## Further Thoughts

MMReason 的参考-based 三元评分机制为评估复杂推理过程提供了新思路，可推广至代码生成或逻辑推理等领域；多模型投票过滤机制不仅适用于基准测试构建，还可用于数据清洗或训练数据质量控制；论文揭示了 MLLMs 在多模态长链推理中的领域特定薄弱环节（如工程领域），提示未来研究可聚焦知识增强或多模态融合策略。