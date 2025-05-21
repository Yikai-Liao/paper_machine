---
title: "Table-R1: Region-based Reinforcement Learning for Table Understanding"
pubDatetime: 2025-05-18T13:40:18+00:00
slug: "2025-05-table-r1-reinforcement"
type: "arxiv"
id: "2505.12415"
score: 0.6850609445126368
author: "grok-3-latest"
authors: ["Zhenhe Wu", "Jian Yang", "Jiaheng Liu", "Xianjie Wu", "Changzai Pan", "Jie Zhang", "Yu Zhao", "Shuangyong Song", "Yongxiang Li", "Zhoujun Li"]
tags: ["LLM", "Table Understanding", "Reinforcement Learning", "Reasoning", "Supervised Fine-Tuning"]
institution: ["Beihang University", "TeleAI, China Telecom Corp Ltd", "Nanjing University"]
description: "本文提出 Table-R1 框架，通过基于区域的强化学习方法（RE-SFT 和 TARPO），显著提升大型语言模型在表格问答中的性能，平均提升 14.36 个百分点，同时减少 67.5% 的 token 消耗。"
---

> **Summary:** 本文提出 Table-R1 框架，通过基于区域的强化学习方法（RE-SFT 和 TARPO），显著提升大型语言模型在表格问答中的性能，平均提升 14.36 个百分点，同时减少 67.5% 的 token 消耗。 

> **Keywords:** LLM, Table Understanding, Reinforcement Learning, Reasoning, Supervised Fine-Tuning

**Authors:** Zhenhe Wu, Jian Yang, Jiaheng Liu, Xianjie Wu, Changzai Pan, Jie Zhang, Yu Zhao, Shuangyong Song, Yongxiang Li, Zhoujun Li

**Institution(s):** Beihang University, TeleAI, China Telecom Corp Ltd, Nanjing University


## Problem Background

表格数据由于其行列结构化的特性，对大型语言模型（LLMs）提出了独特的理解挑战，尤其是在复杂问答任务中，现有方法虽通过提示和思维链（CoT）技术展现潜力，但如何进一步优化 LLMs 的表格理解性能仍未充分探索。
本文旨在解决这一问题，通过引入基于区域的强化学习方法（Table-R1），增强模型对表格数据的推理能力，特别是在事实验证和问答等关键应用场景中。

## Method

*   **核心思想:** 通过在推理过程中引入表格区域（Table Region）证据，让模型先聚焦于与问题相关的表格子区域，再基于此生成答案，从而提升表格理解的准确性和效率。
*   **具体实现:** 提出 Table-R1 框架，包含两个主要阶段：
    *   **RE-SFT（Region-Enhanced Supervised Fine-Tuning）**：通过监督微调，指导模型在推理前识别相关表格区域，并将此步骤嵌入多种推理方法中，包括直接提示（Direct Prompting, DP）、文本思维链（Textual Chain-of-Thought, TCoT）、符号思维链（Symbolic Chain-of-Thought, SCoT）和程序思维（Program-of-Thought, PoT）。具体而言，模型会在 CoT 过程中插入表格区域识别步骤，基于最小区域数据进行后续推理。
    *   **TARPO（Table-Aware Group Relative Policy Optimization）**：基于强化学习优化方法，扩展自 GRPO，引入混合奖励机制以平衡表格区域识别的准确性和最终答案的正确性。奖励计算包括区域奖励（基于 IoU 计算行列重叠）和答案奖励（基于正确性），并通过动态权重（随训练进程减少区域奖励比例）调整重点。此外，TARPO 引入一致性偏好，通过惩罚区域和答案优化方向不一致的情况，确保推理过程的合理性。
*   **关键特点:** 不修改模型架构，仅通过训练和推理策略调整实现性能提升，同时控制 token 消耗以提高效率。

## Experiment

*   **有效性:** Table-R1 在多个基准数据集（TableBench, WikiTQ, WikiSQL）上显著提升了性能，基于不同基模型（3B-8B 参数）平均提升 14.36 个百分点，尤其在 PoT 任务中表现突出（如 Qwen3-8B 基模型在 TableBench PoT 上从 29.15 提升至 44.54），甚至 8B 参数的 Qwen3 模型整体超越了 GPT-4o。
*   **效率提升:** TARPO 相比 GRPO 减少了 67.5% 的响应 token 消耗，有效解决了引入表格区域后 token 增加的问题（如在 TableBench 上从 GRPO 的 1549 个 token 降至 612 个）。
*   **实验设置合理性:** 实验覆盖多种基模型（Qwen, Llama, DeepSeek 等）和数据集，设置了多层次对比（基线、仅 RE-SFT、完整 Table-R1），并通过消融研究验证了各组件贡献（如 TARPO 的动态权重和一致性偏好对泛化能力的提升）。
*   **局限性:** 实验局限于中小规模模型（3B-8B）和有限 token 长度（输入 8192，输出 2048），对更大模型或更长序列的表现有待验证，部分跨领域测试（如 WikiTQ）性能略有下降。

## Further Thoughts

表格区域作为推理中间步骤的思路具有广泛适用性，不仅限于表格理解，还可扩展至其他结构化数据任务（如图表分析、数据库查询），通过聚焦关键区域减少模型处理无关信息的负担；此外，TARPO 的动态奖励权重和一致性偏好机制为强化学习在多目标优化中的应用提供了新思路，值得探索如何在其他复杂任务中平衡中间步骤和最终结果的奖励分配。