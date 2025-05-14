---
title: "ToolACE-DEV: Self-Improving Tool Learning via Decomposition and EVolution"
pubDatetime: 2025-05-12T12:48:30+00:00
slug: "2025-05-toolace-dev-evolution"
type: "arxiv"
id: "2505.07512"
score: 0.7085683608852088
author: "grok-3-latest"
authors: ["Xu Huang", "Weiwen Liu", "Xingshan Zeng", "Yuefeng Huang", "Xinlong Hao", "Yuxian Wang", "Yirong Zeng", "Chuhan Wu", "Yasheng Wang", "Ruiming Tang", "Defu Lian"]
tags: ["LLM", "Tool Learning", "Self-Evolution", "Task Decomposition", "Data Generation"]
institution: ["University of Science and Technology of China", "Huawei Noah’s Ark Lab", "Huawei Technologies Co., Ltd"]
description: "本文提出 ToolACE-DEV 框架，通过任务分解和自进化机制显著提升大型语言模型的工具调用能力，减少对高级模型的依赖，并在多个基准数据集上取得优异性能。"
---

> **Summary:** 本文提出 ToolACE-DEV 框架，通过任务分解和自进化机制显著提升大型语言模型的工具调用能力，减少对高级模型的依赖，并在多个基准数据集上取得优异性能。 

> **Keywords:** LLM, Tool Learning, Self-Evolution, Task Decomposition, Data Generation

**Authors:** Xu Huang, Weiwen Liu, Xingshan Zeng, Yuefeng Huang, Xinlong Hao, Yuxian Wang, Yirong Zeng, Chuhan Wu, Yasheng Wang, Ruiming Tang, Defu Lian

**Institution(s):** University of Science and Technology of China, Huawei Noah’s Ark Lab, Huawei Technologies Co., Ltd


## Problem Background

大型语言模型（LLMs）在访问实时信息和执行复杂任务时存在局限性，如事实性错误和无法获取最新数据，而工具集成被认为是解决这些问题的有效途径。
然而，当前通过高级模型（如 GPT-4）合成数据进行蒸馏的方法面临高昂的推理成本、数据兼容性问题（合成数据与目标模型知识范围不匹配导致幻觉）以及数据隐私风险。
因此，论文提出了一种自进化框架 ToolACE-DEV，旨在减少对高级模型的依赖，让轻量级模型通过自身迭代改进工具使用能力。

## Method

*   **核心思想:** 通过任务分解和自进化机制，提升大型语言模型的工具使用能力，避免对高级模型的依赖，实现轻量级模型的自主改进。
*   **阶段一 - 工具文档适应（Tool Documentation Adaption）:** 通过对工具文档的持续预训练，增强模型对工具定义、语法和功能的理解，类似于领域特定预训练，但特别针对工具使用场景设计，适用于已进行指令调优的模型，保持其指令跟随能力。
*   **阶段二 - 查询感知的工具生成与调用（Query-Aware Tool Generation and Invocation）:** 将工具学习任务分解为两个子任务：基于查询生成候选工具（Tool Generation）和基于查询与候选工具进行工具调用（Tool Invocation）。这种分解通过两个训练目标增强模型对查询与工具关系的理解，并为其自进化奠定基础。
*   **阶段三 - 自进化（Self-Evolution）:** 在前两阶段训练后，模型针对新查询自主生成候选工具和工具调用，形成新的训练数据。通过 top-k 采样和多数投票（self-consistency decoding）确保生成数据的质量，并通过规则检查过滤无效样本（如格式错误或幻觉工具）。最终，模型基于自生成数据进行迭代更新，实现能力提升。
*   **关键点:** 该框架不依赖外部高级模型，强调任务分解和自生成数据的结构化训练，逐步构建模型的工具使用能力，同时通过采样和过滤机制控制数据质量。

## Experiment

*   **有效性:** ToolACE-DEV 在 Berkeley Function Calling Leaderboard (BFCL) 上取得了与更大规模模型（如 LLaMA-3-70B）甚至闭源模型（如 GPT-4o）相当的性能，尤其在 8B 规模下表现突出，整体准确率达到 82.44%，在同规模模型中名列前茅；在 API-Bank 和 T-Eval 等其他基准数据集上也显著优于基线模型。
*   **对比分析:** 与未使用自进化框架的 ToolACE-8B 相比，ToolACE-DEV 展现了进一步提升，尤其在更具挑战性的 Live 子集（真实用户查询）上表现更好，表明自进化生成的数据具有更高的信息价值。
*   **自进化迭代效果:** 在三轮自进化中，模型性能持续提升，尤其在 Live 数据集上的增幅更大，但增幅随迭代次数增加而减弱，可能是因为模型信心逐渐饱和，自生成数据的多样性下降。
*   **消融研究:** 消融实验表明，工具文档适应和工具生成任务对性能提升有显著贡献，尤其是联合训练（Invo.+Gen. w. Adaption）效果最佳，验证了各阶段训练目标的有效性。
*   **泛化性:** 在不同模型（如 Qwen2.5 系列、Mistral-7B）和不同规模（1.5B 到 8B）上，框架均有效，但较大模型（7-8B）在自进化中表现更稳定，小模型（1.5B）易过拟合，性能波动较大。
*   **实验设置合理性:** 实验涵盖了多个基准数据集、不同规模和架构的模型，使用 LoRA 参数高效微调，设置全面且结果可信。

## Further Thoughts

任务分解的策略为复杂 NLP 任务提供了新思路，是否可以将类似方法应用于多步骤推理或多模态交互等场景？
自进化机制展示了轻量级模型通过迭代自生成数据实现能力提升的可能性，未来是否可以通过引入外部反馈（如用户交互）进一步优化自进化过程？
模型规模与自进化效果的关系提示我们，可能需要根据模型规模调整迭代策略或数据过滤机制，以避免小模型过拟合或大模型数据冗余的问题。