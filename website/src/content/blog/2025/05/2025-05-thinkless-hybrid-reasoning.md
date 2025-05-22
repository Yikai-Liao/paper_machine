---
title: "Thinkless: LLM Learns When to Think"
pubDatetime: 2025-05-19T17:24:16+00:00
slug: "2025-05-thinkless-hybrid-reasoning"
type: "arxiv"
id: "2505.13379"
score: 0.857657365288996
author: "grok-3-latest"
authors: ["Gongfan Fang", "Xinyin Ma", "Xinchao Wang"]
tags: ["LLM", "Reasoning", "Reinforcement Learning", "Hybrid Reasoning", "Efficiency"]
institution: ["National University of Singapore"]
description: "本文提出 Thinkless 框架，通过解耦组相对策略优化（DeGRPO）算法，使大型语言模型根据任务复杂度和自身能力自适应选择推理模式，显著提升推理效率并保持性能。"
---

> **Summary:** 本文提出 Thinkless 框架，通过解耦组相对策略优化（DeGRPO）算法，使大型语言模型根据任务复杂度和自身能力自适应选择推理模式，显著提升推理效率并保持性能。 

> **Keywords:** LLM, Reasoning, Reinforcement Learning, Hybrid Reasoning, Efficiency

**Authors:** Gongfan Fang, Xinyin Ma, Xinchao Wang

**Institution(s):** National University of Singapore


## Problem Background

推理型语言模型（Reasoning Language Models）在处理复杂任务时表现出色，但对所有问题统一采用详细的链式推理（Chain-of-Thought Reasoning）会导致不必要的计算开销，尤其是在面对简单问题时，产生冗余 token、增加内存占用和推理时间。
论文提出一个核心问题：语言模型能否学会‘何时思考’，即根据任务复杂度和自身能力动态选择短格式（Short-Form）或长格式（Long-Form）推理方式，以在效率和性能之间取得平衡。

## Method

* **整体框架：Thinkless**：这是一个基于强化学习的框架，旨在训练语言模型自适应选择推理模式，通过两个阶段实现目标。
* **第一阶段：蒸馏预热（Distillation for Warm-up）**：
  * 使用两个专家模型生成配对的长短格式响应数据集，其中一个模型擅长长链推理（Reasoning Model），另一个擅长简洁回答（Instruction-Following Model）。
  * 通过监督微调（Supervised Fine-Tuning, SFT），目标模型学习在控制标记 `<think>`（长格式）和 `<short>`（短格式）的引导下生成两种风格的响应。
  * 这一阶段确保模型具备生成两种推理模式的基础能力，并通过配对数据保持响应分布平衡，为后续强化学习提供多样化起点。
* **第二阶段：基于解耦组相对策略优化（Decoupled Group Relative Policy Optimization, DeGRPO）的强化学习**：
  * 将模式选择建模为强化学习问题，模型输出第一个 token 作为控制标记（`<think>` 或 `<short>`），决定后续推理模式。
  * 提出 DeGRPO 算法，将学习目标分解为两个部分：控制标记的选择（Mode Selection）和响应内容的准确性提升（Accuracy Improvement）。
  * 通过对控制标记和响应标记的梯度独立归一化，并引入权重参数 α 平衡两者的优化贡献，有效避免传统 GRPO 算法中的模式崩塌问题（Mode Collapse）。
  * 奖励函数设计鼓励简洁且正确的回答（短格式正确答案奖励高于长格式），推动效率提升。
* **关键创新**：DeGRPO 的解耦设计解决了控制标记更新被响应标记长序列稀释的问题，确保模式选择和响应质量的平衡优化。

## Experiment

* **数据集与设置**：实验基于数学推理数据集（AIME、Minerva Algebra、MATH-500、GSM8K），使用 DeepSeek-R1-Distill-Qwen-1.5B 作为基础模型，对比了基线模型、短链推理技术（如模型合并和 CoT-Valve）以及路由器方法。
* **效果显著性**：Thinkless 显著减少了长链推理的使用比例（50%-90%），例如在 Minerva Algebra 上仅对 25.88% 的样本启用长链推理，token 使用量减少到原来的三分之一，性能仅下降 1% 以内，效率提升明显。
* **对比优势**：相比模型合并和 CoT-Valve，Thinkless 无需手动调整参数，能自适应任务难度；相比路由器方法，其联合考虑输入复杂度和模型能力，决策更精准。
* **训练动态**：DeGRPO 展现出 U 型学习曲线，初期倾向长链推理保证准确性，后期随着短链响应准确性提升，逐渐增加短链模式比例，体现了合理性。
* **实验全面性与局限**：实验覆盖不同难度数据集，分析了模式崩塌和权重参数影响，设置较为全面；但预热阶段 SFT 可能导致性能略降，且数据集主要集中于数学领域，泛化性待验证。

## Further Thoughts

控制标记（Control Token）作为推理模式的‘开关’具有广泛应用潜力，可扩展至控制文本风格、情感或详细程度；DeGRPO 的解耦思想启发在复杂任务中将决策与执行分开优化，避免单一目标导致训练不稳定；此外，联合考量模型能力与任务复杂度的思路提示未来可探索更精细的模型自评估机制，让 LLM 不仅学会‘何时思考’，还能学会‘思考多深’或‘调用哪些资源’。