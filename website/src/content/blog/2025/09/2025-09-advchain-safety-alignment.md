---
title: "AdvChain: Adversarial Chain-of-Thought Tuning for Robust Safety Alignment of Large Reasoning Models"
pubDatetime: 2025-09-29T04:27:23+00:00
slug: "2025-09-advchain-safety-alignment"
type: "arxiv"
id: "2509.24269"
score: 0.5864985543484661
author: "grok-3-latest"
authors: ["Zihao Zhu", "Xinyu Wu", "Gehan Hu", "Siwei Lyu", "Ke Xu", "Baoyuan Wu"]
tags: ["LLM", "Reasoning", "Safety Alignment", "Chain of Thought", "Adversarial Tuning"]
institution: ["The Chinese University of Hong Kong, Shenzhen", "State University of New York at Buffalo", "Huawei International, Singapore"]
description: "本文提出 AdvChain 框架，通过对抗性链式推理调优训练模型动态自我纠正，显著提升大型推理模型对攻击的鲁棒性和对良性请求的实用性，同时保持推理能力。"
---

> **Summary:** 本文提出 AdvChain 框架，通过对抗性链式推理调优训练模型动态自我纠正，显著提升大型推理模型对攻击的鲁棒性和对良性请求的实用性，同时保持推理能力。 

> **Keywords:** LLM, Reasoning, Safety Alignment, Chain of Thought, Adversarial Tuning

**Authors:** Zihao Zhu, Xinyu Wu, Gehan Hu, Siwei Lyu, Ke Xu, Baoyuan Wu

**Institution(s):** The Chinese University of Hong Kong, Shenzhen, State University of New York at Buffalo, Huawei International, Singapore


## Problem Background

大型推理模型（Large Reasoning Models, LRMs）通过链式推理（Chain-of-Thought, CoT）在复杂问题解决中表现出色，但其多步骤推理特性引入了新的安全挑战。
现有安全对齐方法（如 Safety CoT Tuning）存在‘雪球效应’（Snowball Effect）失败模式，即推理过程中的微小偏差会逐步放大，导致对有害请求的逐步有害顺从或对良性请求的过度拒绝，根源在于模型仅学会模仿完美推理脚本而缺乏自我纠正能力。

## Method

*   **核心思想:** 提出 AdvChain，一种对抗性链式推理调优（Adversarial CoT Tuning）范式，旨在通过训练模型动态自我纠正来打破‘雪球效应’中的认知惯性，而非仅仅模仿无错误的推理路径。
*   **数据集构建:** 程序化改写现有推理链，生成两种对抗性样本：
    *   **Temptation-Correction（诱惑-纠正）样本**：模拟模型对有害请求的‘诱惑’推理偏差，并注入纠正步骤，教模型从有害路径中恢复。
    *   **Hesitation-Correction（犹豫-纠正）样本**：模拟对良性请求的不必要谨慎，并注入纠正步骤，教模型克服过度拒绝倾向。
    具体生成过程包括：基于有害或良性提示生成基础推理路径，选择逻辑插入点注入错误推理步骤（诱惑或犹豫），随后添加纠正步骤，最终整合为完整的对抗性推理链。
*   **模型调优:** 使用构建的数据集对模型进行微调，优化标准自回归目标函数，使模型内化错误识别与纠正机制，增强其在推理过程中的鲁棒性和适应性。
*   **创新点:** 通过‘对抗性’训练（故意引入错误并要求纠正），区别于传统对齐方法，AdvChain 直接针对推理过程中的偏差放大问题，提供了一种更主动的安全对齐策略。

## Experiment

*   **安全性能与鲁棒性:** 在多个安全基准（如 HarmBench, WildJailbreak）上，AdvChain 显著降低了攻击成功率（ASR），例如在 DeepSeek-R1-7B 上 ASR 从 51% 降至 4.5%（HarmBench），与使用 15 倍数据量的 RealSafe-R1 性能相当，显示出高数据效率；在对抗性 CoT 劫持攻击测试中，ASR 降至 9.33%，远低于基线模型，表明推理稳定性更强。
*   **过度拒绝问题:** 在良性请求基准（如 XSTest）上，AdvChain 的过度拒绝率（ORR）显著降低，例如在 DeepSeek-R1-7B 上 ORR 从 STAR-1 的 42% 降至 18%，改善了安全与实用性的权衡。
*   **推理能力保持:** 在数学和编码任务（如 Math-500, LiveCodeBench）上，AdvChain 的 Pass@1 得分与基础模型相当（如 DeepSeek-R1-7B 从 92.8% 微升至 93.4%），表明安全对齐未损害核心能力。
*   **实验设置合理性:** 实验覆盖多种模型规模（DeepSeek-R1 和 Qwen3 系列）和攻击场景，与相同数据量（1k）和更大数据量（15k）的基线对比，评估维度全面（安全、鲁棒性、过度拒绝、推理能力），数据支持结论可信。

## Further Thoughts

AdvChain 的‘对抗性调优’理念启发我们可以通过故意引入错误并训练纠正来增强模型鲁棒性，这种思路可推广至提升模型泛化能力或多轮对话中的错误恢复；此外，Temptation-Correction 和 Hesitation-Correction 样本的设计提示我们可以通过针对性数据构造解决特定认知偏差，未来或许可以探索动态生成对抗性样本（例如通过在线学习或强化学习），以适应不断演变的攻击模式和复杂场景。