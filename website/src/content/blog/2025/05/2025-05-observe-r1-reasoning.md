---
title: "Observe-R1: Unlocking Reasoning Abilities of MLLMs with Dynamic Progressive Reinforcement Learning"
pubDatetime: 2025-05-18T14:08:03+00:00
slug: "2025-05-observe-r1-reasoning"
type: "arxiv"
id: "2505.12432"
score: 0.6440211401864484
author: "grok-3-latest"
authors: ["Zirun Guo", "Minjie Hong", "Tao Jin"]
tags: ["LLM", "Multimodal Learning", "Reinforcement Learning", "Reasoning", "Progressive Learning"]
institution: ["Zhejiang University"]
description: "本文提出 Observe-R1 框架，通过渐进学习、多模态格式约束、额外奖励和动态加权机制，显著提升多模态大语言模型的推理能力，并在推理与通用任务上取得优异表现。"
---

> **Summary:** 本文提出 Observe-R1 框架，通过渐进学习、多模态格式约束、额外奖励和动态加权机制，显著提升多模态大语言模型的推理能力，并在推理与通用任务上取得优异表现。 

> **Keywords:** LLM, Multimodal Learning, Reinforcement Learning, Reasoning, Progressive Learning

**Authors:** Zirun Guo, Minjie Hong, Tao Jin

**Institution(s):** Zhejiang University


## Problem Background

多模态大语言模型（MLLMs）在处理文本和图像等多模态信息时，推理能力提升面临挑战，尤其是在如何适配强化学习（RL）以应对多模态数据的复杂性方面研究不足。
作者从人类学习模式（从简单到复杂、从易到难的渐进学习）中汲取灵感，试图解决 MLLM 在推理任务中结构化和准确处理多模态信息的问题，同时探索单模态与多模态任务推理过程的差异。

## Method

*   **核心思想:** 提出 Observe-R1 框架，通过强化学习（RL）提升 MLLM 的推理能力，针对多模态任务特性设计渐进学习和结构化输出机制。
*   **具体实现:**
    *   **NeuraLadder 数据集构建:** 基于问题难度（正确率倒数）和复杂性（回答长度）组织多模态数据集，采用平滑采样策略，使模型从易到难逐步学习，模拟人类学习过程。
    *   **多模态格式约束:** 通过系统提示词（如 `<observe></observe>`、`<think></think>`、`<answer></answer>`）引导模型先观察图像，再逐步推理，最后输出答案，提升视觉信息提取能力和推理结构化程度。
    *   **额外奖励机制:** 在奖励函数中加入额外奖励项，鼓励简洁且正确的回答，通过长度约束避免冗长推理，优化推理质量。
    *   **动态加权与采样:** 基于模型对样本的不确定性动态调整训练权重，优先关注中等难度问题，过滤全对或全错样本以提高训练稳定性，并随训练进程动态调整难度关注点。
*   **技术基础:** 基于 GRPO（Group Relative Policy Optimization）算法进行优化，结合规则奖励（如准确性奖励和格式奖励）进行训练。
*   **关键点:** 方法不改变模型架构，仅通过数据组织、提示设计和奖励机制优化训练过程，适配多模态任务特性。

## Experiment

*   **有效性:** 在 Qwen2.5-VL-3B 和 7B 模型上，使用 NeuraLadder 数据集的 20k 样本训练后，Observe-R1-3B 在推理基准（如 MathVista 得分 64.9）上显著超越更大规模模型（7-11B 参数）和部分闭源模型，同时在通用任务（如 MMMU、MMBench）上保持强劲性能。
*   **推理质量:** 通过案例对比，Observe-R1 的推理过程更结构化、清晰且简洁，尤其在视觉信息提取和推理一致性上优于基线（如 GRPO）。
*   **消融研究:** 验证了各组件效果，NeuraLadder 数据集提升 MathVista 3.2%，多模态格式提高训练效率和推理清晰度，额外奖励优化简洁性，动态加权提升数据利用率。
*   **实验设置合理性:** 涵盖多个推理和通用基准数据集，模型规模（3B 和 7B）和数据量（20k）选择合理，实验设计全面，但受限于计算资源，未在更大模型上进一步验证。
*   **局限性:** 论文指出资源限制导致无法探索更大模型或更复杂加权函数，但现有结果已展现显著提升和泛化能力。

## Further Thoughts

渐进学习范式（Progressive Learning）可推广至其他 AI 任务，如代码生成或多步决策，是否能通过动态任务难度调整模拟人类‘螺旋式上升’学习过程？
多模态格式约束的成功提示结构化输出对推理能力的重要性，未来是否能针对不同模态（如音频、视频）设计定制化引导格式？
动态加权机制启发对样本‘学习价值’的精准量化，是否可结合主动学习或元学习进一步优化训练效率？
额外奖励机制虽有效，但存在追求简洁而忽略推理完整性的风险，是否能通过多目标优化平衡简洁性和全面性？