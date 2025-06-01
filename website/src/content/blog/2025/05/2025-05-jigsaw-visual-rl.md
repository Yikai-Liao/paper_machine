---
title: "Jigsaw-R1: A Study of Rule-based Visual Reinforcement Learning with Jigsaw Puzzles"
pubDatetime: 2025-05-29T16:01:22+00:00
slug: "2025-05-jigsaw-visual-rl"
type: "arxiv"
id: "2505.23590"
score: 0.5141091874136124
author: "grok-3-latest"
authors: ["Zifu Wang", "Junyi Zhu", "Bo Tang", "Zhiyu Li", "Feiyu Xiong", "Jiaqian Yu", "Matthew B. Blaschko"]
tags: ["LLM", "Multimodal Learning", "Reinforcement Learning", "Visual Reasoning", "Generalization"]
institution: ["KU Leuven", "University of Science and Technology of China", "Institute for Advanced Algorithms Research, Shanghai", "Memory Tensor, Shanghai", "Samsung R&D Institute China, Beijing"]
description: "本文通过拼图任务系统研究规则驱动的视觉强化学习在多模态大语言模型中的应用，揭示其学习、泛化和推理特性，并证明 RL 在泛化能力上优于 SFT。"
---

> **Summary:** 本文通过拼图任务系统研究规则驱动的视觉强化学习在多模态大语言模型中的应用，揭示其学习、泛化和推理特性，并证明 RL 在泛化能力上优于 SFT。 

> **Keywords:** LLM, Multimodal Learning, Reinforcement Learning, Visual Reasoning, Generalization

**Authors:** Zifu Wang, Junyi Zhu, Bo Tang, Zhiyu Li, Feiyu Xiong, Jiaqian Yu, Matthew B. Blaschko

**Institution(s):** KU Leuven, University of Science and Technology of China, Institute for Advanced Algorithms Research, Shanghai, Memory Tensor, Shanghai, Samsung R&D Institute China, Beijing


## Problem Background

本文聚焦于规则驱动的强化学习（Rule-based Reinforcement Learning, RL）在多模态大语言模型（MLLMs）中的应用，特别是在视觉任务中的表现。
尽管该方法在纯文本领域（如 DeepSeek-R1）已展现出强大的推理能力和泛化能力，但在多模态场景下，由于需要整合文本和视觉信息，存在独特的挑战和潜在偏差。
论文以拼图（Jigsaw Puzzles）作为实验框架，旨在解决以下关键问题：MLLMs 在视觉任务中的表现如何？是否能通过拼图训练实现对下游任务的泛化？显式推理是否对视觉任务有益？以及 RL 与监督微调（SFT）在泛化能力上的对比。

## Method

*   **任务设计**：将输入图像分割为 m×n 的网格，随机打乱后生成拼图任务，任务类型包括‘Full’（重建整个图像，要求模型输出每个碎片的原始位置索引）和‘Pair’（判断两个碎片的相对位置，设计为多选题）。
*   **推理模式**：测试两种提示方式，即‘Thinking’（要求模型在<think>标签内输出显式推理过程，随后在<answer>标签内输出最终答案）和‘Non-thinking’（直接输出最终答案，无需显式推理）。
*   **奖励机制**：设计基于规则的奖励系统，包括准确性奖励（Accuracy Reward，根据答案正确性计算，如‘Full’任务中为正确位置比例，‘Pair’任务中为二元正确与否）和格式奖励（Format Reward，确保输出符合指定格式，如标签顺序和数量）。
*   **训练算法**：采用 GRPO 强化学习算法进行训练，设置超参数如 KL 系数（β=0.04）、采样温度（temperature=1）和 top-k 采样（k=50），训练步数根据推理模式不同为1000或2000步。
*   **核心思想**：通过规则驱动的奖励信号，引导 MLLMs 在视觉任务中学习和推理，无需依赖大量人工标注数据，同时探索显式推理对模型表现的影响。

## Experiment

*   **有效性**：未经微调的 MLLMs 在拼图任务上表现接近随机猜测（如 2x1 拼图准确率约50%），但经过微调后，模型在简单拼图上达到近完美准确率（如 Qwen2.5-VL-3B 在 2x1 拼图上的准确率接近100%），并能泛化到更复杂的未见配置（如 3x1 拼图）。
*   **泛化性**：拼图训练提升了下游视觉任务（如 CV-Bench, MMVP, SAT, Super-CLEVR）的表现，例如 Qwen2.5-VL-72B 在非推理模式下平均准确率提升至82.58%，但效果受拼图大小、问题类型和训练数据集的影响。
*   **推理模式对比**：开源模型（如 Qwen2.5-VL 系列）在直接回答（Non-thinking）模式下表现更优，而专有模型（如 GPT-4.1）在显式推理（Thinking）模式下表现更好；此外，模型在训练后可能忽略推理过程，直接输出答案，导致推理一致性下降。
*   **RL vs SFT**：RL 在泛化能力上显著优于 SFT，且 SFT 作为冷启动阶段可能削弱后续 RL 优化效果。
*   **实验设置合理性**：实验覆盖了多种模型（专有和开源）、任务类型（Full, Pair）和数据集，设置较为全面，但拼图任务的特殊性可能限制结论的普适性，论文也提出需在其他视觉任务中进一步验证。

## Further Thoughts

拼图任务作为视觉 RL 的实验框架，提供了一种无需人工标注的真值奖励信号，这种自监督特性可推广至其他模态的预训练任务（如文本、视频、音频），为低成本训练多模态模型开辟新路径；此外，规则驱动的奖励设计（准确性+格式）启发我们在其他多模态任务中设计简洁有效的奖励机制，以避免奖励欺骗问题；RL 相较于 SFT 的泛化优势也提示我们探索基于探索-利用范式的训练方法，而非单纯依赖监督信号。