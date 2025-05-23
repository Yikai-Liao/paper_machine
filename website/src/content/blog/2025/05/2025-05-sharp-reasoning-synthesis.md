---
title: "SHARP: Synthesizing High-quality Aligned Reasoning Problems for Large Reasoning Models Reinforcement Learning"
pubDatetime: 2025-05-20T09:54:42+00:00
slug: "2025-05-sharp-reasoning-synthesis"
type: "arxiv"
id: "2505.14147"
score: 0.558953210744767
author: "grok-3-latest"
authors: ["Xiong Jun Wu", "Zhenduo Zhang", "Zujie Wen", "Zhiqiang Zhang", "Wang Ren", "Lei Shi", "Cai Chen", "Deng Zhao", "Dingnan Jin", "Qing Cui", "Jun Zhou"]
tags: ["LLM", "Reasoning", "Synthetic Data", "Reinforcement Learning", "Self-Alignment"]
institution: ["AI Alignment at Ant Group", "NextEvo at Ant Group"]
description: "SHARP 方法通过自对齐策略和结构化框架生成高质量 STEM 推理问题，显著提升大型推理模型在复杂任务中的性能。"
---

> **Summary:** SHARP 方法通过自对齐策略和结构化框架生成高质量 STEM 推理问题，显著提升大型推理模型在复杂任务中的性能。 

> **Keywords:** LLM, Reasoning, Synthetic Data, Reinforcement Learning, Self-Alignment

**Authors:** Xiong Jun Wu, Zhenduo Zhang, Zujie Wen, Zhiqiang Zhang, Wang Ren, Lei Shi, Cai Chen, Deng Zhao, Dingnan Jin, Qing Cui, Jun Zhou

**Institution(s):** AI Alignment at Ant Group, NextEvo at Ant Group


## Problem Background

大型推理模型（LRMs）在 STEM 领域的复杂推理任务训练中，面临高质量、多样化且可验证问题集稀缺的挑战。
现有合成数据方法（如 Chain-of-Thought 提示）生成的推理问题往往过于简单或不可验证，限制了模型在研究生或奥林匹克级别难度任务上的进步。
论文旨在解决这一关键问题，通过生成逻辑一致、答案无歧义的高难度 STEM 问题，支持强化学习提升模型推理能力。

## Method

*   **核心思想:** 提出 SHARP（Synthesizing High-quality Aligned Reasoning Problems）方法，通过自对齐原则和结构化框架，系统生成高质量、复杂且可验证的 STEM 推理问题，用于强化学习训练大型推理模型。
*   **SHARP 策略:** 定义了一套自对齐指导原则，包括问题难度（研究生或奥赛级别）、推理一致性、答案无歧义和可验证性，确保生成内容符合高质量标准。
*   **SHARP 框架:** 包含三个阶段：
    *   **对齐阶段（Alignment）:** 设定具体约束（如难度、推理风格）和推理蓝图，确保逻辑一致性和结构化输出。
    *   **实例化阶段（Instantiation）:** 基于‘三层分类’（Three-Tier Subject Categorization）知识结构（如学科→类别→主题），通过种子主题库生成多样化、针对性的 STEM 问题，确保主题覆盖广度和深度。
    *   **推理阶段（Inference）:** 利用最先进的 LRM（如 DeepSeek R1）生成具体问题、推理过程和参考答案，随后通过验证器（如 Math-Verify）进行质量检查。
*   **SHARP 实现:** 将生成的样本用于强化学习（如 RL Zero 训练），通过可验证奖励信号（RLVR）优化模型推理能力，强调自对齐减少人工标注依赖。
*   **关键点:** 方法不依赖复杂提示工程，而是通过结构化自对齐和验证机制，确保生成问题的复杂性和可靠性，同时支持大规模数据合成。

## Experiment

*   **有效性:** SHARP 方法生成的 19 万个 STEM 问题样本显著提升模型性能。在蒸馏实验中，SHARP-Qwen2.5-7B-Instruct-Distill 模型在 GPQA Diamond 基准上得分 54.7，较基线（46.4）提升 8.3 个百分点，较 DeepSeek-R1-Distill-Qwen-7B（49.9）提升 4.8 个百分点；在 RL Zero 训练中，SHARP-Open-Reasoner-Zero-7B 得分 37.0，较基线（35.5）提升 1.5 个百分点。
*   **领域差异:** 性能提升在物理和生物科目较为明显，但在化学科目 RL Zero 训练中略有下降，论文归因于化学问题对领域知识和符号推理的更高依赖，提示无监督 RL 方法的局限性。
*   **实验设置合理性:** 实验覆盖多个 STEM 领域（物理、化学、生物），使用 GPQA 等高难度基准，设置多个对比模型（蒸馏和 RL Zero），并通过 pass rate 等指标分析数据难度分布和模型能力差异，设置较为全面。但缺乏统计显著性分析（如误差条或置信区间），可能影响结果稳健性。
*   **数据质量:** 附录中对 SHARP 生成数据的分布、难度和主题覆盖进行了详尽分析，表明其难度分布与真实数据集接近，且通过三层分类结构确保了多样性。

## Further Thoughts

SHARP 的自对齐原则和三层分类结构提供了减少人工标注依赖、生成高质量训练数据的思路，这种方法可扩展到其他领域（如代码生成或法律推理），通过结构化知识框架确保数据多样性和深度覆盖；此外，可验证奖励信号（RLVR）在强化学习中的应用为高精度任务（如教育或医疗诊断）提供了新视角，值得进一步探索如何设计领域特定的验证机制以提升模型可靠性。