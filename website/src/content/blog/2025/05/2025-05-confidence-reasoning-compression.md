---
title: "ConCISE: Confidence-guided Compression in Step-by-step Efficient Reasoning"
pubDatetime: 2025-05-08T01:40:40+00:00
slug: "2025-05-confidence-reasoning-compression"
type: "arxiv"
id: "2505.04881"
score: 0.7009941709012159
author: "grok-3-latest"
authors: ["Ziqing Qiao", "Yongheng Deng", "Jiali Zeng", "Dong Wang", "Lai Wei", "Fandong Meng", "Jie Zhou", "Ju Ren", "Yaoxue Zhang"]
tags: ["LLM", "Reasoning", "Compression", "Confidence Guidance"]
institution: ["Tsinghua University", "Pattern Recognition Center, WeChat AI, Tencent Inc., China"]
description: "本文提出 ConCISE 框架，通过信心引导的推理压缩方法，显著减少大型推理模型的推理链冗余，同时保持高准确率，为高效推理提供了新途径。"
---

> **Summary:** 本文提出 ConCISE 框架，通过信心引导的推理压缩方法，显著减少大型推理模型的推理链冗余，同时保持高准确率，为高效推理提供了新途径。 

> **Keywords:** LLM, Reasoning, Compression, Confidence Guidance

**Authors:** Ziqing Qiao, Yongheng Deng, Jiali Zeng, Dong Wang, Lai Wei, Fandong Meng, Jie Zhou, Ju Ren, Yaoxue Zhang

**Institution(s):** Tsinghua University, Pattern Recognition Center, WeChat AI, Tencent Inc., China


## Problem Background

大型推理模型（Large Reasoning Models, LRMs）在复杂推理任务中通过链式思维（Chain-of-Thought, CoT）提示表现出色，但生成的推理链往往过于冗长，包含大量冗余反思内容，导致计算开销增加和用户体验下降。
现有压缩方法如事后修剪（post-hoc pruning）可能破坏推理连贯性，而基于采样的选择（sampling-based selection）无法在生成过程中有效干预。
论文从模型内部信心（confidence）的视角分析冗余反思的成因，识别出两种关键模式：信心不足（Confidence Deficit），即模型因信心低而重新考虑正确步骤；终止延迟（Termination Delay），即在得出自信答案后仍继续推理，旨在解决如何在保持准确率的同时有效压缩推理链的问题。

## Method

*   **核心思想:** 提出 ConCISE 框架（Confidence-guided Compression in Step-by-step Efficient Reasoning），通过信心引导的视角，在推理生成过程中动态增强模型信心，抑制冗余反思步骤，构建简洁推理链。
*   **具体实现:** 
    *   **信心注入（Confidence Injection）:** 在推理过程中，当检测到可能的反思步骤（即内部信心低于动态阈值）时，从预设的信心短语池（如‘Therefore’、‘Let’s proceed’等20个短语）中随机抽取一个短语，插入当前推理上下文，重新生成下一步骤，以增强模型对中间步骤的信心，减少不必要的反思行为。这一机制针对信心不足问题，确保模型在正确步骤上不过度犹豫。
    *   **提前终止（Early Stopping）:** 在生成第一个答案后，使用轻量级信心检测器监控模型内部信心。检测器基于探查提示（如‘So, I’m’）和信心相关token（如‘confident’、‘sure’）的概率计算信心分数，一旦分数超过预设阈值（0.5），即终止推理过程，避免答案后的冗余反思，解决终止延迟问题。
    *   **数据集构建与微调:** 使用 ConCISE 框架生成简洁推理数据，随后通过监督微调（Supervised Fine-Tuning, SFT）和简单偏好优化（Simple Preference Optimization, SimPO）对模型进行微调，使其学习基于信心的生成策略，避免冗余反思。
*   **关键特点:** 该方法在生成过程中实时干预，而非事后处理，确保推理连贯性；同时通过信心引导的机制，精准识别并抑制冗余步骤，而非简单删减内容。

## Experiment

*   **有效性:** 实验在 DeepSeek-R1-Distill-Qwen-7B 和 1.5B 模型上进行，ConCISE 在多个基准数据集（GSM8K, Math-500, AIME24, GPQA_diamond）上显著压缩推理链长度，SimPO 设置下压缩率达约 50%（如 Math-500 上 token 数从 3854.2 降至 1945.7），同时保持与原始模型相当的准确率（Math-500 上准确率从 90.8% 微降至 91.0%）。
*   **优越性:** 相比基线方法 OverThink（基于采样选择最短正确推理链）和 Spirit（基于困惑度逐步删除步骤），ConCISE 在压缩率和准确率之间取得更好平衡。OverThink 压缩率较低，Spirit-SimPO 虽接近 ConCISE 压缩率，但在困难数据集（如 AIME24）上准确率下降更明显（从 54.2% 降至 38.3%）。
*   **实验设置合理性:** 实验覆盖不同规模模型和多个基准数据集，包括分布内和分布外任务，验证了方法的泛化能力。消融研究表明信心注入和提前终止机制结合才能实现最佳效果，单独使用任一机制压缩效果均不如整体框架。
*   **不足与开销:** 论文指出方法未能在步骤内部进一步压缩，且信心检测在答案前阶段不够直接；额外开销主要来自信心检测器的计算和信心短语插入的重新生成，但整体开销较小，因检测器设计轻量。

## Further Thoughts

论文基于信心的视角为优化推理过程提供了新思路，启发我们可以用信心作为控制推理行为的关键因素，不仅用于压缩，还可能用于提高推理鲁棒性或自适应调整推理深度；提前终止机制提示可以在推理中动态评估模型状态，或启发资源分配策略，根据任务难度调整计算量；信心短语的随机注入表明语言模型对上下文微小变化的敏感性，这可能为提示工程或上下文操控研究开辟新方向。