---
title: "Certainty-Guided Reasoning in Large Language Models: A Dynamic Thinking Budget Approach"
pubDatetime: 2025-09-09T14:57:15+00:00
slug: "2025-09-certainty-guided-reasoning"
type: "arxiv"
id: "2509.07820"
score: 0.6291572726106672
author: "grok-3-latest"
authors: ["João Paulo Nogueira", "Alonso Silva", "Wentao Sun", "Laith Zumot"]
tags: ["LLM", "Reasoning", "Sampling", "Test Time Scaling", "Post-Training"]
institution: ["Institut Polytechnique de Paris", "Nokia Bell Labs", "École Polytechnique", "Nokia"]
description: "本文提出确定性引导推理（CGR）方法，通过动态评估模型确定性调整推理预算，在保持准确性的同时显著降低计算成本，提升大型推理语言模型的效率和可靠性。"
---

> **Summary:** 本文提出确定性引导推理（CGR）方法，通过动态评估模型确定性调整推理预算，在保持准确性的同时显著降低计算成本，提升大型推理语言模型的效率和可靠性。 

> **Keywords:** LLM, Reasoning, Sampling, Test Time Scaling, Post-Training

**Authors:** João Paulo Nogueira, Alonso Silva, Wentao Sun, Laith Zumot

**Institution(s):** Institut Polytechnique de Paris, Nokia Bell Labs, École Polytechnique, Nokia


## Problem Background

大型推理语言模型（LRLMs）在复杂多步推理任务中表现出色，但固定思考预算（thinking budget，即分配的推理 token 数量）存在局限：预算不足可能导致结论错误，预算过多则浪费计算资源。
关键问题是如何动态调整推理过程，使模型在准确性和效率之间取得平衡，避免过早停止或过度推理。

## Method

*   **核心思想：** 提出确定性引导推理（Certainty-Guided Reasoning, CGR），通过模型自身的确定性（certainty）作为信号，动态决定是否停止推理，优化资源分配。
*   **确定性估计：** 基于模型在每个 token 预测时的 softmax 概率分布，计算答案中最低概率 token 的值作为确定性指标，确保评估的保守性。
*   **定期探查机制：** 在推理过程中，每隔固定 token 数（如 1000 个）检查当前答案的确定性，若达到预设阈值（如 0.97），则提前终止推理，节省计算资源。
*   **预算强制（Budget Forcing）：** 若模型试图过早结束推理（如输出结束标记），但确定性未达标，则插入‘等待’指令，强制模型继续推理以提升答案质量。
*   **早停与动态调整：** 结合早停机制和预算强制，形成自适应推理框架，允许模型在高确定性时节省资源，在不确定时深入探索。
*   **灵活实现：** 确定性探查既可由推理模型自身完成，也可引入独立的小型模型进行评估，适应不同计算约束场景。

## Experiment

*   **准确性表现：** 在 AIME2024 数据集上，CGR 使 DeepSeek 模型准确率从 15/30 微升至 16/30；在 AIME2025 上基本持平（从 13/30 到 12/30），但显著减少了高置信错误答案。
*   **效率提升：** CGR 大幅降低 token 使用量，在 64 个种子测试中，确定性阈值为 0.96 时总共节省超过 330 万 token，平均每问题节省约 1760 个 token。
*   **稳定性验证：** 多种子（64 个）测试显示 CGR 性能方差较低，证明其对随机初始化的鲁棒性，Grade 指标（考虑错误惩罚）在有惩罚场景下也表现更优。
*   **实验设置合理性：** 实验覆盖多个模型（DeepSeek、Qwen、Phi4）、两个数据集（AIME2024/2025）及多阈值测试，设计全面；但准确性提升有限，可能与任务难度或阈值选择有关。

## Further Thoughts

确定性作为推理充分性信号的理念非常具有启发性，不仅限于数学推理，还可扩展至代码生成或开放域问答等领域；此外，使用小型模型进行确定性探查的思路让我思考是否可以通过异构模型协作（大型模型推理，小型模型评估）进一步优化效率；另外，token 节省与问题难度的关联性分析也启发了我，是否可以基于此设计自适应计算分配策略，甚至用于模型训练的课程设计，针对性提升模型在困难任务上的表现？