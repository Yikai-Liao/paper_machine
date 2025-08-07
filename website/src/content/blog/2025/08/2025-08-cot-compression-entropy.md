---
title: "Compressing Chain-of-Thought in LLMs via Step Entropy"
pubDatetime: 2025-08-05T11:48:18+00:00
slug: "2025-08-cot-compression-entropy"
type: "arxiv"
id: "2508.03346"
score: 0.7613305640667151
author: "grok-3-latest"
authors: ["Zeju Li", "Jianyuan Zhong", "Ziyang Zheng", "Xiangyu Wen", "Zhijian Xu", "Yingying Cheng", "Fan Zhang", "Qiang Xu"]
tags: ["LLM", "Chain of Thought", "Compression", "Reasoning", "Training Strategy"]
institution: ["The Chinese University of Hong Kong", "Huawei Technologies Co., Ltd"]
description: "本文提出基于步骤熵的链式思维压缩框架，通过识别并剪枝低熵冗余步骤及两阶段训练策略，使大型语言模型在推理时显著减少 token 使用量（16%-57%）而维持准确率。"
---

> **Summary:** 本文提出基于步骤熵的链式思维压缩框架，通过识别并剪枝低熵冗余步骤及两阶段训练策略，使大型语言模型在推理时显著减少 token 使用量（16%-57%）而维持准确率。 

> **Keywords:** LLM, Chain of Thought, Compression, Reasoning, Training Strategy

**Authors:** Zeju Li, Jianyuan Zhong, Ziyang Zheng, Xiangyu Wen, Zhijian Xu, Yingying Cheng, Fan Zhang, Qiang Xu

**Institution(s):** The Chinese University of Hong Kong, Huawei Technologies Co., Ltd


## Problem Background

大型语言模型（LLMs）在使用链式思维（Chain-of-Thought, CoT）提示时，虽然在复杂推理任务中表现出色，但生成的推理过程往往冗长且充满冗余，导致推理成本高、效率低下，尤其在模型规模增大和大规模部署时成为显著瓶颈。
作者旨在解决这一问题，通过识别并移除推理链中的冗余步骤，在不牺牲准确性的前提下提升推理效率。

## Method

*   **核心思想:** 提出‘步骤熵’（Step Entropy）作为度量推理步骤信息贡献的指标，假设低熵步骤（生成时确定性高）是冗余的，可以安全移除，从而压缩链式思维过程。
*   **步骤熵定义:** 基于信息论中的香农熵，计算每个推理步骤中所有 token 的熵值总和，高熵表示生成不确定性大（信息贡献高），低熵表示确定性强（可能冗余）。
*   **低熵步骤剪枝策略:** 在生成完整 CoT 后，计算每个步骤的熵值，按熵值升序排列，移除占比为 κ（例如 80%）的低熵步骤，并用 [SKIP] 标记替代，保留高熵步骤以维持推理核心结构。
*   **两阶段训练策略:** 
    *   **监督微调（SFT）:** 使用基于熵值压缩的训练数据（问题-压缩 CoT 对），训练模型学习预测何时使用 [SKIP] 标记，减少冗余步骤生成。
    *   **组相对策略优化（GRPO）:** 基于强化学习进一步优化，通过设计复合奖励函数（包括正确性、跳跃比例、跳跃数量惩罚、响应长度惩罚），使模型在推理时自主平衡准确性和效率，动态决定是否跳跃步骤。
*   **关键特点:** 方法不依赖特定模型架构，剪枝以步骤为语义单位（而非 token），确保推理结构的完整性，同时通过训练实现自主压缩，减少人工干预。

## Experiment

*   **有效性:** 实验表明，剪枝高达 80% 的低熵步骤后，模型准确率几乎不受影响（例如 DeepSeek-R1-7B 在 GSM8k 上从 80.36% 微升至 80.82%），而 token 使用量显著减少（16.2%-57% 跨多个基准数据集）。
*   **对比分析:** 相比高熵步骤剪枝或随机剪枝，低熵步骤剪枝在高比例剪枝（80%）下仍维持性能，而其他策略在低比例（40%）时已导致显著性能下降，验证了低熵步骤的冗余性。
*   **训练效果:** 两阶段训练（SFT+GRPO）进一步提升效率，token 减少比例达 35%-57%，准确率保持稳定甚至略有提升（例如 GSM8k 上从 78.54% 升至 79.15%），表明模型学会了自主跳跃冗余步骤。
*   **实验设置合理性:** 实验覆盖多个模型（DeepSeek-R1-7B/14B, Qwen3-8B）和数学推理基准（GSM8k, Math500, AIME 2024/2025），并扩展至 MMLU 任务，验证了方法的普适性；同时对比了 token 级别剪枝，证明步骤级别剪枝更有效。
*   **局限性:** 80% 剪枝阈值可能不适用于所有架构或领域，跨领域实验数据量较少，结论稳健性待进一步验证。

## Further Thoughts

步骤熵的概念揭示了 LLMs 推理过程中的大量冗余，启发我们思考是否可以通过改进预训练或后训练范式，从源头上减少冗余生成；此外，跨领域压缩效果的差异提示可以设计自适应压缩策略，根据任务类型或模型架构动态调整剪枝阈值和奖励函数权重，以进一步优化效率-准确性平衡。