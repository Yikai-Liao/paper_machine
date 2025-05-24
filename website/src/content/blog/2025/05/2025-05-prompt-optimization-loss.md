---
title: "PMPO: Probabilistic Metric Prompt Optimization for Small and Large Language Models"
pubDatetime: 2025-05-22T06:59:10+00:00
slug: "2025-05-prompt-optimization-loss"
type: "arxiv"
id: "2505.16307"
score: 0.7053155712926437
author: "grok-3-latest"
authors: ["Chenzhuo Zhao", "Ziqian Liu", "Xingda Wang", "Junting Lu", "Chaoyi Ruan"]
tags: ["LLM", "Prompt Optimization", "Loss Evaluation", "Iterative Refinement", "Cross-Entropy"]
institution: ["Peking University", "Unaffiliated", "National University of Singapore"]
description: "PMPO 提出了一种高效统一的提示优化框架，通过 token 级别的损失评估和迭代重写显著提升语言模型性能，同时降低计算成本，适用于大小模型和多种任务。"
---

> **Summary:** PMPO 提出了一种高效统一的提示优化框架，通过 token 级别的损失评估和迭代重写显著提升语言模型性能，同时降低计算成本，适用于大小模型和多种任务。 

> **Keywords:** LLM, Prompt Optimization, Loss Evaluation, Iterative Refinement, Cross-Entropy

**Authors:** Chenzhuo Zhao, Ziqian Liu, Xingda Wang, Junting Lu, Chaoyi Ruan

**Institution(s):** Peking University, Unaffiliated, National University of Singapore


## Problem Background

提示设计是提升大型语言模型（LLM）性能的关键，尤其在微调成本高或受限的情况下，自动提示优化成为重要替代方案。
现有方法依赖输出生成、模型自省或人类反馈，导致计算成本高、适用范围窄（尤其对小型模型），且缺乏跨任务通用性。
PMPO 旨在通过轻量级、基于损失的评估机制，解决这些问题，支持大小模型和多种任务类型。

## Method

*   **核心思想:** PMPO（Probabilistic Metric Prompt Optimization）是一个基于概率度量的提示优化框架，利用交叉熵损失作为直接评估信号，避免输出生成或外部评分，降低计算成本。
*   **具体步骤:**
    *   **掩码引导的重要性评估（Mask-Guided Importance Evaluation）:** 将提示分解为语义单元，通过掩码每个单元并计算批量交叉熵损失变化（∆L），识别对性能贡献正向、负向或中性的片段。这种方法直接利用模型内部概率分布，无需生成输出。
    *   **基于损失的提示评估（Loss-Based Prompt Evaluation）:** 使用 token 级别的交叉熵损失量化提示质量，支持批量计算以提高效率；对于偏好任务，引入偏好损失，通过比较正负样本的对数概率差异进行优化。
    *   **提示变体生成（Prompt Variant Generation）:** 针对高损失样本（Hard Examples），利用语言模型生成多个提示变体，重点改进低效片段，采用温度控制的 top-p 采样确保多样性，同时通过任务描述和编辑指令指导重写。
    *   **迭代优化（Iterative Refinement）:** 通过多轮迭代，选择损失最低的提示变体作为下一轮基础，直至达到最大迭代次数（例如 20 轮），每次迭代仅需前向传播，无需复杂推理或多步生成。
*   **关键优势:** 方法轻量（仅需前向传播），通用性强（支持监督任务和偏好任务），适用于大小模型，尤其对资源受限场景友好。

## Experiment

*   **有效性:** PMPO 在多个基准数据集上表现出色：在 BBH 上平均准确率达 80.6%，超越 OPRO（77.1%）和 EvoPrompt（78.0%），在 23 个任务中 11 个排名第一；在 GSM8K 和 AQUA-RAT 上准确率分别为 94.0% 和 84.6%，领先所有基线；在 AlpacaEval 2.0 上将 Qwen2.5-14B 胜率从 31.81% 提升至 51.52%，接近 GPT-4 Omni 水平。
*   **提升显著性:** 相比依赖生成输出的方法（如 OPRO），PMPO 通过单次前向传播评估显著降低计算成本，同时在多步推理和开放式任务中性能提升明显，尤其在数学推理任务中中间步骤质量更高。
*   **实验设置合理性:** 实验覆盖多种模型规模（Qwen2.5-0.5B 到 32B）和任务类型（数学、逻辑、指令跟随），数据集划分合理（训练集上限 50 样本，测试集完整），基线方法全面（包括 CoT、OPRO 等）；跨模型泛化实验验证了提示迁移性，尽管小型模型效果稍逊。
*   **局限性:** 闭源模型因无法访问完整对数概率而应用受限；极低资源场景下可能过拟合，但中等规模数据集上表现鲁棒。

## Further Thoughts

PMPO 利用模型内部损失信号进行优化的思路令人启发，或许可以扩展到多模态模型，通过类似掩码分析识别图像或文本输入中的关键部分进行优化；此外，其对小型模型的支持提示我们，未来可以探索自适应提示复杂性策略，根据模型能力动态调整提示结构，以进一步提升跨规模适用性。