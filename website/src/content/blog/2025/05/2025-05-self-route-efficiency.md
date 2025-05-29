---
title: "Self-Route: Automatic Mode Switching via Capability Estimation for Efficient Reasoning"
pubDatetime: 2025-05-27T03:18:31+00:00
slug: "2025-05-self-route-efficiency"
type: "arxiv"
id: "2505.20664"
score: 0.692717573372876
author: "grok-3-latest"
authors: ["Yang He", "Xiao Ding", "Bibo Cai", "Yufei Zhang", "Kai Xiong", "Zhouhao Sun", "Bing Qin", "Ting Liu"]
tags: ["LLM", "Reasoning", "Efficiency", "Capability Estimation", "Routing"]
institution: ["Harbin Institute of Technology, China"]
description: "本文提出 Self-Route 框架，通过预推理和能力估计实现推理模式的自动切换，在保持高准确率的同时显著降低 30%-55% 的Token消耗，展现了广泛的实用性。"
---

> **Summary:** 本文提出 Self-Route 框架，通过预推理和能力估计实现推理模式的自动切换，在保持高准确率的同时显著降低 30%-55% 的Token消耗，展现了广泛的实用性。 

> **Keywords:** LLM, Reasoning, Efficiency, Capability Estimation, Routing

**Authors:** Yang He, Xiao Ding, Bibo Cai, Yufei Zhang, Kai Xiong, Zhouhao Sun, Bing Qin, Ting Liu

**Institution(s):** Harbin Institute of Technology, China


## Problem Background

大型语言模型（LLMs）通过推理增强（Reasoning-Augmented LLMs, RLLMs）在复杂任务上表现出色，但长链推理（Long Chain-of-Thought, Long CoT）在简单任务中会导致‘过度思考’（Overthinking），生成大量不必要的中间步骤和Token，造成计算资源浪费。
论文旨在解决如何在保证准确率的同时，动态选择合适的推理模式（短链推理 Short CoT 或长链推理 Long CoT），以优化推理效率并减少资源消耗。

## Method

*   **核心思想:** 提出 Self-Route 框架，通过轻量级预推理（Pre-Inference）阶段提取模型内部隐藏层表示作为能力感知嵌入（Capability-Aware Embedding），利用路由器（Router）根据能力估计动态选择推理模式（通用模型或推理模型），以平衡准确率和效率。
*   **具体步骤:**
    1. **预推理阶段:** 在正式推理前，模型对输入问题进行有限Token预算下的初步推理，生成短链推理轨迹，从多个Transformer层提取隐藏表示，形成能力感知嵌入，用于评估模型解决问题的能力。
    2. **路由器设计与决策:** 路由器是一个线性函数，利用预推理阶段的隐藏层向量，预测通用模型（Short CoT）解决问题的概率；若概率高于阈值，则选择通用模型，否则回退到长链推理模型（Long CoT）。
    3. **训练数据支持:** 构建基于模型难度估计的梯度数据集 *Gradient-10K*，包含从简单到复杂的密集难度分布问题，用于训练路由器，确保其能精准区分模型能力边界。
*   **特点:** 该方法不依赖外部规则或手动选择，而是利用模型内部信号实现自适应推理模式切换，同时预推理阶段的额外计算成本极低。

## Experiment

*   **有效性:** Self-Route 在多种模型（如 Qwen2.5-7B, Qwen2.5-32B, Qwen3-8B）和数据集（如 GSM8K, MATH500, GPQA Diamond）上，平均减少了 30%-55% 的Token消耗，同时准确率损失小于 2%，在某些简单任务上甚至与长链推理模型持平。
*   **合理性:** 对于简单任务，Self-Route 倾向于选择短链推理，显著降低Token消耗；对于复杂任务，则选择长链推理，确保准确率，体现了动态切换的合理性。
*   **全面性:** 实验覆盖了不同参数规模的模型和多种任务类型（数学推理、科学问答），测试了跨模型路由和单一混合模型内部模式切换的效果，验证了方法的广泛适用性。
*   **额外分析:** 预推理阶段的Token消耗极低（通常不到长链推理的5%），且 *Gradient-10K* 数据集的密集难度梯度对路由器训练至关重要，相比无梯度数据集（如 GSM8K），准确率提升最高达11%。

## Further Thoughts

Self-Route 利用模型内部隐藏层表示进行能力自评估的思路，启发我们可以在多任务学习中动态分配资源，或在对话系统中根据问题复杂度调整响应深度；此外，*Gradient-10K* 的难度梯度构建方法可应用于教育领域的自适应学习系统；未来扩展到多专家路由（Multi-Expert Routing），根据任务类型动态选择专门模型或模块，可能进一步提升效率和准确率。