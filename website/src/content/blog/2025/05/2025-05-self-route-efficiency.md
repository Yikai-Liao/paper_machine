---
title: "Self-Route: Automatic Mode Switching via Capability Estimation for Efficient Reasoning"
pubDatetime: 2025-05-27T03:18:31+00:00
slug: "2025-05-self-route-efficiency"
type: "arxiv"
id: "2505.20664"
score: 0.692717573372876
author: "grok-3-latest"
authors: ["Yang He", "Xiao Ding", "Bibo Cai", "Yufei Zhang", "Kai Xiong", "Zhouhao Sun", "Bing Qin", "Ting Liu"]
tags: ["LLM", "Reasoning", "Sampling", "Test Time Scaling", "Pre-Training"]
institution: ["Harbin Institute of Technology, China"]
description: "本文提出 Self-Route 框架，通过预推理和能力估计动态切换推理模式，在减少 30%-55% token 消耗的同时保持准确率损失小于 2%，为高效推理提供了通用解决方案。"
---

> **Summary:** 本文提出 Self-Route 框架，通过预推理和能力估计动态切换推理模式，在减少 30%-55% token 消耗的同时保持准确率损失小于 2%，为高效推理提供了通用解决方案。 

> **Keywords:** LLM, Reasoning, Sampling, Test Time Scaling, Pre-Training

**Authors:** Yang He, Xiao Ding, Bibo Cai, Yufei Zhang, Kai Xiong, Zhouhao Sun, Bing Qin, Ting Liu

**Institution(s):** Harbin Institute of Technology, China


## Problem Background

大型语言模型（LLMs）通过长链推理（Long Chain-of-Thought, Long CoT）在复杂任务中表现出色，但对于简单任务会导致‘过度思考’（overthinking），产生大量不必要的 token 消耗，增加计算成本和推理时间；而短链推理（Short CoT）在简单任务上高效，但在复杂任务上能力不足。
论文旨在解决如何根据任务难度和模型能力动态选择合适的推理模式，以在保证准确率的同时最大化推理效率。

## Method

*   **核心思想:** 提出 Self-Route 框架，通过轻量级的预推理（Pre-inference）阶段提取模型内部隐藏层表示（capability-aware embedding），估计模型解决当前问题的能力，并利用路由器（Router）动态选择推理模式（Short CoT 或 Long CoT），以优化效率和准确率的平衡。
*   **具体实现:**
    *   **预推理阶段:** 在正式推理前，以有限 token 预算生成简短推理轨迹，从多个隐藏层提取最终时间步的隐藏表示，形成能力感知嵌入，用于快速评估模型能力。
    *   **路由器设计:** 路由器采用线性函数，基于预推理的隐藏层向量预测一般模型（General Model）解决问题的概率；若概率高于阈值（τ），选择 Short CoT 模式，否则切换到 Long CoT 模式。
    *   **训练数据:** 构建 Gradient-10K 数据集，基于模型难度估计，包含从简单到复杂的密集难度分布问题，用于训练路由器以准确感知模型能力边界。
*   **关键点:** 预推理阶段计算开销极低，路由决策不依赖人工干预，框架适用于不同规模模型和推理范式，具有广泛适用性。

## Experiment

*   **有效性:** Self-Route 在多种模型（如 Qwen2.5-7B, Qwen2.5-32B, Qwen3-8B）和数据集（如 GSM8K, MATH500, GPQA）上，平均减少了 30%-55% 的 token 消耗，同时准确率损失小于 2%，在某些场景下甚至不到 1%。
*   **优越性:** 相比纯推理模型（Long CoT），Self-Route 在简单任务上倾向选择 Short CoT，显著降低资源使用；在复杂任务上切换到 Long CoT，确保准确率；相比一般模型（General Model），准确率有明显提升。
*   **实验设置合理性:** 实验覆盖了多种参数规模模型和不同领域数据集（数学推理、科学问答等），充分验证了方法的普适性；同时对比了 Gradient-10K 和无难度梯度数据集（GSM8K）对路由器训练的影响，证明了密集难度梯度的重要性。
*   **开销分析:** 预推理阶段 token 消耗极低，通常不到 Long CoT 的 5%，证明其为高性价比策略。

## Further Thoughts

Self-Route 的能力估计机制启发我们探索模型的‘自我认知’能力，未来可以扩展为多专家路由（Multi-Expert Routing），根据任务类型动态选择专门模型（如数学推理、代码生成专家），进一步提升效率；此外，是否可以通过非线性或注意力机制优化路由器，捕捉更复杂的能力-任务关系；Gradient-10K 的难度梯度构建方法也可能推广到其他领域（如图像处理），通过自适应生成难度梯度减少人工标注成本。