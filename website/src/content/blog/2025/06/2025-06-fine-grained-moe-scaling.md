---
title: "Scaling Fine-Grained MoE Beyond 50B Parameters: Empirical Evaluation and Practical Insights"
pubDatetime: 2025-06-03T13:55:48+00:00
slug: "2025-06-fine-grained-moe-scaling"
type: "arxiv"
id: "2506.02890"
score: 0.7902986050172035
author: "grok-3-latest"
authors: ["Jakub Krajewski", "Marcin Chochowski", "Daniel Korzekwa"]
tags: ["Large Language Models", "Mixture of Experts", "Scaling Laws", "Routing Strategy", "Training Efficiency"]
institution: ["NVIDIA", "IDEAS NCBR, University of Warsaw"]
description: "本文通过实验验证了细粒度 MoE 在大规模模型（超过50B参数）中的性能优势，提出有效训练策略，显著提升了模型质量和计算效率。"
---

> **Summary:** 本文通过实验验证了细粒度 MoE 在大规模模型（超过50B参数）中的性能优势，提出有效训练策略，显著提升了模型质量和计算效率。 

> **Keywords:** Large Language Models, Mixture of Experts, Scaling Laws, Routing Strategy, Training Efficiency

**Authors:** Jakub Krajewski, Marcin Chochowski, Daniel Korzekwa

**Institution(s):** NVIDIA, IDEAS NCBR, University of Warsaw


## Problem Background

大型语言模型（LLMs）的训练和推理对计算资源的需求极高，限制了模型规模的进一步扩展。
Mixture of Experts (MoE) 架构通过仅激活部分参数来提高效率，但标准 MoE 在收敛速度和模型质量上仍有改进空间。
本文聚焦于细粒度 MoE（Fine-Grained MoE），探索其在大规模模型（超过50B参数）下的性能表现，解决如何在保持计算效率的同时提升模型质量和训练效率的关键问题。

## Method

*   **核心思想:** 细粒度 MoE 通过增加专家数量、减小每个专家规模，并为每个 token 分配更多专家（例如从 Top-1 到 Top-8 或 Top-16），在保持总计算量不变的前提下提升模型灵活性和性能。
*   **模型架构设计:** 在 Transformer 架构中，用 MoE 层替换前馈网络层。对比了四种 MoE 变体：标准 Switch 模型（8个专家，Top-1 路由）、Mixtral 模型（8个专家，Top-2 路由）及其细粒度版本（64个专家，Top-8 或 Top-16 路由）。
*   **训练策略:** 使用 NVIDIA H100 GPU 和 Megatron-LM 框架，结合流水线并行、专家并行和张量并行技术。训练数据包括多语言文本和代码，预训练后在高质量数据集上继续训练以提升基准测试表现。优化器采用 AdamW，学习率遵循余弦调度，并设置了负载平衡损失和容量因子以缓解专家负载不平衡。
*   **路由器设计优化:** 研究了路由器中 softmax 和 Top-k 顺序的影响，发现对细粒度 MoE 而言，Top-k 后进行 softmax 能显著提升性能。
*   **评估方式:** 通过预训练验证损失和多个下游任务基准测试（如 MMLU、ARC-E）评估模型质量和效率。

## Experiment

*   **有效性:** 在 11B 参数规模下，细粒度 MoE（1xFLOPs-G8）在 Switch 变体中显著优于标准模型（1xFLOPs-G1），验证损失降低（如从 2.233 到 2.183），下游任务平均准确率提升（如从 48.4% 到 50.6%）。在 56B 参数规模下，细粒度 MoE 在 Switch 和 Mixtral 变体中均表现出更低验证损失（如 2xFLOPs-G8 为 1.757，优于 2xFLOPs-G1 的 1.780）和更高下游准确率（如平均准确率从 58.8% 提升至 60.5%）。
*   **训练时长影响:** 细粒度 MoE 的性能优势随训练数据量增加而更显著，例如在 11B 模型训练至 100B token 时，1xFLOPs-G8 性能接近 2xFLOPs-G1，但激活参数更少，计算效率更高。
*   **实验设置合理性:** 实验控制了数据集、评估协议和计算资源，确保公平对比。基准测试覆盖多个任务类型（如常识推理、阅读理解），评估全面。但未深入探讨硬件利用率（MFU）差异，可能影响实际效率结论。
*   **局限性:** 细粒度 MoE 在小规模或短时训练中优势不明显，可能因路由器初期难以有效利用多个专家。

## Further Thoughts

细粒度 MoE 的路由器在训练初期倾向于集中于少数专家，限制了早期性能，未来可探索预训练路由器或设计新初始化策略以加速学习过程；此外，增加专家数量可能对并行计算和通信开销提出更高要求，是否能设计自适应专家分配算法，根据硬件特性动态调整专家规模和数量；另外，细粒度 MoE 的专家分配是否能揭示模型对不同输入的处理机制，例如某些专家是否更擅长特定任务类型，这可能为模型可解释性研究提供新视角。