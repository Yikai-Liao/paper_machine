---
title: "Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures"
pubDatetime: 2025-05-14T12:39:03+00:00
slug: "2025-05-deepseek-v3-scaling"
type: "arxiv"
id: "2505.09343"
score: 0.4066137751871037
author: "grok-3-latest"
authors: ["Chenggang Zhao", "Chengqi Deng", "Chong Ruan", "Damai Dai", "Huazuo Gao", "Jiashi Li", "Liyue Zhang", "Panpan Huang", "Shangyan Zhou", "Shirong Ma", "Wenfeng Liang", "Ying He", "Yuqing Wang", "Yuxuan Liu", "Y.X. Wei"]
tags: ["LLM", "Mixture of Experts", "Low Precision", "Inference Speed", "Network Topology"]
institution: ["DeepSeek-AI", "Beijing, China"]
description: "本文通过 DeepSeek-V3 展示了硬件与模型协同设计在解决大型语言模型扩展挑战中的潜力，提出 MLA、MoE、FP8 训练、多 token 预测及多平面网络等创新方法，显著提升了训练和推理效率。"
---

> **Summary:** 本文通过 DeepSeek-V3 展示了硬件与模型协同设计在解决大型语言模型扩展挑战中的潜力，提出 MLA、MoE、FP8 训练、多 token 预测及多平面网络等创新方法，显著提升了训练和推理效率。 

> **Keywords:** LLM, Mixture of Experts, Low Precision, Inference Speed, Network Topology

**Authors:** Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Huazuo Gao, Jiashi Li, Liyue Zhang, Panpan Huang, Shangyan Zhou, Shirong Ma, Wenfeng Liang, Ying He, Yuqing Wang, Yuxuan Liu, Y.X. Wei

**Institution(s):** DeepSeek-AI, Beijing, China


## Problem Background

大型语言模型（LLMs）在快速扩展过程中面临硬件架构的重大限制，包括内存容量不足、计算效率低下以及互连带宽瓶颈等问题。
论文以 DeepSeek-V3 为例，探讨如何通过硬件与模型协同设计（co-design），在成本可控的前提下实现高效训练和推理，解决现有硬件（如 NVIDIA H800 GPU）无法完全满足 AI 工作负载需求的挑战。

## Method

*   **多头潜在注意力（Multi-head Latent Attention, MLA）**：通过将注意力机制中的 Key-Value（KV）缓存压缩为潜在向量，显著减少内存消耗，尤其适用于长上下文推理场景，相比传统方法大幅降低存储需求。
*   **专家混合（Mixture of Experts, MoE）架构**：采用稀疏计算策略，仅激活模型参数的一部分（如 DeepSeek-V3 拥有 671B 参数，但每 token 仅激活 37B），从而降低训练和推理的计算成本，同时保持高性能，特别适合个人化部署。
*   **FP8 混合精度训练**：利用低精度计算（如 FP8）减少内存占用和计算开销，通过精细量化（如 tile-wise 和 block-wise 量化）确保模型质量，同时探索自定义精度格式以优化通信。
*   **多 token 预测模块（Multi-Token Prediction Module, MTP）**：通过并行预测多个 token 并结合推测解码（speculative decoding），缓解传统自回归模型的顺序瓶颈，显著提升推理速度。
*   **多平面网络拓扑（Multi-Plane Network Topology）**：设计两层 Fat-Tree 网络结构，通过多平面设计实现成本高效的集群通信，同时优化带宽利用和故障隔离，支持大规模 GPU 集群的训练和推理。
*   **硬件感知并行策略**：根据硬件特性（如 NVLink 和 InfiniBand 带宽差异）调整并行方式，避免低效的张量并行，增强流水线并行和专家并行，确保计算与通信的高效重叠。

## Experiment

*   **内存效率提升**：通过 MLA，DeepSeek-V3 的 KV 缓存大小仅为 70.272 KB/token，相比 Qwen-2.5 72B（327.680 KB/token）和 LLaMA-3.1 405B（516.096 KB/token）有显著降低，适合长上下文任务。
*   **计算成本优化**：MoE 架构使 DeepSeek-V3 的训练成本为 250 GFLOPS/token，远低于密集模型如 LLaMA-405B（2448 GFLOPS/token），实现了性能与成本的良好平衡。
*   **推理速度改进**：MTP 模块将生成速度提升约 1.8 倍，理论上在高带宽网络（如 GB200 NVL72）下可达 1200 token/s，但实际受限于硬件和通信开销。
*   **网络性能验证**：多平面 Fat-Tree 网络在 All-to-All 通信和训练吞吐量上与单平面多轨网络表现相当（如 2048 GPU 训练时 MFU 接近 43.7%），且成本更低，实验设置覆盖小规模验证到大规模训练，较为全面，但部分理论极限未完全实证。

## Further Thoughts

硬件与模型协同设计的理念为资源受限环境下的 AI 优化提供了新思路，特别是在边缘设备上的潜在应用；MoE 架构在个人化和本地化部署中的低成本高性能特性，可能推动个性化 AI 服务的普及；此外，多平面网络拓扑和低精度通信（如 LogFMT）的探索，启发未来网络硬件可以更深度地与 AI 工作负载结合，优化通信效率。