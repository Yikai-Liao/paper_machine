---
title: "Towards Building Private LLMs: Exploring Multi-Node Expert Parallelism on Apple Silicon for Mixture-of-Experts Large Language Model"
pubDatetime: 2025-06-30T09:04:25+00:00
slug: "2025-06-private-llm-apple-silicon"
type: "arxiv"
id: "2506.23635"
score: 0.5378554226231821
author: "grok-3-latest"
authors: ["Mu-Chi Chen", "Po-Hsuan Huang", "Xiangrui Ke", "Chia-Heng Tu", "Chun Jason Xue", "Shih-Hao Hung"]
tags: ["LLM", "Mixture of Experts", "Parallel Computing", "Cost Efficiency", "Inference Optimization"]
institution: ["National Taiwan University", "National Cheng Kung University", "Mohamed bin Zayed University of Artificial Intelligence"]
description: "本文通过在 Apple Silicon 集群上实现 Mixture-of-Experts 模型的高效推理，提出并验证了成本高效的私人大型语言模型系统构建方法。"
---

> **Summary:** 本文通过在 Apple Silicon 集群上实现 Mixture-of-Experts 模型的高效推理，提出并验证了成本高效的私人大型语言模型系统构建方法。 

> **Keywords:** LLM, Mixture of Experts, Parallel Computing, Cost Efficiency, Inference Optimization

**Authors:** Mu-Chi Chen, Po-Hsuan Huang, Xiangrui Ke, Chia-Heng Tu, Chun Jason Xue, Shih-Hao Hung

**Institution(s):** National Taiwan University, National Cheng Kung University, Mohamed bin Zayed University of Artificial Intelligence


## Problem Background

构建私人大型语言模型（LLM）系统以服务个人或小型团体（如 Apple Intelligence）面临高成本和可扩展性挑战。
传统高性能计算设备（如 NVIDIA H100 GPU 集群）成本高昂且维护复杂，而随着模型规模和复杂性增加，系统扩展变得困难，亟需成本高效的解决方案。

## Method

*   **核心思想:** 利用 Apple Silicon（M2 Ultra 芯片）构建 Mac Studio 集群，通过专家并行化（Expert Parallelism）加速 Mixture-of-Experts (MoE) 模型（如 DBRX）的推理，同时优化软件栈以降低成本和提升效率。
*   **硬件与集群构建:** 使用配备 M2 Ultra 芯片的 Mac Studio（最高 192GB 统一内存），通过 10Gb 以太网连接 2-4 个节点，运行未量化的 DBRX 模型（132B 参数）。
*   **专家并行化实现:** 将 MoE 模型的 16 个专家分布到多个节点上，利用专家并行技术使每个专家独立处理输入，减少推理时间。
*   **优化策略:** 针对 Apple 软件栈（如 MLX 和 Metal 框架）的内存管理开销，提出以下优化：
    *   **专家权重预堆叠（Prestacking）:** 将每个专家的权重预加载为单一数组，通过一次性预处理脚本避免重复的驱动处理开销，提升数据引用效率。
    *   **多节点负载均衡:** 包括‘忙碌全加载（Busy Full Loading）’策略（让所有专家每层都计算，即使未被选中）和‘路由辅助动态加载（Router-Aided Dynamic Loading）’策略（根据路由器选择动态分配计算任务给最近最少使用的专家），确保专家频繁被调用以避免权重被卸载。
    *   **去中心化自注意力与路由器:** 在每个节点复制自注意力、路由器和加权求和组件，减少节点间通信次数（从每层 2 次降至 1 次），通过异步 gRPC 服务器进一步减少对 GPU 计算的干扰。
*   **性能建模:** 构建性能模型，基于 GPU 加载/计算时间和通信时间（包括延迟和数据传输），预测不同节点数和网络配置下的系统性能，为私人 LLM 系统设计提供指导。
*   **关键点:** 不依赖昂贵的高端 GPU，而是利用消费级硬件和软件优化实现高效推理，同时控制通信和内存管理开销。

## Experiment

*   **有效性:** 在两节点 Mac Studio 集群上，优化后的方法（结合预堆叠、动态负载均衡和去中心化设计）将 token 生成吞吐量从 1.2 tokens/sec 提升至 6.1 tokens/sec，MoE 执行时间加速 5.2 倍，显著提升推理效率。
*   **可扩展性:** 节点数从 2 增加到 4 时，吞吐量从 6.1 提升至 7.0 tokens/sec，但通信时间占比从 23% 增至 33%，表明网络延迟是主要瓶颈。
*   **成本效率:** 与 Databricks 的 8x H100 GPU 系统相比，Mac Studio 集群在吞吐量/美元指标上高出 1.15 倍，显示出显著的成本优势（两节点系统成本仅为 H100 系统的 1/22）。
*   **实验设置合理性:** 实验覆盖了 2-4 个节点、不同优化组合，并与业界标杆对比，设置较为全面；性能模型预测与实际结果趋势一致，验证了模型有效性；但单用户负载和 128 token 输入/输出限制可能未反映多用户或长序列场景的表现。
*   **开销分析:** 主要开销来自节点间通信延迟和初始权重加载时间，优化后通信时间显著减少，但仍限制了进一步扩展。

## Further Thoughts

Apple Silicon 作为成本更低的硬件选择，展示了在消费级设备上运行超大规模模型的可能性，启发我们探索其他非传统硬件（如移动设备芯片）在分布式推理中的潜力；同时，MoE 架构的专家并行化虽有效，但通信延迟瓶颈提示未来可聚焦于优化网络协议（如 RDMA）或设计更高效并行策略；此外，性能建模的应用为系统设计提供了指导，是否可扩展至其他模型架构或硬件平台，值得进一步研究。