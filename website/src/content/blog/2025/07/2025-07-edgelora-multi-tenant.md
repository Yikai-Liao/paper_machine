---
title: "EdgeLoRA: An Efficient Multi-Tenant LLM Serving System on Edge Devices"
pubDatetime: 2025-07-02T07:47:28+00:00
slug: "2025-07-edgelora-multi-tenant"
type: "arxiv"
id: "2507.01438"
score: 0.5081545908627444
author: "grok-3-latest"
authors: ["Zheyu Shen", "Yexiao He", "Ziyao Wang", "Yuning Zhang", "Guoheng Sun", "Wanghao Ye", "Ang Li"]
tags: ["LLM", "Edge Computing", "Parameter Efficient Tuning", "Batch Processing", "Memory Management"]
institution: ["University of Maryland, College Park"]
description: "EdgeLoRA 通过自适应适配器选择、异构内存管理和批量 LoRA 推理，显著提升了边缘设备上多租户大型语言模型服务的吞吐量和可扩展性。"
---

> **Summary:** EdgeLoRA 通过自适应适配器选择、异构内存管理和批量 LoRA 推理，显著提升了边缘设备上多租户大型语言模型服务的吞吐量和可扩展性。 

> **Keywords:** LLM, Edge Computing, Parameter Efficient Tuning, Batch Processing, Memory Management

**Authors:** Zheyu Shen, Yexiao He, Ziyao Wang, Yuning Zhang, Guoheng Sun, Wanghao Ye, Ang Li

**Institution(s):** University of Maryland, College Park


## Problem Background

大型语言模型（LLMs）在边缘设备上的部署因其低延迟、高隐私性和个性化响应的优势而备受关注，但边缘设备资源受限（如内存和计算能力有限），且多租户场景下需要同时服务多个任务或用户，面临适配器选择复杂、内存开销大以及并发请求处理效率低等关键挑战。

## Method

*   **核心思想:** 设计 EdgeLoRA 系统，通过系统级优化在资源受限的边缘设备上高效服务多个 LoRA 适配器，同时适应多租户场景的动态工作负载。
*   **自适应适配器选择 (Adaptive Adapter Selection):** 引入一个基于性能评估的适配器路由器，利用多标签分类器根据用户输入提示和内存缓存中的适配器可用性，动态选择最适合的 LoRA 适配器，避免手动选择的复杂性和错误。
*   **异构内存管理 (Heterogeneous Memory Management):** 结合 LRU（Least Recently Used）缓存策略和预分配内存池，优化适配器在内存和磁盘之间的交换，减少运行时内存分配开销，提高缓存命中率，确保在资源受限环境下的低延迟访问。
*   **批量 LoRA 推理 (Batch LoRA Inference):** 提出一种批量处理方法，将不同适配器的请求整合到一个批次中，利用 GPU 并行性同时处理基础模型和适配器特定计算，通过分组相同适配器的请求进一步优化计算效率，显著提高吞吐量和资源利用率。

## Experiment

*   **有效性:** 在多个边缘设备（Jetson AGX Orin, Jetson Orin Nano, Raspberry Pi 5）上测试，EdgeLoRA 相较于基线系统 llama.cpp 实现了 2-4 倍的吞吐量提升，同时支持的适配器数量从几十个增加到上千个，性能提升显著。
*   **合理性:** 实验覆盖了多种模型（如 Llama3.1-8B, Llama3.2-3B）、量化配置（如 Q8_0, Q4_0）和设备类型，并通过合成工作负载模拟现实场景中的长尾分布和高突发性请求，设置全面且贴近实际应用。
*   **局限性与开销:** 自适应适配器选择引入了少量计算开销（相当于解码输入提示的时间），但对服务水平目标（SLO）达成影响较小；内存管理和批量推理的开销主要在初始加载和适配器交换上，通过缓存和预分配策略得到有效控制。
*   **鲁棒性:** 在高适配器局部性和高工作负载偏态场景下，EdgeLoRA 仍保持稳定性能，且能耗低于基线，显示出良好的适应性和效率。

## Further Thoughts

EdgeLoRA 的自适应适配器选择机制启发了我，是否可以将智能路由思想扩展到多模态模型或联邦学习场景中，动态分配任务和资源；批量 LoRA 推理的并行优化策略也让我思考，如何在其他资源受限环境中（如物联网设备）应用类似方法处理异构任务；此外，异构内存管理策略可能为边缘设备上的多组件系统（如多模态模型的模块切换）提供新的资源优化思路。