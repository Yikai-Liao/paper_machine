---
title: "VeOmni: Scaling Any Modality Model Training with Model-Centric Distributed Recipe Zoo"
pubDatetime: 2025-08-04T11:33:04+00:00
slug: "2025-08-veomni-omni-modal-training"
type: "arxiv"
id: "2508.02317"
score: 0.6647389858454567
author: "grok-3-latest"
authors: ["Qianli Ma", "Yaowei Zheng", "Zhelun Shi", "Zhongkai Zhao", "Bin Jia", "Ziyue Huang", "Zhiqi Lin", "Youjie Li", "Jiacheng Yang", "Yanghua Peng", "Zhi Zhang", "Xin Liu"]
tags: ["LLM", "Multi-Modal", "Distributed Training", "Parallelism", "Scalability"]
institution: ["ByteDance Seed"]
description: "VeOmni 提出一个模型中心化的分布式训练框架，通过解耦模型定义与并行逻辑，支持全模态大语言模型的高效、可扩展训练，显著提升吞吐量和长上下文处理能力。"
---

> **Summary:** VeOmni 提出一个模型中心化的分布式训练框架，通过解耦模型定义与并行逻辑，支持全模态大语言模型的高效、可扩展训练，显著提升吞吐量和长上下文处理能力。 

> **Keywords:** LLM, Multi-Modal, Distributed Training, Parallelism, Scalability

**Authors:** Qianli Ma, Yaowei Zheng, Zhelun Shi, Zhongkai Zhao, Bin Jia, Ziyue Huang, Zhiqi Lin, Youjie Li, Jiacheng Yang, Yanghua Peng, Zhi Zhang, Xin Liu

**Institution(s):** ByteDance Seed


## Problem Background

随着大型语言模型（LLMs）从单模态向全模态（omni-modal）理解与生成演进，模型架构变得高度异构，包含多种模态特定网络，导致传统训练框架在扩展到端到端‘任意到任意’（any-to-any）场景时面临负载不均衡、通信与计算耦合、以及高工程成本等问题，缺乏针对全模态 LLMs 的可扩展训练基础设施。

## Method

* **核心思想**：提出 VeOmni，一个模块化且高效的训练框架，通过模型中心化的分布式训练策略（model-centric distributed recipes），解耦模型定义与并行逻辑，支持全模态 LLMs 的大规模训练。
* **模块化设计**：采用编码器-基础模型-解码器架构，提供轻量级配置接口，支持以‘即插即用’方式集成新模态（如图像、音频、视频），只需少量代码修改即可扩展。
* **分布式训练策略**：
  * **完全分片数据并行（FSDP）**：通过在所有设备上分片参数、梯度和优化器状态，显著降低单 GPU 内存需求，支持超大模型训练，且不侵入模型代码。
  * **序列并行（SP）**：基于 DeepSpeed Ulysses，沿序列维度分割激活值，通过 all-to-all 通信优化长上下文训练（如 160K 令牌），并支持与 FlashAttention 集成以提升效率。
  * **专家并行（EP）**：针对混合专家（MoE）模型，通过跨设备分片专家参数实现高效扩展，并采用通信-计算重叠技术减少路由延迟。
  * **多维并行组合**：支持 2D（如 FSDP+SP）和 3D（如 FSDP+SP+EP）并行策略的灵活组合，适配不同模态组件需求。
* **系统优化**：包括动态批处理减少填充开销、高效内核（如 FlashAttention、Liger-kernel）提升计算速度、内存优化（如层级重计算、激活卸载）支持更大批次，以及高效分布式检查点保存等。
* **关键创新**：通过高层次 API（如 parallel plan）抽象并行逻辑，用户无需管理底层分布式细节即可定制训练策略，确保框架对异构架构的高度适应性。

## Experiment

* **训练效率**：VeOmni 在 8 到 128 个 GPU 上训练 7B 到 72B 参数模型，吞吐量最高达 2800+ 令牌/秒/GPU（30B MoE 模型，128 GPU），内存利用率（MFU）最高达 61.5%（7B 模型，192K 令牌），显示出优异的可扩展性。
* **长上下文支持**：通过 FSDP+SP 组合，支持 Qwen2-VL 7B 模型训练至 256K 令牌，72B 模型至 96K 令牌；3D 并行（FSDP+SP+EP）支持 30B MoE 模型至 160K 令牌，显著优于传统框架。
* **与基线对比**：与 TorchTitan 对比，VeOmni 在内存效率和吞吐量上均有提升，尤其在长上下文和高参数规模下避免了内存溢出（OOM）问题，表现出更强的稳定性。
* **训练稳定性**：在多模态理解与生成任务上，三个全模态模型（Janus, LLaMA#Omni, Qwen3-MoE#Omni）均展现稳定损失收敛，验证了框架的训练可靠性。
* **实验设置评价**：实验覆盖多种模型架构（密集与 MoE）、模态类型（文本、图像、视频、音频）和上下文长度，设置全面合理；但缺乏对训练总成本和更大规模集群（>128 GPU）的测试，可能需进一步验证框架上限。

## Further Thoughts

VeOmni 的模型-系统解耦理念启发我们思考是否可以将‘模态感知’并行策略引入框架，根据不同模态的计算特性（如图像的高计算需求 vs 文本的序列依赖）动态分配并行资源；此外，轻量级模态定制接口的‘即插即用’设计提示未来可能实现模型组件的在线‘热插拔’，支持实时多模态应用中的动态模态切换。