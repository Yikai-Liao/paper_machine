---
title: "Edge-First Language Model Inference: Models, Metrics, and Tradeoffs"
pubDatetime: 2025-05-22T10:43:00+00:00
slug: "2025-05-edge-first-inference"
type: "arxiv"
id: "2505.16508"
score: 0.4570571479066254
author: "grok-3-latest"
authors: ["SiYoung Jang", "Roberto Morabito"]
tags: ["LLM", "Edge Computing", "Small Language Models", "Distributed AI", "Performance Metrics"]
institution: ["Nokia Bell Labs", "EURECOM"]
description: "本文通过边缘优先视角，系统评估小型语言模型（SLMs）在边缘环境中的部署可行性，提出新度量指标和优化策略，为构建高效自适应语言模型推理系统提供重要见解。"
---

> **Summary:** 本文通过边缘优先视角，系统评估小型语言模型（SLMs）在边缘环境中的部署可行性，提出新度量指标和优化策略，为构建高效自适应语言模型推理系统提供重要见解。 

> **Keywords:** LLM, Edge Computing, Small Language Models, Distributed AI, Performance Metrics

**Authors:** SiYoung Jang, Roberto Morabito

**Institution(s):** Nokia Bell Labs, EURECOM


## Problem Background

大型语言模型（LLMs）因计算需求高、延迟、成本和隐私问题，主要依赖云端部署，而边缘计算通过将任务下放到设备端或边缘集群，可以降低这些问题；
小型语言模型（SLMs）通过模型压缩技术得以在资源受限的边缘设备上运行，但性能和泛化能力不足；
论文旨在探索如何在边缘环境中利用 SLMs 减少对云端的依赖，同时保持性能、隐私和响应性。

## Method

*   **基准测试（Benchmarking）:** 对开源 SLMs 在边缘设备（如 Jetson AGX Orin 和 Jetson Nano Orin）上的性能进行详细评估，测量关键指标如令牌生成速度（Token Generation Speed, TGS）、首次令牌时间（Time-to-First-Token, TTFT）、能耗和准确率，分析模型参数规模、硬件能力和模型特化（如通用模型 vs. 专用模型）对推理效果的影响。
*   **度量指标设计:** 提出新的评估框架，定义性能-成本比（Performance-Cost Ratio, PCR）作为综合指标，结合质量（Quality, Q，如准确率）、响应性（Responsiveness, R，如 TTFT）和成本（Cost per Response, CPR，基于能耗或 API 费用），以量化边缘与云端部署的效率差异，为部署决策提供依据。
*   **模拟与策略优化:** 针对边缘环境的动态性和异构性，设计滑动窗口（Sliding Window）速率限制策略，控制资源使用并决定云端回退时机；
同时测试三种设备选择方法（Random：随机分配，Weighted：基于设备能力加权，Load-Aware：动态负载感知），以优化分布式边缘集群中的请求分配和资源利用，减少云端依赖。

## Experiment

*   **单设备性能:** 在边缘设备上，SLMs 参数规模与准确率正相关，但增益在一定规模后趋缓（如 Qwen-2.5 在 GSM8K 数据集）；专用模型（如 Qwen-2-math:1.5B）在特定任务上表现接近 7B 通用模型，但资源占用仅 19.8%；较小模型生成速度更快、能耗更低，但内存限制导致部分模型无法运行（OOM）。
*   **性能-成本权衡:** 通过 PCR 指标，边缘设备（如 Jetson Nano 上的 Qwen-2.5:3B）综合效率远超云端（如 GPT-4），成本极低（0.0017 厘/请求 vs. 1.65 厘/请求），尽管质量和响应性稍逊。
*   **分布式模拟:** 在模拟实验中，Load-Aware 设备选择策略在突发负载下表现最佳，减少云端回退令牌数和成本，同时实现更均衡的设备利用率；实验设置全面，涵盖用户数量、负载模式（稳定与突发）、令牌需求分布和设备异构性，结果显著。
*   **总体评价:** 实验表明边缘部署在成本和效率上优势明显，但性能瓶颈和云端回退的必要性也得到验证，实验设计合理且数据支持结论。

## Further Thoughts

论文提出的度量驱动部署决策（如 PCR 指标）可扩展到其他 AI 任务在边缘环境中的优化；
动态资源分配策略（如滑动窗口和 Load-Aware）展示了自适应潜力，未来可结合负载预测进一步提升效率；
异构环境适配问题启发设计自适应模型或框架，根据设备能力自动调整推理策略；
此外，边缘推理的隐私优势可结合联邦学习或差分隐私技术，进一步实现安全数据处理。