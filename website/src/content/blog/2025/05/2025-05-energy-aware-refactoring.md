---
title: "Leveraging LLMs to Automate Energy-Aware Refactoring of Parallel Scientific Codes"
pubDatetime: 2025-05-04T17:05:34+00:00
slug: "2025-05-energy-aware-refactoring"
type: "arxiv"
id: "2505.02184"
score: 0.5344693133641912
author: "grok-3-latest"
authors: ["Matthew T. Dearing", "Yiheng Tao", "Xingfu Wu", "Zhiling Lan", "Valerie Taylor"]
tags: ["LLM", "Code Generation", "Energy Efficiency", "Parallel Computing", "Optimization"]
institution: ["University of Illinois Chicago", "Argonne National Laboratory"]
description: "本文提出 LASSI-EE 框架，利用大型语言模型通过多阶段、自我校正的管道自动化重构并行科学代码，在 NVIDIA A100 GPU 上实现平均 47% 的能量节省，展示了 LLM 在能效优化中的潜力。"
---

> **Summary:** 本文提出 LASSI-EE 框架，利用大型语言模型通过多阶段、自我校正的管道自动化重构并行科学代码，在 NVIDIA A100 GPU 上实现平均 47% 的能量节省，展示了 LLM 在能效优化中的潜力。 

> **Keywords:** LLM, Code Generation, Energy Efficiency, Parallel Computing, Optimization

**Authors:** Matthew T. Dearing, Yiheng Tao, Xingfu Wu, Zhiling Lan, Valerie Taylor

**Institution(s):** University of Illinois Chicago, Argonne National Laboratory


## Problem Background

高性能计算（HPC）系统对能源的需求日益增加，现有 exascale 系统功耗高达 24-39 MW，对基础设施和环境造成显著压力。
当前大型语言模型（LLMs）在代码生成中主要关注功能正确性，忽视了并行科学代码的性能和能效问题，因此需要一种自动化方法来优化代码能耗。

## Method

*   **核心思想:** 提出 LASSI-EE，一个基于 LLM 的自动化、自我校正的代码重构框架，通过多阶段管道优化并行科学代码的能效，同时保证功能正确性。
*   **具体流程:** 
    *   **Stage 0 - 代码分析:** 对输入代码在目标系统上进行性能和功耗基准测试，建立比较基线。
    *   **Stage 1 - 基准重构:** 使用零样本提示（Zero-Shot Prompting）让 LLM 初步生成优化代码，并通过自我校正模块修复编译和运行错误。
    *   **Stage 2 - 上下文构建与计划:** 为 LLM 提供丰富的上下文（如 CUDA 优化指南、硬件配置信息），让其总结背景知识并制定详细的重构计划，识别代码中的能效瓶颈。
    *   **Stage 3 - 迭代重构:** 基于重构计划，LLM 迭代生成优化代码，结合自我提示和反馈机制，通过‘LLM-as-a-Judge’代理验证功能正确性，并根据能耗指标选择最佳版本。
    *   **Stage 4 - 终止与比较:** 输出最终优化代码，并与原始代码进行详细比较以供参考。
*   **关键创新:** 利用 LLM 的语义理解能力，通过上下文增强、自我校正和迭代优化，逐步逼近能效最优解，同时设计了硬件感知的提示策略以适配目标平台。

## Experiment

*   **有效性:** 在 NVIDIA A100 GPU 上测试 20 个 HeCBench 基准代码，LASSI-EE 在 85% 的代码中实现了能效提升，平均能量减少 47%，平均功耗降低 31.34%，运行时间缩短 51.93%。
*   **结果分布:** 节能幅度从 2.35% 到 99.56% 不等，部分极端优化（如 bsearch）可能存在功能偏差，显示出方法的适用性广但稳定性有待提升。
*   **实验设置合理性:** 基准代码覆盖多种科学计算类别（如生物信息学、机器学习），输入参数调整以确保运行时间足够用于功耗测量；使用 pyNVML 库精确采集 GPU 功耗数据，计算净能耗以消除空闲功耗干扰。
*   **局限性:** 部分结果（如 99.56% 节能）需人工验证，LLM-as-a-Judge 未能完全识别功能偏差；实验未使用温度参数调整（因模型限制），可能限制了迭代优化的潜力。

## Further Thoughts

LASSI-EE 展示了 Agentic AI 在代码优化中的潜力，未来可以探索 LLM 在自动化实验设计或跨领域代码迁移中的应用；此外，框架可进一步整合实时硬件反馈（如动态功耗变化）指导优化，或通过多模型协作（如分工上下文总结与代码生成）提升效果；跨架构扩展（如支持 OpenMP 或 SYCL）也值得研究，以实现更广泛的适用性。