---
title: "QPART: Adaptive Model Quantization and Dynamic Workload Balancing for Accuracy-aware Edge Inference"
pubDatetime: 2025-06-30T15:03:35+00:00
slug: "2025-06-qpart-edge-inference"
type: "arxiv"
id: "2506.23934"
score: 0.4204459716507776
author: "grok-3-latest"
authors: ["Xiangchen Li", "Saeid Ghafouri", "Bo Ji", "Hans Vandierendonck", "Deepu John", "Dimitrios S. Nikolopoulos"]
tags: ["Edge Computing", "Model Quantization", "Inference Partitioning", "Accuracy Degradation", "Workload Balancing"]
institution: ["Not explicitly mentioned in the provided text, likely affiliated with universities or research institutions in the US based on funding sources"]
description: "本文提出QPART，一个准确性感知的边缘推理系统，通过联合模型量化和分区优化，显著降低延迟和能耗，同时保持推理准确性。"
---

> **Summary:** 本文提出QPART，一个准确性感知的边缘推理系统，通过联合模型量化和分区优化，显著降低延迟和能耗，同时保持推理准确性。 

> **Keywords:** Edge Computing, Model Quantization, Inference Partitioning, Accuracy Degradation, Workload Balancing

**Authors:** Xiangchen Li, Saeid Ghafouri, Bo Ji, Hans Vandierendonck, Deepu John, Dimitrios S. Nikolopoulos

**Institution(s):** Not explicitly mentioned in the provided text, likely affiliated with universities or research institutions in the US based on funding sources


## Problem Background

随着深度学习模型在边缘设备上的部署需求增加，边缘推理面临计算能力有限、内存限制、高延迟、带宽依赖以及隐私风险等挑战。
不同设备和任务对准确性、延迟和能耗的需求各异，传统方法（如模型剪枝、知识蒸馏）往往需要重新训练或大幅修改模型，成本高且适应性差。
论文旨在设计一个自适应的推理系统，针对异构边缘设备和动态环境，动态优化模型和计算负载分配，以平衡延迟、能耗和准确性。

## Method

*   **核心思想:** 提出QPART，一个准确性感知的边缘推理服务系统，通过自适应模型量化和动态负载均衡，在不重新训练模型的前提下，优化推理性能。
*   **模型量化:** 采用训练后量化（Post-Training Quantization），对模型参数和中间激活值进行层级量化（Layer-wise Quantization），降低内存占用和传输延迟；通过量化噪声和对抗噪声的度量，确保准确性降级在可接受范围内。
*   **模型分区:** 将神经网络分为两部分，一部分在边缘设备执行，另一部分在服务器执行，通过优化分区点（Partition Point）平衡本地和服务器计算负载。
*   **联合优化框架:** 构建优化问题，综合考虑时间消耗、能耗、服务器成本和传输延迟，以准确性降级为约束，通过闭合解（Closed-form Solution）确定每层量化位宽和最佳分区点。
*   **离线与在线算法:** 离线算法预计算不同分区点和准确性需求下的量化模式，减少实时计算开销；在线算法根据设备请求（如计算能力、信道容量、准确性需求）动态选择最优模式并量化模型。
*   **关键优势:** 无需修改原始模型，适应性强，理论度量确保准确性控制，同时优化多目标性能。

## Experiment

*   **有效性:** 在多个数据集（如MNIST, SVHN, CIFAR10, CIFAR100, ImageNet）和模型（如ResNet18, ResNet34, ResNet50）上，QPART将通信负载压缩至11.88%-18.12%，准确性下降控制在0.08%-0.66%，显著降低延迟和能耗。
*   **对比分析:** 与无优化直接推理、基于自编码器和模型剪枝的方法相比，QPART在时间消耗、能耗和总成本上均表现出色，尤其在不同分区点下性能稳定。
*   **实验设置:** 实验基于模拟平台，涵盖多种数据集和模型架构，模拟了边缘设备的计算和通信条件（如时钟频率、信道容量），设置较为全面；但缺乏真实设备部署测试，可能存在理论与实际环境的偏差。
*   **显著性:** 通信负载减少高达80%以上，准确性下降低于1%，方法提升明显，适用于资源受限的边缘场景。

## Further Thoughts

论文中准确性降级的理论度量（基于量化噪声和对抗噪声）为模型压缩提供了可控框架，启发我们可以在其他压缩技术中引入类似分析以预测性能影响；
离线预计算与在线动态选择的优化策略有效降低实时开销，可扩展至联邦学习或实时调度等领域；
联合优化的灵活性（通过权重调整时间、能耗和成本目标）提示我们设计更通用的多目标优化框架，适应复杂边缘AI应用场景。