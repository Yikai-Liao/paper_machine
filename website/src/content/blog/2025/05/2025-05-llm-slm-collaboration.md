---
title: "A Survey on Collaborative Mechanisms Between Large and Small Language Models"
pubDatetime: 2025-05-12T11:48:42+00:00
slug: "2025-05-llm-slm-collaboration"
type: "arxiv"
id: "2505.07460"
score: 0.6239965267617734
author: "grok-3-latest"
authors: ["Yi Chen", "JiaHao Zhao", "HaoHao Han"]
tags: ["LLM", "SLM", "Collaboration", "Efficiency", "Edge Computing"]
institution: ["Chengdu Institute of Computer Applications, Chinese Academy of Sciences", "University of Chinese Academy of Sciences"]
description: "本文系统综述了大型语言模型（LLMs）与小型语言模型（SLMs）之间的协作机制，分类多种协作模式，分析关键技术和应用场景，并探讨当前挑战与未来趋势，为高效 AI 系统设计提供了全面指导。"
---

> **Summary:** 本文系统综述了大型语言模型（LLMs）与小型语言模型（SLMs）之间的协作机制，分类多种协作模式，分析关键技术和应用场景，并探讨当前挑战与未来趋势，为高效 AI 系统设计提供了全面指导。 

> **Keywords:** LLM, SLM, Collaboration, Efficiency, Edge Computing

**Authors:** Yi Chen, JiaHao Zhao, HaoHao Han

**Institution(s):** Chengdu Institute of Computer Applications, Chinese Academy of Sciences, University of Chinese Academy of Sciences


## Problem Background

大型语言模型（LLMs）在自然语言处理、代码生成等领域表现出强大的能力，但其高资源消耗和高延迟限制了在边缘设备上的部署，尤其是在需要低延迟和高隐私的场景中；小型语言模型（SLMs）则因其轻量化和高效性适合资源受限环境，但性能有限。
论文旨在解决如何通过 LLMs 和 SLMs 的协作机制，平衡性能与效率，满足边缘设备上的低延迟、隐私保护、个性化等实际需求。

## Method

*   **核心思想:** 通过设计多种协作机制，结合 LLMs 的强大语言能力和 SLMs 的高效部署特性，构建智能、高效的 AI 推理系统。
*   **具体协作模式:**
    *   **管道协作（Pipeline Collaboration）**：SLMs 负责初步处理或生成候选结果，LLMs 进行复杂推理或知识整合，例如在推荐系统中 SLMs 基于实时数据重新排序 LLMs 生成的候选项。
    *   **混合/路由协作（Hybrid/Routing Collaboration）**：通过路由机制根据任务复杂度、领域或延迟需求选择合适的模型处理，如 CITER 框架在 token 级别动态分配任务。
    *   **辅助/增强协作（Auxiliary/Enhancement Collaboration）**：一个模型辅助另一个模型提升性能，例如 Collab-RAG 框架中 SLMs 分解复杂查询以增强 LLMs 的推理能力。
    *   **知识蒸馏驱动协作（Knowledge Distillation-Driven Collaboration）**：通过知识蒸馏将 LLMs 的知识转移到 SLMs，提升 SLMs 在复杂任务上的能力，同时保持计算优势。
    *   **集成/融合协作（Integration/Fusion Collaboration）**：在架构或输出层面深度结合两者的优势，如 Hymba 模型通过混合注意力机制实现高效上下文总结。
*   **关键技术支持:**
    *   **任务分配与智能路由**：基于任务复杂度动态分配模型，如 Hybrid LLM 框架根据查询难度选择模型。
    *   **模型间通信与接口设计**：通过自然语言接口或结构化中间表示确保高效信息传递。
    *   **模型融合与结果整合**：通过权重平均或集成学习结合模型输出。
    *   **状态同步与上下文管理**：在多轮对话中保持一致性。
    *   **动态资源调度**：优化计算资源分配，结合边缘与云计算。
*   **特点:** 这些方法旨在不牺牲性能的前提下优化效率，特别关注边缘设备上的实际应用需求。

## Experiment

*   **有效性:** 论文作为综述未直接提供实验数据，但引用了大量研究成果验证协作机制的效果。例如，Hybrid SLM-LLM 框架在边缘-云协作中将端到端延迟降低了 40%，同时保持高响应质量；在工业异常检测中，Speculative Decoding 机制将响应时间从秒级缩短到毫秒级，精度接近纯 LLM 方案。
*   **应用场景全面性:** 论文覆盖了多种实际场景（如实时低延迟推理、隐私敏感场景、个性化定制、离线环境、能源受限场景），表明协作机制在不同需求下均有显著优势。
*   **评估合理性与局限:** 实验设置引用了多个框架（如 CITER、CoGenesis），但缺乏统一基准测试平台，评估复杂性较高，部分场景的实验全面性仍需进一步验证。
*   **开销分析:** 协作机制在引入路由决策和模型间通信时会增加额外延迟和计算成本，尤其在边缘-云场景中通信开销可能成为瓶颈。

## Further Thoughts

论文中提出的动态自适应协作框架（基于强化学习和元学习）非常具有启发性，未来 AI 系统可能通过‘学会如何协作’实现更高的自主性和效率；此外，模型能力在神经网络层级上的深度融合为构建更紧密耦合的混合模型提供了新思路；多模态和具身智能中的协作模式（SLMs 作为特定模态处理器，LLMs 负责跨模态推理）也为构建可扩展的多模态 AI 系统提供了创新方向，特别是在机器人和虚拟代理领域可能有广泛应用。