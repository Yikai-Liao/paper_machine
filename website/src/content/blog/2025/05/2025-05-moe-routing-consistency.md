---
title: "Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models"
pubDatetime: 2025-05-21T22:13:09+00:00
slug: "2025-05-moe-routing-consistency"
type: "arxiv"
id: "2505.16056"
score: 0.644942851954028
author: "grok-3-latest"
authors: ["Miren Tian", "Zhongyu Wei", "Jingcong Liang", "Yitong Li", "Siyuan Wang", "Duyu Tang"]
tags: ["LLM", "Mixture of Experts", "Expert Offloading", "Routing Consistency", "Caching Strategy"]
institution: ["Huawei Technologies Ltd.", "Fudan University", "University of Southern California"]
description: "本文提出SRP和SCH指标，系统分析MoE模型的局部路由一致性差异及其影响因素，为内存高效的MoE设计和专家卸载部署提供指导。"
---

> **Summary:** 本文提出SRP和SCH指标，系统分析MoE模型的局部路由一致性差异及其影响因素，为内存高效的MoE设计和专家卸载部署提供指导。 

> **Keywords:** LLM, Mixture of Experts, Expert Offloading, Routing Consistency, Caching Strategy

**Authors:** Miren Tian, Zhongyu Wei, Jingcong Liang, Yitong Li, Siyuan Wang, Duyu Tang

**Institution(s):** Huawei Technologies Ltd., Fudan University, University of Southern California


## Problem Background

混合专家模型（Mixture-of-Experts, MoE）通过稀疏激活专家模块实现大型语言模型（LLMs）的高效扩展，但在内存受限设备上的部署面临挑战。
专家卸载（Expert Offloading）技术通过将部分专家缓存到快速内存、其余存储在慢速内存来降低内存需求，但频繁的专家切换会导致推理效率下降。
论文关注专家激活的局部路由一致性（Local Routing Consistency）——即连续token激活相似专家的特性——对卸载效率的影响，并指出不同MoE模型在这一特性上的差异尚未被充分研究。

## Method

*   **核心思想:** 提出两个指标来量化MoE模型的局部路由一致性，分析其与模型架构和专家特化性的关系，为高效专家卸载提供指导。
*   **具体指标:**
    *   **Segment Routing Best Performance (SRP):** 衡量一个固定专家组在一段连续token上的路由决策与原始路由器的接近程度，反映专家激活的局部一致性。此指标不依赖额外参数（如缓存大小），适用于分析单个专家或专家组的激活模式。
    *   **Segment Cache Best Hit Rate (SCH):** 在给定缓存大小限制下，评估段级缓存的最大命中率，反映实际专家卸载系统的性能潜力。SCH通过计算每个段内专家激活频率，确定缓存固定数量专家时的最佳命中率。
*   **分析维度:** 使用这两个指标，研究不同模型架构（如是否每层应用MoE、是否使用共享专家）对一致性的影响，并探讨专家特化性（如领域特化、词汇特化）的作用。
*   **实现细节:** 在实验中，基于输入序列的专家激活数据，计算每个段的激活频率，进而通过枚举阈值（如SRP中的α）或排序专家频率（如SCH中的Top-ρk）来确定最佳性能。

## Experiment

*   **有效性:** 实验覆盖20个MoE模型（参数规模3B至54B），结果显示局部路由一致性在不同模型间差异显著，尤其在长段长度（如m=16及以上）时差异更为明显。每层应用MoE且不使用共享专家的模型（如LLaMA-MoE-v2）表现出最高一致性。
*   **关键发现:** 领域特化专家对一致性的贡献大于词汇特化专家；缓存大小约为活跃专家数量的2倍时，大多数模型能在缓存效果和效率间达到最佳平衡。
*   **实验设置合理性:** 实验数据来源于RedPajama的多个领域子集，涵盖不同规模和架构的模型，并通过置信区间分析增强结果可信度。但实验未涉及更大规模模型（如DeepSeek-V3），且指标的理论性质可能与实际卸载系统性能存在偏差。

## Further Thoughts

论文中领域特化专家对局部路由一致性贡献的发现启发了我：是否可以通过设计更强的领域特化机制（如针对特定任务的专家预训练）来提升一致性，从而优化专家卸载效率？此外，是否可以将局部路由一致性与动态路由策略结合，通过实时调整路由决策适应不同输入上下文特性？这可能需要在训练阶段引入一致性约束或在推理时使用上下文感知的缓存策略。