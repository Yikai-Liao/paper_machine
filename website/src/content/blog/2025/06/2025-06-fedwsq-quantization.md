---
title: "FedWSQ: Efficient Federated Learning with Weight Standardization and Distribution-Aware Non-Uniform Quantization"
pubDatetime: 2025-06-30T04:46:25+00:00
slug: "2025-06-fedwsq-quantization"
type: "arxiv"
id: "2506.23516"
score: 0.48340308586806857
author: "grok-3-latest"
authors: ["Seung-Wook Kim", "Seongyeol Kim", "Jiah Kim", "Seowon Ji", "Se-Ho Lee"]
tags: ["Federated Learning", "Data Heterogeneity", "Communication Efficiency", "Quantization", "Weight Standardization"]
institution: ["Pukyong National University", "Konkuk University", "Jeonbuk National University"]
description: "FedWSQ 通过权重标准化和分布感知非均匀量化，显著提升了联邦学习在数据异构性和通信受限场景下的性能与效率。"
---

> **Summary:** FedWSQ 通过权重标准化和分布感知非均匀量化，显著提升了联邦学习在数据异构性和通信受限场景下的性能与效率。 

> **Keywords:** Federated Learning, Data Heterogeneity, Communication Efficiency, Quantization, Weight Standardization

**Authors:** Seung-Wook Kim, Seongyeol Kim, Jiah Kim, Seowon Ji, Se-Ho Lee

**Institution(s):** Pukyong National University, Konkuk University, Jeonbuk National University


## Problem Background

联邦学习（Federated Learning, FL）在现实应用中面临数据异构性（客户端数据分布非独立同分布）、部分客户端参与和通信瓶颈三大挑战，导致本地梯度发散、全球模型收敛缓慢和性能下降。
本文旨在通过提高训练稳定性和通信效率，解决这些关键问题，使联邦学习更适用于资源受限和数据分布复杂的场景。

## Method

*   **核心思想:** 提出 FedWSQ 框架，通过权重标准化（Weight Standardization, WS）和分布感知非均匀量化（Distribution-Aware Non-Uniform Quantization, DANUQ）结合，增强联邦学习的训练稳定性和通信效率。
*   **权重标准化（WS）:** 
    *   WS 是一种即插即用的技术，通过标准化卷积或线性层的权重向量，稳定神经网络的学习过程。
    *   在联邦学习中，WS 通过梯度过滤机制减少客户端漂移（client drift），具体方法是将梯度投影到特定子空间，去除导致过拟合的成分（如参数对齐和均值成分），从而提高模型对数据异构性的鲁棒性和泛化能力。
    *   与 FedWon 等方法不同，FedWSQ 传输预标准化参数（PSP），保留本地信息，同时通过梯度过滤隐式缓解有害发散。
*   **分布感知非均匀量化（DANUQ）:** 
    *   DANUQ 是一种新型量化策略，基于本地模型参数更新（LMPUs）服从正态分布的假设，利用统计特性设计固定的量化级别（Quantization Levels, QLs），以最小化量化误差。
    *   相比均匀量化（UQ），DANUQ 在高密度数据区域分配更细的量化间隔，保留更多关键信息；使用标准差而非最大绝对值作为缩放因子，增强对异常值的鲁棒性。
    *   引入全局缩放向量，通过客户端协作更新，确保量化一致性；采用数值搜索算法确定最优量化级别。
*   **混合位宽分配:** 提出固定位宽分配（FBA）和动态位宽分配（DBA）策略，允许客户端根据通信条件选择 1-bit、2-bit 或 4-bit 量化，进一步优化通信成本。
*   **关键点:** WS 和 DANUQ 互补，WS 提升训练稳定性，DANUQ 降低通信开销，且整个框架可与现有联邦学习算法（如 FedAvg）无缝集成。

## Experiment

*   **有效性:** FedWSQ 在 CIFAR-10、CIFAR-100 和 Tiny-ImageNet 数据集上显著优于现有方法（如 FedAvg、FedProx、FedPAQ、FedRCL、FedACG），尤其在高度异构设置（α=0.1）下，1-bit 量化仍保持高性能，例如在 CIFAR-100 上准确率达 62.05%，比 FedRCL 高 3.79%。
*   **通信效率:** 通过 DANUQ 和混合位宽策略（FBA 和 DBA），FedWSQ 以平均 2.3 位/参数的通信成本实现接近 4-bit 量化的性能，通信开销大幅降低。
*   **对比分析:** 相比均匀量化（UQ），DANUQ 在低位宽下性能下降更小；结合 WS 后，整体性能进一步提升，显示两者的协同效应。
*   **实验设置合理性:** 实验覆盖 i.i.d. 和 non-i.i.d. 数据分布（Dirichlet 参数 α=0.1、0.3、0.6），参与率设为 5%（100 个客户端），并测试多种位宽（1-bit、2-bit、4-bit、32-bit），数据集和对比方法选择全面；但未充分探讨极端通信延迟或更大规模客户端场景下的鲁棒性。
*   **额外验证:** 损失景观分析显示 FedWSQ 具有更平滑的损失曲面（Hessian 顶特征值最低为 135.8），表明其泛化能力更强；对不同骨干网络（如 ResNet-18、MobileViT）的适用性也得到验证。

## Further Thoughts

1. **梯度过滤的扩展应用:** WS 的梯度过滤机制可推广至其他分布式学习或多任务学习场景，用于减少模型间干扰，未来可探索自适应过滤策略，根据客户端数据特性动态调整过滤强度。
2. **量化分布的动态适应:** DANUQ 基于正态分布假设，但实际数据分布可能随任务变化，是否可引入在线学习机制或混合分布模型（如高斯混合模型）动态调整量化级别，提升量化精度？
3. **位宽分配的智能优化:** 混合位宽分配策略目前较为简单，未来可通过强化学习或优化算法，根据客户端通信条件和数据重要性动态分配位宽，进一步平衡性能与效率。