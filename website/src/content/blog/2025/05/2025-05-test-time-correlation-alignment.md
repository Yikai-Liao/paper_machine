---
title: "Test-time Correlation Alignment"
pubDatetime: 2025-05-01T13:59:13+00:00
slug: "2025-05-test-time-correlation-alignment"
type: "arxiv"
id: "2505.00533"
score: 0.616068933900758
author: "grok-3-latest"
authors: ["Linjing You", "Jiabao Lu", "Xiayuan Huang"]
tags: ["Test-Time Adaptation", "Correlation Alignment", "Domain Shift", "Feature Distribution", "Efficiency"]
institution: ["Institute of Automation, Chinese Academy of Sciences", "College of Science, Beijing Forestry University"]
description: "本文提出测试时相关性对齐（TCA）范式，通过简单的线性变换实现特征相关性对齐，显著提升测试时适应性能，同时保持高效率和抗遗忘能力。"
---

> **Summary:** 本文提出测试时相关性对齐（TCA）范式，通过简单的线性变换实现特征相关性对齐，显著提升测试时适应性能，同时保持高效率和抗遗忘能力。 

> **Keywords:** Test-Time Adaptation, Correlation Alignment, Domain Shift, Feature Distribution, Efficiency

**Authors:** Linjing You, Jiabao Lu, Xiayuan Huang

**Institution(s):** Institute of Automation, Chinese Academy of Sciences, College of Science, Beijing Forestry University


## Problem Background

深度神经网络在训练和测试数据分布不一致（distribution shift）时性能下降，而传统领域适应方法因隐私和资源限制无法访问源数据，难以适用。
现有测试时适应（Test-Time Adaptation, TTA）方法主要关注实例级对齐，忽视特征相关性对齐（correlation alignment），且依赖高成本的反向传播操作，同时面临领域遗忘问题（即适应测试领域后对源领域性能下降）。
本文提出测试时相关性对齐（Test-time Correlation Alignment, TCA），旨在通过相关性对齐提升模型在测试领域的性能，同时降低计算开销并减少领域遗忘。

## Method

*   **核心思想:** 在不访问源数据的情况下，通过从测试数据中构建‘伪源相关性’（pseudo-source correlation），实现测试数据特征相关性对齐，提升模型适应性，同时避免模型更新带来的计算成本和领域遗忘。
*   **具体实现 - 伪源构建:** 从测试数据中选择高置信度（high-certainty）实例，计算其特征相关性矩阵，作为伪源相关性。论文通过理论分析证明，高置信度实例的相关性可以近似源领域的相关性。
*   **具体实现 - LinearTCA:** 基于伪源相关性和测试数据的相关性矩阵，计算一个线性变换矩阵 W，将测试数据的特征分布对齐到伪源分布，同时对齐实例级的均值偏移（mean shift）。该方法仅通过前向计算完成适应，不需要更新模型参数，避免了反向传播的高计算成本。
*   **具体实现 - LinearTCA[+]:** 作为一种即插即用（plug-and-play）模块，LinearTCA[+] 可与现有 TTA 方法结合，在其他方法更新模型后，进一步应用线性变换提升性能，增强适应效果。
*   **关键优势:** 方法简单高效，不依赖反向传播，计算开销低；不更新模型参数，有效防止领域遗忘；理论上提供了相关性对齐降低测试误差的保证。

## Experiment

*   **准确率提升:** LinearTCA 相比源模型在多个数据集上表现出显著提升，例如在 OfficeHome 数据集上平均提升 1.69%-8.48%；LinearTCA[+] 作为增强模块，在所有测试数据集和骨干网络上均提升性能，尤其在 CIFAR-10C (ViT-B/16) 上比最佳基线提升 5.88%。
*   **效率优势:** LinearTCA 计算开销极低，在 CIFAR-10C 数据集上，GPU 内存使用和运行时间远低于其他 TTA 方法，例如运行时间仅为最佳基线 EATA 的 6‰，适合资源受限的边缘设备。
*   **抗遗忘能力:** LinearTCA 在适应测试领域后对源领域性能下降极小，甚至在 PACS 数据集上表现出‘正向后向迁移’（positive backward transfer），即源领域性能有所提升，优于其他方法。
*   **实验设置合理性:** 实验覆盖领域泛化（PACS, OfficeHome）和图像损坏适应（CIFAR-10C, CIFAR-100C）任务，使用多种骨干网络（ResNet-18/50, ViT-B/16），评估指标全面（准确率、效率、抗遗忘），设置合理。
*   **局限性:** LinearTCA 对非线性分布偏移的适应能力有限，在部分数据集（如 CIFAR-10/100C）上准确率不如一些高级 TTA 方法，可能是线性变换的限制。

## Further Thoughts

伪源相关性构建的思路启发我们在无源数据场景下，利用测试数据自身的统计特性近似源分布，可扩展至其他无监督学习任务；
线性变换的高效性提示在资源受限场景中，简单统计方法可能比复杂深度学习方法更实用；
正向后向迁移现象值得深入研究，可能揭示领域适应中的知识迁移机制，未来可设计实验探索其在不同数据集上的普适性。