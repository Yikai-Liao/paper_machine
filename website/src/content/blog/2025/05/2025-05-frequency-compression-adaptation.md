---
title: "Frequency Composition for Compressed and Domain-Adaptive Neural Networks"
pubDatetime: 2025-05-27T08:33:04+00:00
slug: "2025-05-frequency-compression-adaptation"
type: "arxiv"
id: "2505.20890"
score: 0.4798793220088603
author: "grok-3-latest"
authors: ["Yoojin Kwon", "Hongjun Suh", "Wooseok Lee", "Taesik Gong", "Songyi Han", "Hyung-Sin Kim"]
tags: ["Neural Network", "Model Compression", "Domain Adaptation", "Frequency Decomposition", "Quantization"]
institution: ["Seoul National University", "UNIST", "Google"]
description: "本文提出 *CoDA* 框架，通过频率分解统一解决模型压缩和域适应问题，在训练时聚焦低频分量增强泛化性，测试时利用全频分量和频率感知批归一化适应目标域，显著提升了资源受限设备上的性能。"
---

> **Summary:** 本文提出 *CoDA* 框架，通过频率分解统一解决模型压缩和域适应问题，在训练时聚焦低频分量增强泛化性，测试时利用全频分量和频率感知批归一化适应目标域，显著提升了资源受限设备上的性能。 

> **Keywords:** Neural Network, Model Compression, Domain Adaptation, Frequency Decomposition, Quantization

**Authors:** Yoojin Kwon, Hongjun Suh, Wooseok Lee, Taesik Gong, Songyi Han, Hyung-Sin Kim

**Institution(s):** Seoul National University, UNIST, Google


## Problem Background

现代设备端神经网络应用需要在资源受限的情况下运行，同时适应不可预测的域偏移（domain shift）。
传统研究分别关注模型压缩（compression）和域适应（domain adaptation），而未解决两者的结合问题：压缩模型注重固定域内效率，大型模型则擅长处理域偏移。
本文旨在解决这一双重挑战，确保在资源受限设备上部署的神经网络既高效又能适应动态环境。

## Method

*   **核心思想:** 提出 *CoDA* 框架，通过频率分解（frequency decomposition）统一解决模型压缩和域适应问题，利用图像的不同频率分量分别增强模型的泛化性和适应性。
*   **训练阶段 (Low-Frequency Components QAT):** 
    *   使用量化感知训练（Quantization-Aware Training, QAT），将输入图像通过 2D 傅里叶变换分解为低频分量（LFC）和高频分量（HFC）。
    *   仅基于 LFC 进行训练，因为 LFC 被认为是更具泛化性的特征，适合容量受限的压缩模型，帮助模型优先学习通用知识而非源域细节。
    *   通过调整低通滤波器的半径（radius）控制 LFC 的范围，确保模型捕获最鲁棒的特征。
*   **测试阶段 (Test-Time Adaptation with FABN):** 
    *   在测试时，利用全频分量（FFC，包括 LFC 和 HFC）来适应目标域，捕捉域特定细节。
    *   提出频率感知批归一化（Frequency-Aware Batch Normalization, FABN），分别处理 LFC 和 HFC：
        *   对 LFC，初始化为源域的运行统计量，并通过指数移动平均（EMA）逐步更新，保留通用知识。
        *   对 HFC，直接使用当前测试批量的统计量，快速适应目标域细节。
        *   最终整合 LFC 和 HFC 的统计量进行归一化。
    *   FABN 可与现有 TTA 方法（如 NORM、TENT、SAR）协同工作，通过调整批归一化参数而非模型权重，降低计算开销。
*   **关键优势:** 不替代现有 QAT 和 TTA 方法，而是通过频率分解策略增强其性能，同时保持设备端部署的可行性。

## Experiment

*   **有效性:** *CoDA* 在显著压缩模型大小（4-16 倍）的同时，相比全精度 TTA 基线，在 CIFAR10-C 上提升了 7.96 个百分点，在 ImageNet-C 上提升了 5.37 个百分点，展现了在资源受限环境下的优越性能。
*   **优越性:** 与现有 TTA 方法（如 NORM、TENT、SAR）结合后，*CoDA* 一致性地提升了性能，尤其在低位量化（如 2-bit）模型上表现突出，甚至超越全精度模型。
*   **全面性:** 实验覆盖了多种模型架构（ResNet、MobileNet、EfficientNet）、量化位宽（2-bit、4-bit、8-bit）和域偏移基准数据集（CIFAR10-C、ImageNet-C、ImageNet-R、ImageNet-Sketch），验证了方法的普适性；消融研究表明 LFC QAT 和 FABN 的协同作用显著优于单独使用任一组件。
*   **开销:** 主要额外开销在于频率分解和 FABN 的计算，但由于针对压缩模型，整体资源需求仍在设备端可接受范围内。

## Further Thoughts

频率分解的思路非常具有启发性，低频分量作为域无关的通用特征，高频分量作为域特定细节的分离策略，可以扩展到其他领域。例如，是否可以通过频率分解设计更高效的模型蒸馏方法，优先传递低频知识以减少过拟合？或者在多任务学习中动态调整频率关注点以平衡任务需求？此外，在隐私保护领域，是否可以通过丢弃高频分量减少数据中的敏感细节，同时保留通用特征用于推理？