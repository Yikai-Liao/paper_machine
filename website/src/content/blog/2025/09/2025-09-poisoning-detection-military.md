---
title: "AutoDetect: Designing an Autoencoder-based Detection Method for Poisoning Attacks on Object Detection Applications in the Military Domain"
pubDatetime: 2025-09-03T10:05:02+00:00
slug: "2025-09-poisoning-detection-military"
type: "arxiv"
id: "2509.03179"
score: 0.5823599604022743
author: "grok-3-latest"
authors: ["Alma M. Liezenga", "Stefan Wijnja", "Puck de Haan", "Niels W. T. Brink", "Jip J. van Stijn", "Yori Kamphuis", "Klamer Schutte"]
tags: ["Object Detection", "Poisoning Attack", "Anomaly Detection", "Adversarial Defense", "Military Application"]
institution: ["TNO, The Hague, The Netherlands"]
description: "本文提出 AutoDetect，一种基于自编码器的轻量化方法，用于检测军事目标检测系统中的毒化攻击，并在多个数据集上展现出优于现有方法的性能，同时揭示了毒化攻击的实际威胁与局限。"
---

> **Summary:** 本文提出 AutoDetect，一种基于自编码器的轻量化方法，用于检测军事目标检测系统中的毒化攻击，并在多个数据集上展现出优于现有方法的性能，同时揭示了毒化攻击的实际威胁与局限。 

> **Keywords:** Object Detection, Poisoning Attack, Anomaly Detection, Adversarial Defense, Military Application

**Authors:** Alma M. Liezenga, Stefan Wijnja, Puck de Haan, Niels W. T. Brink, Jip J. van Stijn, Yori Kamphuis, Klamer Schutte

**Institution(s):** TNO, The Hague, The Netherlands


## Problem Background

在军事领域，目标检测系统因广泛使用开源数据集和预训练模型而面临毒化攻击（Poisoning Attacks）的威胁，这种攻击通过在训练数据中注入对抗性补丁等恶意样本破坏模型性能，可能导致严重后果。
现有研究对目标检测系统的毒化攻击及其检测方法关注不足，尤其在军事场景中缺乏针对性解决方案，因此亟需评估此类攻击的实际威胁并开发有效的检测手段。

## Method

*   **核心思想:** 提出 AutoDetect，一种基于自编码器的无监督异常检测方法，利用自编码器在正常数据上的低重建误差特性，识别包含对抗性补丁的异常图像。
*   **具体实现:** 
    *   首先，在干净的非异常图像数据集上预训练自编码器，学习正常数据的分布特征。
    *   然后，将输入图像切分为等大小的切片（Slices），计算每个切片的平均重建误差（Slice Error），并基于验证集的切片误差拟合正态分布。
    *   测试时，计算查询图像的最大切片误差，并根据其在正态分布中的概率（通过手动设置阈值）判断是否为异常（即是否包含对抗性补丁）。
*   **优势设计:** 方法轻量化且模型无关，仅依赖于数据本身，无需访问目标检测模型，易于在不同军事场景中部署。
*   **对比方法:** 论文还评估了现有异常检测方法（如 PatchCore, PaDIM）和对抗性补丁检测方法（如 PAD），指出其在多样性数据集上的性能不足及高计算成本问题。

## Experiment

*   **毒化攻击效果:** 使用 BadDet 框架的 Global Misclassification Attack（GMA）在自定义军事数据集 MilCivVeh 上测试，攻击成功率（ASR）最高达 52.2%（YOLOv3，40% 毒化率），但需较高毒化率，且模型在干净数据上的 F1 分数显著下降，表明攻击效果可能更多源于误标签而非补丁本身，实际威胁性存疑。
*   **检测方法效果:** AutoDetect 在 MS COCO, VOC2007 和 MilCivVeh 数据集上的 AUROC 均超过 0.94，显著优于其他方法（如 PAD 的 0.633-0.902, PaDIM 的 0.538-0.637），且计算效率高、内存占用低。
*   **实验设置分析:** 实验涵盖了不同补丁大小、类型及切片大小的影响，设置较为全面，但 MilCivVeh 数据集规模小、代表性有限，可能影响结论的普适性；此外，毒化攻击测试中未明确区分补丁与误标签的影响，需进一步验证。

## Further Thoughts

AutoDetect 利用自编码器重建误差检测异常的思路具有广泛应用潜力，可扩展至其他安全领域如网络流量异常检测；同时，论文揭示的物理对抗性补丁在军事场景中的潜在威胁，启发我们思考数字毒化与物理攻击的结合效应，以及跨域检测方法的设计；此外，AutoDetect 对补丁大小和类型的敏感性提示，攻击者可能通过低对比度或小尺寸补丁规避检测，这为对抗性防御研究开辟了新方向。