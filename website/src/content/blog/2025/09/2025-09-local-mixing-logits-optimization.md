---
title: "Generating Transferrable Adversarial Examples via Local Mixing and Logits Optimization for Remote Sensing Object Recognition"
pubDatetime: 2025-09-09T08:20:19+00:00
slug: "2025-09-local-mixing-logits-optimization"
type: "arxiv"
id: "2509.07495"
score: 0.6127894783428978
author: "grok-3-latest"
authors: ["Chun Liu", "Hailong Wang", "Bingqian Zhu", "Panpan Ding", "Zheng Zheng", "Tao Xu", "Zhigang Han", "Jiayao Wang"]
tags: ["Adversarial Attack", "Transferability", "Remote Sensing", "Input Transformation", "Loss Optimization"]
institution: ["State Key Laboratory of Spatial Datum, College of Remote Sensing and Geoinformatics Engineering, Henan University", "School of Computer and Information Engineering, Henan University", "School of Automation Science and Electrical Engineering, Beihang University"]
description: "本文提出了一种通过局部混合和 logits 优化生成高迁移性对抗样本的方法，显著提升了遥感目标识别中黑盒攻击的成功率，特别是在跨架构模型上的表现优于现有技术。"
---

> **Summary:** 本文提出了一种通过局部混合和 logits 优化生成高迁移性对抗样本的方法，显著提升了遥感目标识别中黑盒攻击的成功率，特别是在跨架构模型上的表现优于现有技术。 

> **Keywords:** Adversarial Attack, Transferability, Remote Sensing, Input Transformation, Loss Optimization

**Authors:** Chun Liu, Hailong Wang, Bingqian Zhu, Panpan Ding, Zheng Zheng, Tao Xu, Zhigang Han, Jiayao Wang

**Institution(s):** State Key Laboratory of Spatial Datum, College of Remote Sensing and Geoinformatics Engineering, Henan University, School of Computer and Information Engineering, Henan University, School of Automation Science and Electrical Engineering, Beihang University


## Problem Background

深度神经网络（DNNs）在遥感目标识别中的广泛应用受到对抗性攻击的威胁，通过微小扰动即可误导模型输出错误结果，严重影响系统可靠性和安全性，尤其是在自然灾害监测、城市规划和军事侦察等关键领域。
现有对抗性攻击方法在黑盒场景下的迁移性不足，容易过拟合到替代模型，且传统交叉熵损失优化存在梯度消失问题；此外，遥感图像具有复杂的背景多样性和语义一致性，现有全局混合或区域交换策略可能破坏全局语义特征，降低对抗样本质量。

## Method

*   **核心思想:** 提出一种通过局部混合和 logits 优化生成高迁移性对抗样本的框架，针对遥感目标识别任务，增强非目标黑盒攻击的效果。
*   **局部混合策略 (Local Mixing):** 不同于传统 MixUp（全局混合两张图像）或 MixCut（直接拼接不同图像区域），该方法仅对两张图像的随机矩形区域进行比例混合，生成多样化且语义一致的输入，保留全局语义信息，同时通过多次变换和随机打乱增加输入多样性，减少对替代模型的过拟合。
*   **Logits 优化:** 针对交叉熵损失在迭代优化中的梯度消失问题，提出基于 logits 的非目标攻击损失，直接最小化真实类别的 logits 值，绕过 softmax 操作，保持梯度信号强度，从而提高优化效果。
*   **扰动平滑损失 (Perturbation Smoothing Loss):** 引入低通滤波器（均值卷积核）抑制扰动中的高频噪声，增强对抗样本跨模型的迁移性，避免不同模型对高频成分的敏感性差异导致迁移性下降。
*   **整体流程:** 基于 MI-FGSM 框架，结合动量更新和梯度平滑，多次应用局部混合变换计算平均梯度，确保优化方向稳定，并通过 PGD 迭代更新对抗样本，限制扰动在 L∞ 范数范围内以保持不可感知性。

## Experiment

*   **有效性:** 在 FGSCR-42 数据集上，黑盒攻击成功率（ASR）平均提升 8.84%，最高提升 25.81%；在 MTARSI 数据集上，平均提升 12.15%，最高提升 32.28%，特别是在 ResNet-50 攻击 Inception-ResNet-V2 等异构模型场景下效果显著，证明方法在跨架构迁移性上的优越性。
*   **对比全面性:** 与 12 种主流攻击方法（包括 PGD, MIM, TIM, DIM, Admix 等）对比，涵盖了基于梯度、动量、输入变换等多种策略，论文方法在几乎所有场景下均表现更优。
*   **实验设置合理性:** 实验涉及 6 个经典深度神经网络模型（VGG-16, VGG-19, ResNet-34, ResNet-50, DenseNet-121, Inception-ResNet-V2），覆盖从浅层到深层、不同架构的模型，并在 FGSCR-42 和 MTARSI 两个遥感数据集上测试，设置全面且针对性强。
*   **消融验证:** 消融实验表明局部混合策略优于全局混合（Admix）和无混合（No Mix），logits 优化结合平滑损失显著优于单独使用交叉熵损失，尤其在攻击异构模型时效果明显。
*   **计算开销:** 方法在计算时间上与 Admix 相当，但攻击成功率提升 38.4%，在效率和效果之间取得较好平衡；不过多次变换（M=25）可能增加计算负担，需进一步优化。

## Further Thoughts

局部混合策略启发是否可以通过目标检测算法自适应选择混合区域，优先干扰目标对象周边区域以提升攻击精准性；logits 优化提示是否可以结合对比损失进一步拉开正确类别与其他类别的特征距离；扰动平滑损失的应用让我思考是否可以在频域设计针对遥感图像特定频谱特性的定制化滤波策略；遥感图像背景多样性特性还启发是否可以引入多模态数据（如红外或雷达）生成对抗样本，探索对多模态模型的攻击效果。