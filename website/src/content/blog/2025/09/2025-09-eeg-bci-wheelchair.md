---
title: "EEG-based AI-BCI Wheelchair Advancement: Hybrid Deep Learning with Motor Imagery for Brain Computer Interface"
pubDatetime: 2025-09-30T02:06:04+00:00
slug: "2025-09-eeg-bci-wheelchair"
type: "arxiv"
id: "2509.25667"
score: 0.7531967953315157
author: "grok-3-latest"
authors: ["Bipul Thapa", "Biplov Paneru", "Bishwash Paneru", "Khem Narayan Poudyal"]
tags: ["BCI", "EEG Signal", "Deep Learning", "Motor Imagery", "Wheelchair Control"]
institution: ["Kathmandu University", "Nepal Engineering College, Pokhara University", "Tribhuvan University"]
description: "本文提出了一种基于 BiLSTM-BiGRU 混合深度学习模型的 BCI 轮椅控制系统，通过 EEG 信号分类实现运动想象驱动的轮椅导航，测试准确率达 92.26%，为辅助技术提供了高效解决方案。"
---

> **Summary:** 本文提出了一种基于 BiLSTM-BiGRU 混合深度学习模型的 BCI 轮椅控制系统，通过 EEG 信号分类实现运动想象驱动的轮椅导航，测试准确率达 92.26%，为辅助技术提供了高效解决方案。 

> **Keywords:** BCI, EEG Signal, Deep Learning, Motor Imagery, Wheelchair Control

**Authors:** Bipul Thapa, Biplov Paneru, Bishwash Paneru, Khem Narayan Poudyal

**Institution(s):** Kathmandu University, Nepal Engineering College, Pokhara University, Tribhuvan University


## Problem Background

脑机接口（BCI）技术为行动不便的人群提供了通过脑电图（EEG）信号控制外部设备（如轮椅）的可能性，但现有系统面临分类精度不足、用户适应性差以及对极端身体残疾用户支持有限等挑战。
特别是在 COVID-19 疫情背景下，辅助技术需求迫切，研究旨在通过人工智能和深度学习提升 BCI 轮椅控制系统的性能，为用户带来更高的自主性和生活质量。

## Method

*   **核心思想:** 利用深度学习模型对 EEG 信号进行高效分类，将用户的运动想象（左右手运动）转化为轮椅控制命令，同时确保实时性和硬件兼容性。
*   **数据预处理:** 使用公开 EEG 数据集，通过带通滤波（0.53-100 Hz）去除噪声，并基于事件相关电位（ERP）进行时间窗口分割（每段 4 秒，19 通道，采样频率 200Hz），提取特征向量。
*   **模型架构:** 提出了一种混合深度学习模型 BiLSTM-BiGRU，其中：
    *   BiLSTM 层通过双向处理捕捉 EEG 信号中的长期时间依赖性，编码过去和未来的上下文信息。
    *   BiGRU 层进一步优化时间结构表示，以更少的参数提高计算效率，同时保留关键特征。
    *   最终通过全连接层输出分类结果（静止、右手、左手运动想象）。
*   **验证与评估:** 采用 10 折交叉验证评估模型性能，使用准确率、精确率、召回率和 F1 分数等多指标分析分类效果。
*   **用户界面与硬件:** 开发基于 Tkinter 的 GUI 模拟轮椅运动，系统设计与 Raspberry Pi 兼容，支持实时 EEG 信号处理和电机驱动控制。

## Experiment

*   **有效性:** 提出的 BiLSTM-BiGRU 模型在测试集上达到 92.26% 的准确率，通过 10 折交叉验证的平均准确率为 90.13%，显著优于基线模型（XGBoost 86%、EEGNet 64%、Transformer-based 87%）。
*   **分类性能:** 模型在三个类别（静止、右手、左手运动想象）上的分类指标（Precision、Recall、F1-Score）均表现均衡，误分类率低，表明其对细微神经模式的区分能力较强。
*   **实验设置合理性:** 实验采用公开数据集，通过多模型对比和交叉验证增强结果可信度，同时提供详细的混淆矩阵和训练历史记录，展示了模型的稳定性和泛化能力。
*   **局限性:** 实验未解决轮椅停止控制问题，硬件部署（如 Raspberry Pi）仍需优化，且未充分测试真实环境下的用户变异性和噪声干扰。

## Further Thoughts

混合深度学习模型（如 BiLSTM-BiGRU）在 EEG 信号处理中的成功应用启发我们探索更多组合架构，以捕捉复杂时空特征；此外，基于运动想象的非侵入式 BCI 控制方式提示未来可以扩展到多维度控制（如速度、前后移动），并结合更强大的嵌入式设备或云端计算提升实时性和硬件兼容性。