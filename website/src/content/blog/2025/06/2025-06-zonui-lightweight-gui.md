---
title: "ZonUI-3B: A Lightweight Vision-Language Model for Cross-Resolution GUI Grounding"
pubDatetime: 2025-06-30T03:33:02+00:00
slug: "2025-06-zonui-lightweight-gui"
type: "arxiv"
id: "2506.23491"
score: 0.314178723045507
author: "grok-3-latest"
authors: ["ZongHan Hsieh", "Tzer-Jen Wei", "ShengJing Yang"]
tags: ["Vision-Language Model", "GUI Grounding", "Lightweight Model", "Cross-Resolution", "Data Diversity"]
institution: ["DeepCAT Lab, National Yang Ming Chiao Tung University"]
description: "本文提出 ZonUI-3B，一款 3B 参数的轻量级视觉-语言模型，通过跨平台多分辨率数据集和两阶段微调策略，在 GUI grounding 任务上实现与大模型相当的性能，同时保持高效训练和部署能力。"
---

> **Summary:** 本文提出 ZonUI-3B，一款 3B 参数的轻量级视觉-语言模型，通过跨平台多分辨率数据集和两阶段微调策略，在 GUI grounding 任务上实现与大模型相当的性能，同时保持高效训练和部署能力。 

> **Keywords:** Vision-Language Model, GUI Grounding, Lightweight Model, Cross-Resolution, Data Diversity

**Authors:** ZongHan Hsieh, Tzer-Jen Wei, ShengJing Yang

**Institution(s):** DeepCAT Lab, National Yang Ming Chiao Tung University


## Problem Background

图形用户界面（GUI）grounding 任务，即根据自然语言指令在屏幕上定位 UI 元素，是构建智能 GUI 代理的基础能力。
然而，大型视觉-语言模型（VLMs，参数超过 7B）计算需求高，无法在消费级硬件上高效部署，而小型模型在跨平台和多分辨率环境（尤其是高分辨率桌面界面）中泛化能力不足，面临数据稀缺、分辨率多样性不足和平台间数据不平衡的挑战。

## Method

*   **核心思想：** 开发一个轻量级视觉-语言模型 ZonUI-3B（3B 参数），通过数据构建和训练策略优化，而非模型规模增加，在跨平台、多分辨率 GUI grounding 任务中实现与大模型相当的性能。
*   **数据集构建：** 整合多个公开数据集（如 ShowUI、UGround、AMEX），构建包含 24K 样本的跨平台、多分辨率数据集，涵盖移动端、桌面端和网页界面，强调平台多样性和分辨率多样性，以解决数据稀缺和高分辨率环境适应问题。
*   **数据去冗余与采样：** 通过随机采样策略减少数据集冗余（从 120K 缩减到 16.1K 样本仍保持性能），证明数据多样性比单纯数据量更重要，提高训练效率。
*   **两阶段微调策略：**
    *   **第一阶段（跨平台预训练）：** 在混合 GUI 数据（移动端、网页、桌面端）上微调，使用平衡采样缓解平台数据不平衡，建立通用的 GUI 理解能力，识别常见组件（如按钮、图标）和自然语言指令。
    *   **第二阶段（高分辨率专门化）：** 在高分辨率子集（主要来自 UGround）上进一步微调，引入多种分辨率和屏幕比例，适应密集视觉条件和小型点击目标，提升对复杂桌面界面的鲁棒性。
*   **模型架构与训练效率：** 基于 Qwen2.5-VL-3B 架构，使用 LoRA（低秩适应）进行参数高效微调，避免修改基础模型，保持计算效率。训练在单张 RTX 4090 GPU 上完成，采用混合精度训练、梯度累积和 DeepSpeed ZeRO-2 优化等技术，确保资源可及性。

## Experiment

*   **性能表现：** ZonUI-3B 在 ScreenSpot 基准上准确率达 84.9%，在 ScreenSpot-v2 上达 86.4%，在 sub-4B 模型中表现最佳，相比 UI-TARS-2B 分别提升 2.6% 和 1.7%，甚至与部分 7B 模型（如 Aguvis-7B）相当；在高挑战性的 ScreenSpot-Pro（高分辨率专业软件界面）上准确率为 28.7%，超越 UI-TARS-2B（27.7%）和多个 7B 模型。
*   **实验设置合理性：** 实验覆盖移动端、桌面端和网页界面，特别针对高分辨率桌面环境设计了 ScreenSpot-Pro 数据集；消融研究验证了平衡采样（提升桌面性能 1.3%）、数据多样性和两阶段微调（桌面和网页性能分别提升 3.3% 和 1.8%）的有效性。
*   **计算效率：** 模型在单张 RTX 4090 GPU 上训练，资源需求低，符合轻量级设计目标，展现了小型模型在资源受限场景下的潜力。

## Further Thoughts

论文揭示了数据多样性优于数据量的原则，随机采样减少冗余数据的策略可推广至其他领域（如 NLP 或多模态任务）以降低训练成本；两阶段训练（通用学习与场景适应分离）可应用于跨域泛化任务，如医疗影像分析；此外，轻量级模型通过优化数据和训练流程接近大模型性能的思路，启发我们在边缘设备或移动端探索高效多模态模型设计。