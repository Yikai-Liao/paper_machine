---
title: "Automatic Calibration for Membership Inference Attack on Large Language Models"
pubDatetime: 2025-05-06T10:15:05+00:00
slug: "2025-05-automatic-calibration-mia"
type: "arxiv"
id: "2505.03392"
score: 0.5810227629167952
author: "grok-3-latest"
authors: ["Saleh Zare Zade", "Yao Qiang", "Xiangyu Zhou", "Hui Zhu", "Mohammad Amin Roshani", "Prashant Khanduri", "Dongxiao Zhu"]
tags: ["LLM", "Membership Inference", "Privacy Attack", "Probability Calibration", "Model Security"]
institution: ["Wayne State University", "Oakland University"]
description: "本文提出自动校准成员推断攻击（ACMIA）框架，通过温度调整校准大型语言模型输出概率，显著提高成员推断精度并降低假阳性率，无需外部参考模型。"
---

> **Summary:** 本文提出自动校准成员推断攻击（ACMIA）框架，通过温度调整校准大型语言模型输出概率，显著提高成员推断精度并降低假阳性率，无需外部参考模型。 

> **Keywords:** LLM, Membership Inference, Privacy Attack, Probability Calibration, Model Security

**Authors:** Saleh Zare Zade, Yao Qiang, Xiangyu Zhou, Hui Zhu, Mohammad Amin Roshani, Prashant Khanduri, Dongxiao Zhu

**Institution(s):** Wayne State University, Oakland University


## Problem Background

大型语言模型（LLMs）在预训练过程中会记住训练数据，带来隐私泄露、版权侵犯和评估数据污染等伦理与安全问题。
成员推断攻击（MIA）旨在判断特定文本是否为训练数据的一部分，但现有方法在LLMs上常将非成员误判为成员（高假阳性率），或依赖额外参考模型进行校准，限制了实用性。

## Method

*   **核心思想:** 提出自动校准成员推断攻击（ACMIA）框架，通过引入可调温度（temperature）校准输出概率分布，增强成员与非成员之间的概率差距，从而提高推断精度。
*   **理论基础:** 基于最大似然估计（MLE）的洞察，认为训练样本在概率空间中接近局部最大值，通过温度调整动态优化对数似然分布。
*   **具体实现:** 设计了三种变体以适应不同模型访问权限：
    *   **AC（Automatic Calibration）:** 利用温度调整模拟模型过拟合或欠拟合状态，通过与目标模型自身的对数似然比较，判断样本是否为训练数据。
    *   **DerivAC（Derivative-based AC）:** 引入温度调整后对数概率对温度的导数，作为样本复杂度的公平评分依据，增强成员识别能力。
    *   **NormAC（Normalized AC）:** 通过归一化处理（均值和方差）校准样本难度，确保成员样本更接近局部最大值，非成员被推远。
*   **优势:** 不依赖外部参考模型，仅利用目标模型输出进行校准，降低计算成本；动态适应样本复杂性，减少假阳性和假阴性。

## Experiment

*   **有效性:** ACMIA的三个变体（AC, DerivAC, NormAC）在多个开源LLM（如Baichuan, Qwen1.5, OPT, Pythia）和三个基准数据集（WikiMIA, MIMIR, PatentMIA）上显著优于基线方法（如Loss, Min-K%, DC-PDD），AUROC最高提升至78.5（Baichuan-13B）。
*   **鲁棒性:** 在严格阈值下（如TPR@5%FPR），ACMIA仍保持高精度，显著降低假阳性和假阴性率，尤其在分布相似的数据集（如MIMIR）上表现突出。
*   **全面性:** 实验覆盖多种模型规模、语言（英文和中文）和数据场景，消融研究验证了温度调整的鲁棒性及模型规模、文本长度对性能的影响。
*   **局限性:** 温度调整需少量标记样本优化，可能在完全无监督场景下受限；对非自回归模型的适用性未探讨。

## Further Thoughts

ACMIA的温度调整机制不仅适用于成员推断攻击，还可能扩展到其他概率分布分析任务，如对抗攻击或模型蒸馏中的概率校准；此外，其作为红队工具的潜力启发我们结合差分隐私或机器遗忘技术，开发更安全的LLM训练流程，是否可以通过温度调整揭示模型记忆行为的深层模式？