---
title: "Restoring Calibration for Aligned Large Language Models: A Calibration-Aware Fine-Tuning Approach"
pubDatetime: 2025-05-04T05:42:51+00:00
slug: "2025-05-calibration-finetuning-llm"
type: "arxiv"
id: "2505.01997"
score: 0.8279014294555844
author: "grok-3-latest"
authors: ["Jiancong Xiao", "Bojian Hou", "Zhanliang Wang", "Ruochen Jin", "Qi Long", "Weijie J. Su", "Li Shen"]
tags: ["LLM", "Calibration", "Fine-Tuning", "Preference Alignment", "Overconfidence"]
institution: ["University of Pennsylvania, PA, USA"]
description: "本文提出校准感知微调方法（CFT 和 RCFT），通过理论框架和领域特定微调有效恢复对齐后大型语言模型的校准性能，同时保持或提升语言能力。"
---

> **Summary:** 本文提出校准感知微调方法（CFT 和 RCFT），通过理论框架和领域特定微调有效恢复对齐后大型语言模型的校准性能，同时保持或提升语言能力。 

> **Keywords:** LLM, Calibration, Fine-Tuning, Preference Alignment, Overconfidence

**Authors:** Jiancong Xiao, Bojian Hou, Zhanliang Wang, Ruochen Jin, Qi Long, Weijie J. Su, Li Shen

**Institution(s):** University of Pennsylvania, PA, USA


## Problem Background

大型语言模型（LLMs）在经过偏好对齐（如通过强化学习从人类反馈 RLHF 或直接偏好优化 DPO）后，校准性能显著下降，表现为过自信（预测概率高于实际准确率），这在高风险领域可能导致不可靠决策。
论文旨在探究为何偏好对齐影响校准，并提出方法恢复校准性能，同时维持对齐带来的优势。

## Method

*   **核心思想:** 针对偏好对齐导致的校准下降问题，提出校准感知微调策略，通过领域特定知识学习和正则化手段，降低模型过自信，同时平衡准确率与校准性能。
*   **校准感知微调 (CFT):** 适用于可校准区间（Calibratable Regime）的模型，通过监督微调（Supervised Fine-Tuning, SFT）在领域特定数据集上训练，使用简单损失函数（L_SFT1）基于模型输出答案计算损失，学习领域模式而非具体知识，减少对错误答案的过高置信度，提升校准。
*   **正则化校准感知微调 (RCFT):** 适用于不可校准区间（Non-Calibratable Regime）的模型，即进一步微调后准确率提升但校准恶化的场景，使用复杂损失函数（L_SFT2）结合问题内容理解，同时引入基于期望校准误差（ECE）的正则化项，通过期望最大化（EM）算法估计目标概率分布，动态调整置信度以平衡准确率与校准。
*   **理论框架:** 提出可校准和不可校准区间的概念，通过 ECE 的上下界和关键准确率阈值定义，为方法设计提供理论支持，确保校准恢复的可行性与局限性分析。

## Experiment

*   **有效性:** 实验在四个开源模型（Llama-3.1-Tulu-8B, Vicuna-7B, Olmo2-7B, Mistral-7B）上进行，CFT 显著降低校准误差，例如在 Vicuna-7B 上领域内 conf-ECE 从 0.1422 降至 0.0379（下降约 73%），同时领域外准确率从 0.5233 提升至 0.6172；RCFT 优先提升准确率，例如在 Olmo2-7B 上领域内准确率从 0.6210 提升至 0.8510，但校准误差略高于 CFT。
*   **对比性:** 相较于温度缩放（Temperature Scaling），CFT 和 RCFT 不仅改善校准，还提升语言能力（准确率和胜率），而温度缩放仅调整置信度无性能提升。
*   **合理性与局限性:** 实验设置全面，涵盖多种对齐方法（RLHF, DPO）、领域内和领域外泛化测试，数据集（MMLU, MedMCQA 等）多样；消融研究验证了方法组件必要性，但未深入探讨计算成本，且领域外校准误差在部分场景仍较高，适应性有待提升。

## Further Thoughts

论文提出的可校准与不可校准区间的理论框架启发我们思考校准与准确率之间的动态权衡，是否可以通过任务自适应调整正则化权重来优化结果；此外，领域知识对校准的影响提示是否可引入外部知识库增强校准，尤其在高风险领域；EM 算法在校准中的应用也启发我们探索生成式模型是否能模拟真实概率分布，辅助不确定性量化任务。