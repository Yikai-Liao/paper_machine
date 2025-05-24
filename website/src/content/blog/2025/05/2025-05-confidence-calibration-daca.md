---
title: "Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator"
pubDatetime: 2025-05-22T13:55:39+00:00
slug: "2025-05-confidence-calibration-daca"
type: "arxiv"
id: "2505.16690"
score: 0.7885696726184813
author: "grok-3-latest"
authors: ["Beier Luo", "Shuoyuan Wang", "Yixuan Li", "Hongxin Wei"]
tags: ["LLM", "Confidence Calibration", "Post-Training", "Unsupervised Learning", "Temperature Scaling"]
institution: ["Southern University of Science and Technology", "University of Wisconsin-Madison"]
description: "本文提出无监督置信度校准方法DACA，通过仅使用一致样本对齐预训练和后训练模型的置信度分布，显著改善了大型语言模型的校准性能。"
---

> **Summary:** 本文提出无监督置信度校准方法DACA，通过仅使用一致样本对齐预训练和后训练模型的置信度分布，显著改善了大型语言模型的校准性能。 

> **Keywords:** LLM, Confidence Calibration, Post-Training, Unsupervised Learning, Temperature Scaling

**Authors:** Beier Luo, Shuoyuan Wang, Yixuan Li, Hongxin Wei

**Institution(s):** Southern University of Science and Technology, University of Wisconsin-Madison


## Problem Background

大型语言模型（LLMs）在后训练（Post-Training）后常表现出过自信问题，即对正确和错误输出都赋予过高置信度，降低了在关键应用中的可靠性；
传统的后处理校准方法依赖标注数据，而标注数据在下游任务中往往稀缺且生成成本高昂，因此论文探索如何在无标注数据的情况下以无监督方式校准后训练模型（PoLMs）的置信度。

## Method

*   **核心思想:** 利用预训练语言模型（PLMs）的良好校准特性，通过对齐后训练模型（PoLMs）和PLMs在无标注数据上的置信度分布，实现PoLMs的无监督置信度校准。
*   **初步策略:** 在无标注验证集上，通过最小化PLM和PoLM预测分布之间的KL散度（Kullback-Leibler Divergence），优化PoLM的温度参数（τ），以调整其置信度分布。
*   **问题与改进:** 直接对齐置信度会导致欠自信问题（置信度低于实际准确率），因为当PLM和PoLM预测分歧（Disagreement）时，PLM的置信度低估PoLM的准确率，导致τ被过度调高；为此，提出Disagreement-Aware Confidence Alignment (DACA)方法，仅使用一致样本（Agreement Examples，即两模型预测相同的样本）优化τ，排除分歧样本影响，确保校准更可靠。
*   **技术细节:** DACA通过定义新的损失函数，仅对一致样本计算KL散度梯度，避免分歧样本对τ的干扰；此外，方法可扩展至其他后处理校准技术，如向量缩放（Vector Scaling）和矩阵缩放（Matrix Scaling），通过类似方式优化参数。
*   **优势:** 无需标注数据，仅依赖无标注数据和PLM置信度分数，降低了校准成本，同时理论分析和实践验证了方法的有效性。

## Experiment

*   **有效性:** DACA显著提升了PoLMs的校准性能，例如在MMLU数据集上，将Gemma-3-12B-Instruct的平均预期校准误差（ECE）从23.68%降至8.60%，接近有监督温度缩放（TS，ECE为9.75%）；在MedMCQA数据集上，将GPT-4o的ECE从21.23%降至6.99%。
*   **全面性:** 实验覆盖多种开源模型（如Llama-3、Gemma-3、Qwen2.5）和API模型（如GPT-4o、DeepSeek-V3），多个数据集（MMLU、MedMCQA、MathQA、TruthfulQA），以及多选题和开放式问答任务；同时测试了不同后训练策略（如SFT、DPO、RLHF）的模型，证明方法的普适性。
*   **对比结果:** 与无监督基线（如CAPE、Elicitation）和有监督基线（TS）相比，DACA在ECE、MCE、AECE、Brier Score等指标上表现优异，尤其在无标注场景下接近有监督方法性能。
*   **合理性与局限:** 实验设置合理，数据显著；DACA需额外PLM推理步骤，增加少量计算成本，且排除分歧样本可能减少可用校准数据，但论文认为无标注数据广泛可用，这一权衡可接受。

## Further Thoughts

DACA方法揭示了无监督校准的潜力，未来可探索其他参考模型或数据来辅助校准；分歧样本的负面影响提示是否可以通过加权或单独建模其置信度分布来进一步优化校准，而非简单排除；此外，DACA对PLM选择不敏感，即使架构不同也能有效校准，这暗示校准可能依赖置信度分布的统计特性，是否可基于此开发更通用的校准框架？