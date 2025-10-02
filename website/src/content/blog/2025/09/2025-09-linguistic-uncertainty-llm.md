---
title: "Can Large Language Models Express Uncertainty Like Human?"
pubDatetime: 2025-09-29T02:34:30+00:00
slug: "2025-09-linguistic-uncertainty-llm"
type: "arxiv"
id: "2509.24202"
score: 0.48847389283827325
author: "grok-3-latest"
authors: ["Linwei Tao", "Yi-Fan Yeh", "Bo Kai", "Minjing Dong", "Tao Huang", "Tom A. Lamb", "Jialin Yu", "Philip H.S. Torr", "Chang Xu"]
tags: ["LLM", "Uncertainty Estimation", "Linguistic Confidence", "Calibration", "Discriminability"]
institution: ["University of Sydney, Australia", "City University of Hong Kong, Hong Kong SAR, China", "Shanghai Jiao Tong University, Shanghai, China", "University of Oxford, UK"]
description: "本文通过构建大规模不确定性表达数据集、开发轻量级置信度映射器、系统评估和微调框架，首次全面探索大型语言模型的语言信心表达能力，为不确定性估计提供高效、人性化的新路径。"
---

> **Summary:** 本文通过构建大规模不确定性表达数据集、开发轻量级置信度映射器、系统评估和微调框架，首次全面探索大型语言模型的语言信心表达能力，为不确定性估计提供高效、人性化的新路径。 

> **Keywords:** LLM, Uncertainty Estimation, Linguistic Confidence, Calibration, Discriminability

**Authors:** Linwei Tao, Yi-Fan Yeh, Bo Kai, Minjing Dong, Tao Huang, Tom A. Lamb, Jialin Yu, Philip H.S. Torr, Chang Xu

**Institution(s):** University of Sydney, Australia, City University of Hong Kong, Hong Kong SAR, China, Shanghai Jiao Tong University, Shanghai, China, University of Oxford, UK


## Problem Background

大型语言模型（LLMs）在高风险场景（如教育、医疗、法律）中应用广泛，但其过度自信的回答可能误导用户，导致严重后果；现有不确定性估计方法存在诸多限制，如需要访问模型内部参数（logits）、计算成本高或不符合自然交流习惯，因此论文提出通过语言信心（Linguistic Confidence, LC），即自然语言中的对冲表达（如‘可能’、‘大概’），作为一种轻量级、用户友好的不确定性传递方式，旨在解决 LLMs 是否能像人类一样自然表达不确定性的关键问题。

## Method

* **数据集构建**：构建首个大规模、多样化的语言不确定性表达数据集，包含 40,000 条 LLM 生成的表达，最终筛选出约 1,622 条具有至少三名人类有效标注的表达（置信度评分 0-100），通过 Amazon Mechanical Turk 平台收集人类标注，用于评估和训练。
* **轻量级置信度映射器**：开发基于 DistilRoBERTa 的轻量级模型，将对冲语言转换为数值置信度分数，训练数据包括 LLM 生成句子和人类标注表达，成本和延迟极低（MSE 仅 50.68，延迟 1.32 秒），相比 LLM API 基线具有显著优势。
* **系统性评估**：在多个问答（QA）基准（如 SimpleQA, NQ-Open, PopQA）上，首次系统研究现代 LLMs 的语言信心表现，评估校准性（ECE）和区分性（AUROC），覆盖从小型（gpt-5-mini）到大型（gpt-5, Qwen3-235B）模型，设计普通提示（LC）和增强提示（LC+，明确要求在不确定时对冲）。
* **微调框架**：提出监督微调框架，利用语义不确定性（Semantic Uncertainty）作为代理标签，生成不同置信度水平的回答句子，通过 LoRA 技术对 Qwen3-8B 模型微调，提升语言信心表达能力。

## Experiment

* **有效性**：大多数 LLMs 在普通提示（LC）下语言信心表现较差（ECE 高，AUROC 接近 50%），但增强提示（LC+）显著提升校准性和区分性，接近甚至超过语义不确定性（SU）等强基准；微调后（LC (SFT)）在 NQ-Open 上 ECE 降至 0.2837，AUROC 升至 0.7331，超越数值置信度（VNC）和 SU。
* **实验设置合理性**：实验覆盖多种模型规模（小型到大型）、开源与闭源模型，以及不同难度 QA 数据集（SimpleQA, PopQA, NQ-Open），评估指标（ECE, AUROC）全面衡量不确定性表达能力；映射器低成本低延迟特性验证了实用性。
* **局限性**：对冲语言主观性导致置信度感知差异大，研究局限于 QA 任务，未扩展到推理或多模态场景。

## Further Thoughts

论文提出将不确定性表达建模为分布而非单一平均值，这一想法启发我思考如何通过分布预测捕捉个体差异，提升细粒度表达；此外，多模态不确定性表达（如语音语调、视觉线索）以及推理任务中中间步骤的不确定性动态表达，都是值得深入探索的方向，未来可结合文本、语音和图像构建更接近人类交流的框架。