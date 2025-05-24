---
title: "Leveraging Unit Language Guidance to Advance Speech Modeling in Textless Speech-to-Speech Translation"
pubDatetime: 2025-05-21T10:05:25+00:00
slug: "2025-05-unit-language-s2st"
type: "arxiv"
id: "2505.15333"
score: 0.6566813848031448
author: "grok-3-latest"
authors: ["Yuhao Zhang", "Xiangnan Ma", "Kaiqi Kou", "Peizhuo Liu", "Weiqiao Shan", "Benyou Wang", "Tong Xiao", "Yuxin Huang", "Zhengtao Yu", "Jingbo Zhu"]
tags: ["Speech Translation", "Unit Language", "Cross-Modal Modeling", "Cross-Lingual Modeling", "Multi-Task Learning"]
institution: ["Northeastern University, Shenyang, China", "The Chinese University of Hong Kong, Shenzhen, China", "Kunming University of Science and Technology, Kunming, China", "NiuTrans Research, Shenyang, China"]
description: "本文提出Unit Language作为无文本S2ST的中间表示，通过多任务学习和任务提示建模显著提升跨模态和跨语言建模能力，性能接近文本支持模型。"
---

> **Summary:** 本文提出Unit Language作为无文本S2ST的中间表示，通过多任务学习和任务提示建模显著提升跨模态和跨语言建模能力，性能接近文本支持模型。 

> **Keywords:** Speech Translation, Unit Language, Cross-Modal Modeling, Cross-Lingual Modeling, Multi-Task Learning

**Authors:** Yuhao Zhang, Xiangnan Ma, Kaiqi Kou, Peizhuo Liu, Weiqiao Shan, Benyou Wang, Tong Xiao, Yuxin Huang, Zhengtao Yu, Jingbo Zhu

**Institution(s):** Northeastern University, Shenyang, China, The Chinese University of Hong Kong, Shenzhen, China, Kunming University of Science and Technology, Kunming, China, NiuTrans Research, Shenyang, China


## Problem Background

无文本语音到语音翻译（Textless Speech-to-Speech Translation, S2ST）面临两大核心挑战：跨模态建模（Cross-Modal, CM），即从连续语音信号中提取语言特征；以及跨语言建模（Cross-Lingual, CL），即在长序列中实现不同语言的对齐。
特别是在无文本数据支持的情况下，这两个问题尤为突出，限制了直接S2ST模型的性能。

## Method

*   **核心思想:** 提出一种类似文本的中间表示形式‘Unit Language’，通过无监督语言建模指导无文本S2ST的跨模态（CM）和跨语言（CL）建模过程，以提升翻译性能。
*   **Unit Language 构造:** 基于n-gram语言建模，将连续的语音单元（Units）合并为‘单元词’（Unit Words），形成一种无需标注数据的文本替代表示，适用于无文本书写系统的语言。
*   **模型架构改进:** 在现有S2ST架构基础上，增加两个额外解码器（Source Text Decoder 和 Target Text Decoder），分别处理源语言和目标语言的Unit Language，指导CM和CL建模。
*   **多任务学习:** 设计综合损失函数，结合源单元预测（L_SU）、目标单元预测（L_TU）、源单元语言预测（L_CM）和目标单元语言预测（L_CL）任务，通过权重参数（如α=8, β=8, γ=8）平衡各任务贡献。
*   **任务提示建模:** 针对CM和CL任务间的冲突，引入可学习的任务提示权重（Task Prompts），分别作为CM和CL任务的诱导偏置，并在特定层级切换提示以缓解干扰，同时通过均方误差损失增强提示多样性。

## Experiment

*   **性能提升:** 在VoxPopuli数据集（涉及Es-En, Fr-En, En-Es, En-Fr四种语言对）上，与强基线相比，基于Unit Language的方法平均BLEU分数提升1.2（从20.3到21.5），结合任务提示建模后进一步优化。
*   **与文本模型对比:** 无文本S2ST性能接近有文本支持的模型（如Seamless M4T），验证了Unit Language作为文本替代品的潜力。
*   **任务冲突验证:** 单独应用CM或CL任务时性能提升明显，但同时应用时提升有限甚至下降；任务提示建模有效缓解冲突，所有语言对性能均进一步提高。
*   **实验设置合理性:** 数据集覆盖多语言，数据量充足（数百小时语音）；模型基于Transformer架构，参数设置合理；评价指标采用ASR SacreBLEU，符合领域惯例；但缺乏人类评估（如语音流畅度），存在一定局限性。

## Further Thoughts

Unit Language 的设计启发我们是否可以结合语音的声学特征（如音高、语调）进一步丰富中间表示，不仅捕捉语义信息，还能保留语音风格；任务提示建模的思路是否可扩展到其他多任务学习场景，解决类似任务冲突问题；此外，无文本S2ST的成功是否意味着可以探索更多无监督方法，用于资源匮乏语言的语音翻译任务。