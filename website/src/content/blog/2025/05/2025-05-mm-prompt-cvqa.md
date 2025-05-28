---
title: "MM-Prompt: Cross-Modal Prompt Tuning for Continual Visual Question Answering"
pubDatetime: 2025-05-26T03:21:21+00:00
slug: "2025-05-mm-prompt-cvqa"
type: "arxiv"
id: "2505.19455"
score: 0.5103555898024561
author: "grok-3-latest"
authors: ["Xu Li", "Fan Lyu"]
tags: ["LLM", "Prompt Tuning", "Cross-Modal Learning", "Continual Learning", "Modality Imbalance"]
institution: ["Northeastern University", "Chinese Academy of Sciences"]
description: "MM-Prompt 通过跨模态提示查询和恢复机制，解决了持续视觉问答中跨模态提示隔离导致的模态不平衡问题，显著提升了性能和知识保留能力。"
---

> **Summary:** MM-Prompt 通过跨模态提示查询和恢复机制，解决了持续视觉问答中跨模态提示隔离导致的模态不平衡问题，显著提升了性能和知识保留能力。 

> **Keywords:** LLM, Prompt Tuning, Cross-Modal Learning, Continual Learning, Modality Imbalance

**Authors:** Xu Li, Fan Lyu

**Institution(s):** Northeastern University, Chinese Academy of Sciences


## Problem Background

持续视觉问答（CVQA）基于预训练模型（PTMs）通过提示调优实现多模态持续学习，但现有方法多采用跨模态提示隔离策略，独立构建视觉和文本提示，导致模态不平衡问题加剧，模型倾向于依赖某一模态（通常是语言），随时间推移性能下降。
论文旨在解决这一问题，通过引入跨模态交互机制，缓解模态不平衡，提升模型在持续学习中的准确性和知识保留能力。

## Method

*   **核心思想:** 提出 MM-Prompt 框架，通过跨模态提示查询和跨模态提示恢复两个组件，在提示选择和注入阶段引入跨模态交互，缓解模态不平衡问题。
*   **跨模态提示查询（Cross-Modal Prompt Query）:** 
    *   在提示选择阶段，利用注意力机制将视觉和文本特征相互融合，生成富含跨模态信息的查询向量。
    *   具体实现上，通过残差连接保留原始模态特征，并引入可学习权重调制，控制各特征维度的贡献，确保查询向量在融合跨模态信息的同时不丢失模态特性。
    *   最终基于融合查询向量，通过相似性加权聚合选择提示，避免单一模态偏见。
*   **跨模态提示恢复（Cross-Modal Prompt Recovery）:** 
    *   在提示注入前，通过共享掩码策略对视觉和文本提示应用相同的二进制掩码，创建跨模态缺失区域，强制模态间依赖。
    *   采用分层恢复机制：首先进行模态内恢复（Intra-Modal Recovery），利用自注意力模块和轻量跨模态信号重建缺失内容，保留模态特定模式；随后进行模态间恢复（Inter-Modal Recovery），通过跨模态注意力模块和选择性增强机制，进一步融合互补信息。
    *   引入对齐损失（Alignment Loss），包括模态内损失和模态间损失，确保恢复过程中表示一致性，防止表示漂移。
*   **关键点:** 不修改预训练模型骨干，仅通过提示调优实现跨模态交互，保持计算效率，同时通过多组件损失函数（包括交叉熵损失、查询-键对齐损失等）联合优化任务性能和模态平衡。

## Experiment

*   **有效性:** 在 VQA v2 和 NExT-QA 数据集上，MM-Prompt 在所有持续学习设置（DI, CI, QI）下均显著优于现有方法。例如，在 VQA v2 的 DI 设置下，平均准确率达到 36.223%（对比次优方法 MaPLe 的 35.187%），遗忘率降至 0.447（远低于其他方法）。
*   **模态平衡性:** 通过模态融合效果（Modality Merge Effectiveness）和模态差异（Modality Difference）分析，MM-Prompt 展现出更高的跨模态整合能力和更低的模态不平衡，注意力可视化也表明其能更精准聚焦相关区域。
*   **实验设置合理性:** 实验涵盖静态图像（VQA v2）和视频（NExT-QA）场景，任务类型多样（QI, CI, DI），并与 9 种主流方法对比，消融研究验证了两个组件的互补性，充分体现了方法的鲁棒性。
*   **计算开销:** 推理效率与主流方法相当（0.179 秒/100 样本），略高于 Dual Prompt（0.158 秒），但性能提升和遗忘减少显著，表明额外开销合理。

## Further Thoughts

MM-Prompt 的跨模态交互机制启发我在其他多模态任务（如图像-文本生成或语音-文本理解）中探索类似提示调优策略，通过共享掩码和分层恢复增强模态协同性；此外，随机掩码的局限性提示是否可以设计自适应掩码策略，根据任务上下文动态调整掩码位置，以保留关键信息，提升模型对复杂场景的适应能力。