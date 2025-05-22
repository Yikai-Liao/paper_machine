---
title: "From Unaligned to Aligned: Scaling Multilingual LLMs with Multi-Way Parallel Corpora"
pubDatetime: 2025-05-20T07:43:45+00:00
slug: "2025-05-multilingual-parallel-scaling"
type: "arxiv"
id: "2505.14045"
score: 0.663584815453248
author: "grok-3-latest"
authors: ["Yingli Shen", "Wen Lai", "Shuo Wang", "Kangyang Luo", "Alexander Fraser", "Maosong Sun"]
tags: ["LLM", "Multilingual Data", "Parallel Corpora", "Pre-Training", "Instruction Tuning"]
institution: ["Tsinghua University", "Technical University of Munich", "Munich Center for Machine Learning"]
description: "本文通过构建覆盖 113 种语言的 TED2025 多路并行语料库，系统研究其在持续预训练和指令微调中对多语言大型语言模型的性能提升，显著增强了跨语言迁移和表示对齐能力。"
---

> **Summary:** 本文通过构建覆盖 113 种语言的 TED2025 多路并行语料库，系统研究其在持续预训练和指令微调中对多语言大型语言模型的性能提升，显著增强了跨语言迁移和表示对齐能力。 

> **Keywords:** LLM, Multilingual Data, Parallel Corpora, Pre-Training, Instruction Tuning

**Authors:** Yingli Shen, Wen Lai, Shuo Wang, Kangyang Luo, Alexander Fraser, Maosong Sun

**Institution(s):** Tsinghua University, Technical University of Munich, Munich Center for Machine Learning


## Problem Background

大型语言模型（LLMs）在低资源语言上的性能显著落后于高资源语言，现有基于未对齐多语言数据的持续预训练和指令微调方法无法有效捕捉跨语言语义一致性，而多路并行数据由于其对齐特性具有更大潜力来提升多语言性能。

## Method

*   **数据集构建:** 构建了一个大规模、高质量的多路并行语料库 TED2025，基于 TED 演讲，覆盖 113 种语言，最多支持 50 种语言并行对齐，相较于现有数据集在语言覆盖和领域多样性（352 个领域）上均有显著提升。
*   **持续预训练:** 利用 TED2025 对 LLMs 进行持续预训练，旨在通过多路并行数据的跨语言对齐特性增强模型的多语言流畅性和零样本跨语言迁移能力，实验控制数据规模为 500 万 token 以确保公平性。
*   **指令微调:** 设计了四种基于多路并行数据的指令微调任务，包括机器翻译（MT）、跨语言文本相似性（CLTS）、多语言文本分类（MTC）和跨语言释义（CLP），以进一步优化模型在特定多语言任务上的表现。
*   **影响因素分析:** 研究了并行度（degree of parallelism，2-40 种语言）、语言组合策略（如是否包含英语作为枢纽语言）以及不同指令微调目标对模型性能的影响，探索最优配置策略。
*   **技术细节:** 由于计算资源限制，采用 LoRA（低秩适应）进行参数高效微调，基于 LLaMA-3 和 Qwen-2.5 模型家族，实验在 8 个 NVIDIA A100 GPU 上运行。

## Experiment

*   **有效性:** 在六个多语言基准数据集（MMMLU, XCOPA, FLORES-101/200, xIFEval, SIB）上，使用多路并行数据（TED2025）训练的模型在所有任务上显著优于未对齐数据和基线模型，尤其在低资源语言上提升明显，例如 MMMLU 低资源语言准确率从 18.27% 提升至 22.48%，高资源语言从 33.72% 提升至 41.38%。
*   **零样本迁移:** 在 FLORES-200 上，多路并行模型在未见语言的翻译质量（BLEU 和 COMET 指标）显著优于其他模型，表明其语言无关表示能力更强。
*   **表示对齐:** 通过余弦相似度、CKA、检索准确率和 SVCCA 指标，证明多路并行数据显著提升了跨语言表示对齐，尤其在语言学上较远的语言对之间效果更佳。
*   **并行度影响:** 生成任务（如 FLORES）性能随并行度增加持续提升，而理解和推理任务（如 MMMLU, XCOPA）在 6-10 种语言并行时达到峰值，显示任务类型对并行度需求的差异。
*   **英语枢纽作用:** 英语作为枢纽语言在理解任务中提升性能，但在生成任务中可能阻碍直接跨语言迁移，提示任务依赖性。
*   **实验设置合理性:** 实验覆盖多种任务和语言资源水平，控制数据规模以确保公平性，并分析了并行度和语言组合等变量，设计较为全面，但因采用 LoRA 而非全参数微调可能影响结果普适性。

## Further Thoughts

多路并行数据在跨语言语义对齐上的优势为多语言模型设计提供了新思路，未来是否可以通过自动生成或挖掘更多并行数据（如利用机器翻译结合人工校对）扩展其应用？此外，并行度对不同任务的影响差异提示是否可以开发动态并行度选择机制以优化训练效果？英语作为枢纽语言的双面性是否意味着应探索多枢纽语言策略（如同时使用英语和中文）以平衡不同任务需求？