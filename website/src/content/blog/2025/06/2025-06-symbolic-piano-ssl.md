---
title: "Scaling Self-Supervised Representation Learning for Symbolic Piano Performance"
pubDatetime: 2025-06-30T14:00:14+00:00
slug: "2025-06-symbolic-piano-ssl"
type: "arxiv"
id: "2506.23869"
score: 0.6521193156554934
author: "grok-3-latest"
authors: ["Louis Bradshaw", "Honglu Fan", "Alexander Spangher", "Stella Biderman", "Simon Colton"]
tags: ["Self-Supervised Learning", "Symbolic Music", "Transformer Model", "Contrastive Learning", "Music Generation"]
institution: ["Queen Mary University of London", "University of Southern California", "University of Geneva", "EleutherAI"]
description: "本文通过大规模自监督学习预训练一个自回归变换器模型，利用绝对起始时间分词和对比学习，在符号化钢琴音乐的生成和表征学习任务上取得显著成果，为数据受限场景下的音乐建模提供了通用基础。"
---

> **Summary:** 本文通过大规模自监督学习预训练一个自回归变换器模型，利用绝对起始时间分词和对比学习，在符号化钢琴音乐的生成和表征学习任务上取得显著成果，为数据受限场景下的音乐建模提供了通用基础。 

> **Keywords:** Self-Supervised Learning, Symbolic Music, Transformer Model, Contrastive Learning, Music Generation

**Authors:** Louis Bradshaw, Honglu Fan, Alexander Spangher, Stella Biderman, Simon Colton

**Institution(s):** Queen Mary University of London, University of Southern California, University of Geneva, EleutherAI


## Problem Background

符号化音乐（Symbolic Music）研究受限于数据获取的困难，难以像文本或图像领域那样构建大规模通用模型。
论文旨在探索自监督学习（Self-Supervised Learning, SSL）在符号化钢琴音乐建模中的潜力，利用自动音乐转录技术生成的大规模数据集（如 Aria-MIDI，约10万小时），解决如何通过预训练构建通用模型，既能生成连贯的音乐续篇，又能通过表征学习适应下游任务（如分类、嵌入生成）的问题。

## Method

*   **核心思想:** 通过自监督学习在大规模符号化钢琴音乐数据上预训练一个自回归变换器模型，作为通用基础模型，支持生成建模和表征学习任务。
*   **MIDI 分词设计:** 提出了一种新的分词方案，使用绝对起始时间（Absolute Onset Times）而非相对时间偏移（Time-Shift Tokens），以避免变换器模型在处理中长期时间依赖时的累积误差；时间分辨率为10毫秒，音符速度离散化为12个区间，支持多轨 MIDI 处理。
*   **预训练阶段:** 在 Aria-MIDI 数据集的精选子集（约6万小时）上，使用下一令牌预测（Next-Token Prediction）任务预训练一个基于 LLaMa 3.2 架构的变换器模型（调整为6.5亿参数），通过数据增强（如随机转调、速度调整）提升泛化能力。
*   **生成微调:** 在高质量子集上进行单轮微调，优化模型生成钢琴音乐续篇的能力，引入特殊令牌控制生成结束。
*   **对比表征学习:** 基于 SimCLR 框架进行二次微调，生成音乐嵌入（Embeddings）；通过从同一文件提取不同片段作为正样本、不同文件作为负样本，优化对比损失（NT-Xent），使嵌入捕捉音乐的高层语义（如流派、作曲家、风格）。

## Experiment

*   **生成建模效果:** 通过人类听力测试评估生成的45秒钢琴续篇的音乐连贯性，模型显著优于符号化基线模型（如 Anticipatory Music Transformer，胜率38:6，p-value 9.43e-7），并与专有音频模型（如 Suno 3.5）及人类创作的真实续篇在偏好上无显著差异，表明大规模预训练显著提升了生成质量。
*   **表征学习效果:** 在多个分类任务（如流派、作曲家、音乐时期）上，模型嵌入通过线性探针实验取得最先进性能（如 Composer 任务准确率90.50%），监督微调后进一步提升（如 Composer 任务准确率96.30%）；少样本学习（n=100）仍表现出色（如 Genre 任务准确率89.50%）。
*   **实验设置合理性:** 实验设计全面，涵盖生成和分类任务，与符号化和音频基线模型对比；消融实验验证了预训练对对比学习的重要性（无预训练时性能下降明显）；但存在局限性，如未包含某些闭源模型（如 AudioLM）及部分符号化模型因分词不兼容未纳入比较。

## Further Thoughts

论文提出的绝对起始时间分词方案避免了时间累积误差，这一思路可推广至其他需要精确时间依赖的序列建模任务（如视频帧分析）；此外，预训练与对比学习的结合揭示了自监督表征在数据受限场景下的迁移潜力，启发我们探索跨模态表征学习（如符号化音乐与音频结合）以进一步提升音乐理解能力；构建符号化音乐通用模型的愿景也为未来研究提供了方向，可能推动音乐生成、检索和分类的统一框架。