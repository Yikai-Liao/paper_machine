---
title: "Adaptability of ASR Models on Low-Resource Language: A Comparative Study of Whisper and Wav2Vec-BERT on Bangla"
pubDatetime: 2025-07-02T17:44:54+00:00
slug: "2025-07-bangla-asr-comparison"
type: "arxiv"
id: "2507.01931"
score: 0.5316642342589012
author: "grok-3-latest"
authors: ["Md Sazzadul Islam Ridoy", "Sumi Akter", "Md. Aminur Rahman"]
tags: ["ASR", "Low-Resource Language", "Self-Supervised Learning", "Fine-Tuning", "Speech Representation"]
institution: ["Ahsanullah University of Science and Technology, Dhaka, Bangladesh"]
description: "本文首次系统比较了 Wav2Vec-BERT 和 Whisper 在低资源语言 Bangla 上的 ASR 性能，证明 Wav2Vec-BERT 在准确性和资源效率上更优，为构建高效 Bangla 语音识别系统提供了实用指导。"
---

> **Summary:** 本文首次系统比较了 Wav2Vec-BERT 和 Whisper 在低资源语言 Bangla 上的 ASR 性能，证明 Wav2Vec-BERT 在准确性和资源效率上更优，为构建高效 Bangla 语音识别系统提供了实用指导。 

> **Keywords:** ASR, Low-Resource Language, Self-Supervised Learning, Fine-Tuning, Speech Representation

**Authors:** Md Sazzadul Islam Ridoy, Sumi Akter, Md. Aminur Rahman

**Institution(s):** Ahsanullah University of Science and Technology, Dhaka, Bangladesh


## Problem Background

自动语音识别（ASR）系统在低资源语言如 Bangla 上的应用面临重大挑战，主要由于缺乏高质量标注语音数据，以及 Bangla 语言复杂的正字法系统（包括变音符号、连字和区域发音差异）导致语音到文本映射困难。
本文旨在评估和比较两种先进的 ASR 模型（Wav2Vec-BERT 和 Whisper）在 Bangla 上的性能，探索如何在数据和计算资源有限的情况下构建高效的语音识别系统，以支持教育、医疗和治理等领域的应用。

## Method

*   **研究设计:** 本研究通过比较分析，评估了两种代表性 ASR 模型在 Bangla 语言上的表现：基于自监督学习的 Wav2Vec-BERT 和基于全监督学习的 Whisper（包括 Small 和 Large-v2 变体）。
*   **数据集与预处理:** 使用两个公开数据集（Mozilla Common Voice-17 和 OpenSLR，总计约 86 小时标注语音数据），对音频进行重采样（16kHz 到 8kHz 再回 16kHz）以增强鲁棒性，对文本进行规范化（如扩展缩写、转换数字为 Bangla 词语）。
*   **模型微调:** 通过超参数优化（调整学习率、训练轮数、批大小等）对模型进行微调，Wav2Vec-BERT 初始学习率为 3e-5，Whisper 为 1e-5，并采用梯度累积以优化内存使用。
*   **数据规模实验:** 将数据集划分为 2k、8k、20k、40k 和 70k 样本五个子集，评估模型对训练数据量的依赖性。
*   **硬件测试:** 在高配（NVIDIA RTX 4090）和低配（NVIDIA RTX 3060）两种硬件环境下测试，分析计算资源对训练时间和性能的影响。
*   **评估指标:** 使用词错误率（WER）、字符错误率（CER）、训练时间和计算效率作为评估标准。

## Experiment

*   **性能对比:** Wav2Vec-BERT 在所有关键指标上显著优于 Whisper，例如在 70k 样本数据集上，Wav2Vec-BERT 的 WER 为 14.42%，CER 为 2.67%，而 Whisper Large-v2 的 WER 为 28.86%，CER 为 7.47%，统计显著性测试（p < 0.05）确认了差异。
*   **资源效率:** Wav2Vec-BERT 在低配硬件上也能稳定运行，训练时间较短（如 70k 数据集上为 13:26 小时），而 Whisper（尤其是 Large-v2）在低配环境下遇到内存问题，训练时间较长（如 21:52 小时）。
*   **数据规模影响:** Wav2Vec-BERT 在数据量增加到 40k 后性能提升趋于平缓，显示出数据效率，而 Whisper 随着数据量增加持续改进，但对资源需求更高。
*   **实验合理性:** 实验设置全面，涵盖不同数据规模和硬件条件，并通过错误分析（如音素混淆）提供定性洞察；但未测试真实场景（如噪声环境）中的表现，可能限制结果的泛化性。

## Further Thoughts

自监督学习模型（如 Wav2Vec-BERT）在低资源语言上的高效表现启发我们，是否可以通过结合自监督和全监督模型的优势（如利用 Whisper 的大规模多语言知识初始化，再用 Wav2Vec-BERT 的自监督机制针对特定语言微调）来进一步提升性能？此外，针对 Bangla 语言复杂的音系和正字法特性，是否可以设计专门的预处理或特征提取模块（如针对连字和变音符号的处理）以增强模型对特定语言的适应性？