---
title: "Bielik v3 Small: Technical Report"
pubDatetime: 2025-05-05T10:39:51+00:00
slug: "2025-05-bielik-v3-efficiency"
type: "arxiv"
id: "2505.02550"
score: 0.6680977600738562
author: "grok-3-latest"
authors: ["Krzysztof Ociepa", "Łukasz Flis", "Remigiusz Kinas", "Krzysztof Wróbel", "Adrian Gwoździej"]
tags: ["LLM", "Parameter Efficiency", "Tokenizer Optimization", "Data Quality", "Training Strategy"]
institution: ["SpeakLeash", "ACK Cyfronet AGH", "Jagiellonian University", "Azurro", "Enelpol"]
description: "本文通过创新架构设计、高质量波兰语数据处理和高效训练策略，开发了参数高效的 Bielik v3 模型（1.5B 和 4.5B），为资源受限语言的语言模型开发树立了新标杆。"
---

> **Summary:** 本文通过创新架构设计、高质量波兰语数据处理和高效训练策略，开发了参数高效的 Bielik v3 模型（1.5B 和 4.5B），为资源受限语言的语言模型开发树立了新标杆。 

> **Keywords:** LLM, Parameter Efficiency, Tokenizer Optimization, Data Quality, Training Strategy

**Authors:** Krzysztof Ociepa, Łukasz Flis, Remigiusz Kinas, Krzysztof Wróbel, Adrian Gwoździej

**Institution(s):** SpeakLeash, ACK Cyfronet AGH, Jagiellonian University, Azurro, Enelpol


## Problem Background

波兰语作为一种资源较少的语言，在自然语言处理领域面临大规模多样化数据集和高计算资源不足的挑战，导致现有大型语言模型（LLMs）在波兰语任务上的性能、通用性和可访问性受限。
本文旨在开发参数高效的波兰语专用模型（1.5B 和 4.5B 参数规模），以实现与更大模型相当的性能，同时降低计算资源需求，为资源受限语言的 AI 应用提供可行解决方案。

## Method

*   **模型架构设计**：基于 Qwen2.5 模型，通过深度扩展（Depth Up-Scaling）调整参数规模至 1.5B 和 4.5B，采用 Transformer 架构，结合自注意力机制（带因果掩码）、分组查询注意力（GQA）以降低计算复杂性、SwiGLU 激活函数以提升训练性能、旋转位置编码（RoPE）以增强位置敏感性，以及 RMSNorm 和预归一化技术以提高训练稳定性。
*   **波兰语专用分词器**：开发了自定义的 APT4 分词器，针对波兰语文本优化 token 效率，显著减少 token 数量（相较于 Mistral 和 Qwen2.5 分词器），从而支持更长的上下文处理，并通过 FOCUS 方法初始化新分词器的嵌入矩阵以保留语义信息。
*   **数据准备与预训练**：构建了一个包含 2920 亿 token（303 百万文档）的波兰语语料库，通过数据清洗（去除无关内容、匿名化个人信息）和质量评估（使用 XGBoost 分类器，准确率 95%，分为高、中、低质量类别）确保数据质量；同时加入部分英语数据以避免灾难性遗忘，并提前引入指令数据以平滑后续微调。
*   **训练策略优化**：在预训练后，采用监督微调（SFT）以适应对话任务，使用掩码 token 策略聚焦内容 token 损失，并引入自适应学习率（Adaptive Learning Rate）根据批次 token 数量动态调整学习率；在后训练阶段，测试多种偏好优化方法（如 DPO-P、SimPO），最终选择 DPO-P 以更好地对齐人类偏好；此外，使用 Group Relative Policy Optimization (GRPO) 强化学习方法，针对数学推理任务进行微调，减少计算复杂性。
*   **模型合并**：通过线性合并策略整合不同训练阶段的模型，确保性能提升的同时维持输出一致性。

## Experiment

*   **性能表现**：Bielik v3 模型在多个波兰语基准测试中表现出色，4.5B 模型在 Open PL LLM Leaderboard 上得分 45.47（基础模型）和 56.13（指令微调后），超越 Qwen2.5-7B 等更大模型；在 Polish EQ-Bench 上得分 53.58，接近 9B-12B 模型；在 Complex Polish Text Understanding Benchmark (CPTUB) 上得分 3.38，特别是在隐含意义和短语学任务中表现突出；在 Polish Medical Leaderboard 上得分 43.55%，接近 11B 模型水平；1.5B 模型在同规模中也表现优异，如 Open PL LLM Leaderboard 得分 41.36（指令微调后），接近 Qwen2.5-3B。
*   **参数效率**：与更大模型相比，Bielik v3 展现出显著的参数效率，例如 4.5B 模型性能接近甚至超过 10B 以上模型，证明了架构优化和训练策略的有效性。
*   **实验设置合理性**：实验覆盖了情感分析、问答、文本理解、医学推理等多种任务，基准测试包括 Open PL LLM Leaderboard、CPTUB、EQ-Bench 和 Polish Medical Leaderboard 等，设置全面，能够验证模型在波兰语不同场景下的能力；同时，0-shot 和 5-shot 设置进一步测试了模型的泛化能力。
*   **局限性**：在处理复杂推理任务（如 CPTUB 的 Tricky Questions）时，模型性能仍有提升空间，反映出小规模模型在复杂逻辑推理上的天然劣势。

## Further Thoughts

本文通过深度扩展和分词器优化，证明了小规模模型在特定语言上的潜力，启发我们可以在其他资源受限语言中尝试类似策略，通过定制化分词器和高质量数据精炼提升性能；此外，数据质量优先的理念（95% 分类准确率）提示未来研究可以更多关注数据精炼而非单纯扩展规模；自适应学习率和加权损失等动态调整策略也值得在多语言或多任务场景中进一步探索，以实现更高效的训练。