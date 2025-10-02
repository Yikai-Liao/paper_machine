---
title: "Uni-X: Mitigating Modality Conflict with a Two-End-Separated Architecture for Unified Multimodal Models"
pubDatetime: 2025-09-29T07:05:10+00:00
slug: "2025-09-uni-x-multimodal"
type: "arxiv"
id: "2509.24365"
score: 0.8008169257682246
author: "grok-3-latest"
authors: ["Jitai Hao", "Hao Liu", "Xinyan Xiao", "Qiang Huang", "Jun Yu"]
tags: ["LLM", "Multimodal Model", "Gradient Conflict", "Autoregressive", "Semantic Fusion"]
institution: ["Harbin Institute of Technology, Shenzhen", "Baidu Inc.", "Pengcheng Laboratory"]
description: "本文提出 Uni-X 架构，通过两端分离、中间共享的设计缓解统一多模态模型中的梯度冲突，使 3B 参数模型在多模态任务上媲美 7B 模型，展现出参数效率和扩展性。"
---

> **Summary:** 本文提出 Uni-X 架构，通过两端分离、中间共享的设计缓解统一多模态模型中的梯度冲突，使 3B 参数模型在多模态任务上媲美 7B 模型，展现出参数效率和扩展性。 

> **Keywords:** LLM, Multimodal Model, Gradient Conflict, Autoregressive, Semantic Fusion

**Authors:** Jitai Hao, Hao Liu, Xinyan Xiao, Qiang Huang, Jun Yu

**Institution(s):** Harbin Institute of Technology, Shenzhen, Baidu Inc., Pengcheng Laboratory


## Problem Background

统一多模态模型（UMMs）基于共享自回归变换器（AR Transformer）在处理文本和视觉等多模态任务时，因图像和文本在低层次统计特性上的根本差异，导致浅层和深层出现严重梯度冲突，影响模型性能和多模态协同性。
作者旨在设计一种新架构，缓解这一冲突，提升训练效率和任务表现。

## Method

*   **核心思想：** 提出 Uni-X，一种两端分离、中间共享的架构，通过在浅层和深层使用模态专用参数处理文本和视觉的低层次特征差异，同时在中间层共享参数进行高层次语义融合，形成‘X形’结构以缓解梯度冲突。
*   **输入处理：** 使用 VQGAN tokenizer 将图像编码为离散 token 序列（512x512 图像转为 32x32 网格，代码本大小 8192），与文本 token 统一处理，扩展基础 LLM 的词汇表和嵌入矩阵。
*   **前向传播：** 将模型层分为三部分：初始 N 层和最后 M 层为分离层，分别处理文本和视觉隐藏状态（通过掩码机制区分模态）；中间层为共享层，融合多模态表示；分离层中视觉和文本无交叉交互，确保单模态表示的鲁棒性。
*   **训练目标：** 采用自回归下一 token 预测的交叉熵损失，适用于理解和生成任务，保持框架简洁。
*   **设计优势：** 相比混合自回归-扩散框架或任务特定分支，Uni-X 避免了额外复杂性，直接针对梯度冲突的经验证据优化架构，同时保留了参数共享和跨模态协同的潜力。

## Experiment

*   **效率对比：** 在相同训练条件（28B token 数据，Qwen2.5-1.5B 基模型）下，Uni-X（9:5 配置）在 MMLU、GenEval 和 MMBench 上的平均得分（41.6）显著优于共享变换器（38.0）、MoT（34.6）等基线，训练效率（token/秒/GPU）也较高，仅次于最简单的共享变换器，展现了性能与计算成本的良好平衡。
*   **扩展性测试：** 扩展至 3B 参数、140B token 数据后，Uni-X 在文本任务（平均分 67.1）和图像生成（GenEval 得分 82）上与多个 7B AR-UMMs 相当或更优；在不依赖额外语义编码器的模型中，视觉理解表现也具竞争力。
*   **实验设置合理性：** 实验覆盖文本、图像生成和多模态理解任务，选用多个标准基准（如 MMLU、GenEval、SEEDBench），并通过消融实验优化分离层比例（9:5 最优）；定性案例和上下文学习测试进一步验证了指令跟随和跨模态推理能力。
*   **局限性：** 视觉理解性能略逊于带语义编码器的模型，可能是由于 VQ tokenizer 代码本利用率不足（仅约 38%），提示未来改进空间。

## Further Thoughts

Uni-X 的两端分离、中间共享设计启发了对多模态模型中模态特异性和共享表示平衡的思考；未来可探索动态调整分离层比例以适应不同任务或数据特性；此外，针对 VQ tokenizer 代码本利用率不足的问题，或许可以通过多层次 token 编码或改进 tokenizer 设计进一步提升视觉表示能力，减少模态冲突。