---
title: "Are Large Brainwave Foundation Models Capable Yet? Insights from Fine-tuning"
pubDatetime: 2025-07-01T21:21:42+00:00
slug: "2025-07-brainwave-foundation-tuning"
type: "arxiv"
id: "2507.01196"
score: 0.6273878020334406
author: "grok-3-latest"
authors: ["Na Lee", "Konstantinos Barmpas", "Yannis Panagakis", "Dimitrios Adamos", "Nikolaos Laskaris", "Stefanos Zafeiriou"]
tags: ["Brainwave Modeling", "Foundation Model", "Fine-Tuning", "Parameter Efficiency", "BCI Application"]
institution: ["Imperial College London", "Cogitat", "Archimedes / Athena Research Unit", "National and Kapodistrian University of Athens", "Aristotle University of Thessaloniki"]
description: "本文通过系统微调实验评估了大型脑电波基础模型（LBMs）在脑机接口任务中的性能，揭示其当前局限性，并利用低秩适应（LoRA）技术显著减少参数量而不牺牲性能，为领域特定模型设计提供了关键见解。"
---

> **Summary:** 本文通过系统微调实验评估了大型脑电波基础模型（LBMs）在脑机接口任务中的性能，揭示其当前局限性，并利用低秩适应（LoRA）技术显著减少参数量而不牺牲性能，为领域特定模型设计提供了关键见解。 

> **Keywords:** Brainwave Modeling, Foundation Model, Fine-Tuning, Parameter Efficiency, BCI Application

**Authors:** Na Lee, Konstantinos Barmpas, Yannis Panagakis, Dimitrios Adamos, Nikolaos Laskaris, Stefanos Zafeiriou

**Institution(s):** Imperial College London, Cogitat, Archimedes / Athena Research Unit, National and Kapodistrian University of Athens, Aristotle University of Thessaloniki


## Problem Background

基础模型在自然语言处理和计算机视觉等领域取得了显著成功，但其在脑电波建模和脑机接口（BCI）领域的适用性仍不明朗。
本文旨在探究大型脑电波基础模型（LBMs）是否能在 BCI 任务中显著优于传统深度学习模型，解决当前模型效率低、参数量大以及架构和训练策略是否适合脑电波数据的问题，特别是在记忆任务和睡眠阶段分类等基准任务上的表现。

## Method

*   **模型选择与评估框架**：研究聚焦于两个先进的 LBMs：LaBraM 和 NeuroGPT。LaBraM 通过将 EEG 信号分割为通道特定的 patch 并基于神经码本进行预训练，旨在跨数据集学习；NeuroGPT 结合 EEG 编码器和 GPT 架构，采用自回归训练方式捕捉时空模式。
*   **微调策略**：包括全模型微调和参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）。全模型微调涉及对整个模型参数进行更新，而 PEFT 主要采用低秩适应（Low-Rank Adaptation, LoRA）技术，通过低秩矩阵更新减少可训练参数，同时保持预训练知识。
*   **LoRA 具体实现**：LoRA 将权重更新分解为两个低秩矩阵的乘积，仅训练这些矩阵而冻结原始权重矩阵。研究中对注意力层、全连接层和卷积层分别或组合应用 LoRA，并调整秩（rank）值以探索性能与参数效率的平衡。此外，还测试了在 LoRA 矩阵中引入 dropout 的影响。
*   **对比与消融研究**：将 LBMs 的性能与传统深度学习模型（如 EEGNet 和 EEGInception）进行对比，并在不同微调配置下（如仅微调分类头、不同层组合应用 LoRA）进行消融实验，以揭示模型架构和训练策略的局限性。

## Experiment

*   **性能表现**：实验结果显示，LBMs（如 NeuroGPT 和 LaBraM）在部分 BCI 任务（如运动任务和眼睛开闭分类）上略优于传统深度学习模型，准确率提升幅度为 0.9%-1.2%，但在某些任务（如记忆任务）上被传统模型小幅超越。NeuroGPT 在平均性能上表现最佳。
*   **效率问题**：LBMs 的参数量远高于传统模型（例如 LaBraM 有 580 万参数，而 EEGNet 仅 2394 个参数），性能提升与资源消耗不成正比，效率较低。
*   **LoRA 效果**：通过 LoRA 微调，LBMs 的可训练参数显著减少（从数百万减少到数万），且性能未明显下降。消融研究表明，同时对多个网络组件（如卷积层与注意力层或全连接层）应用 LoRA 时，性能最佳。
*   **实验设置合理性**：实验覆盖了多个 BCI 基准数据集（如 Motor、ERP、Memory、Sleep-EDF 等），采用 10 折交叉验证，确保结果稳健。数据预处理考虑了模型预训练时的输入结构，设置全面合理。
*   **局限性揭示**：当仅训练分类头而冻结其他部分时，LBMs 性能远低于传统模型（落后 8-10%），表明全模型微调或高效微调的必要性。

## Further Thoughts

论文中关于 LoRA 在脑电波基础模型中应用的探索令人启发，特别是通过消融研究揭示了跨层依赖性，提示未来模型设计应更注重领域特定的架构优化，而非简单迁移其他领域的基础模型。此外，论文提出将领域知识（如 EEG 模态特性）融入预训练和微调策略的建议，启发我们思考如何结合脑电波数据的独特时空特性，设计更高效的基础模型，甚至探索脑启发式掩码技术或多模态 EEG 数据融合的可能性。