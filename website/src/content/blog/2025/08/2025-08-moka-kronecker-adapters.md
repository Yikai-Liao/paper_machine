---
title: "MoKA: Mixture of Kronecker Adapters"
pubDatetime: 2025-08-05T14:58:14+00:00
slug: "2025-08-moka-kronecker-adapters"
type: "arxiv"
id: "2508.03527"
score: 0.8004972402456431
author: "grok-3-latest"
authors: ["Mohammadreza Sadeghi", "Mahsa Ghazvini Nejad", "MirHamed Jafarzadeh Asl", "Yu Gu", "Yuanhao Yu", "Masoud Asgharian", "Vahid Partovi Nia"]
tags: ["LLM", "Parameter Efficiency", "Fine-Tuning", "Adapter", "Kronecker Product"]
institution: ["Huawei Noah’s Ark Lab", "Department of Mathematics and Statistics, McGill University"]
description: "MoKA 通过混合 Kronecker 适配器和门控机制显著提升了大型语言模型参数高效微调的表达能力和性能，同时优化硬件兼容性，以极少的参数量实现了优于传统方法的表现。"
---

> **Summary:** MoKA 通过混合 Kronecker 适配器和门控机制显著提升了大型语言模型参数高效微调的表达能力和性能，同时优化硬件兼容性，以极少的参数量实现了优于传统方法的表现。 

> **Keywords:** LLM, Parameter Efficiency, Fine-Tuning, Adapter, Kronecker Product

**Authors:** Mohammadreza Sadeghi, Mahsa Ghazvini Nejad, MirHamed Jafarzadeh Asl, Yu Gu, Yuanhao Yu, Masoud Asgharian, Vahid Partovi Nia

**Institution(s):** Huawei Noah’s Ark Lab, Department of Mathematics and Statistics, McGill University


## Problem Background

大型语言模型（LLMs）的参数高效微调（PEFT）是资源受限环境下适配模型到下游任务的关键技术。
传统低秩适配器（如 LoRA）因秩约束限制了表达能力，在复杂任务上表现不足，而 Kronecker 产品适配器虽有更高表达力，但结构假设和硬件支持不足限制了应用。
MoKA 旨在通过提升表达能力和计算效率，解决这些局限。

## Method

*   **核心思想**：提出混合 Kronecker 适配器（Mixture of Kronecker Adapters, MoKA），通过多个不同形状的 Kronecker 滤波器组合，增强参数空间的表达能力，同时利用门控机制动态加权各适配器的贡献。
*   **具体实现**：
    *   **混合适配器设计**：将权重更新建模为多个 Kronecker 产品的混合，每个 Kronecker 适配器由一对可学习矩阵（A_i 和 B_i）组成，滤波器形状多样化以捕捉不同结构模式，突破低秩或固定结构的限制。
    *   **门控机制**：通过一组可学习的门控参数（g_i），经 softmax 函数计算每个适配器的权重（α_i），使模型能根据输入和任务自适应地选择最合适的适配器组合。
    *   **硬件优化**：利用数学等式将 Kronecker 产品操作重写为标准矩阵乘法和重塑操作，避免直接计算 Kronecker 产品，从而兼容 GPU 优化的内核，提升计算效率。
    *   **轻量变体 MoKA_s**：将一个 Kronecker 因子固定为单位矩阵，形成块对角矩阵结构，利用 Transformer 注意力机制中局部 token 重要性偏见，进一步减少参数量和计算开销。
*   **关键特点**：不改变预训练模型权重，仅通过附加适配器进行任务特定调整，同时在表达力和效率之间取得平衡。

## Experiment

*   **性能提升**：在指令微调任务中，MoKA 在 LLaMA2-7B 上相比 QLoRA 和 QDoRA 分别提升 6.7% 和 3.13%（5-shot 准确率），在 LLaMA3-8B 上提升 1.67% 和 1.71%；在常识推理任务中，MoKA 平均提升 1%-3%（zero-shot 准确率），甚至超过半精度基线。
*   **参数效率**：MoKA 的可训练参数量显著减少，例如在 LLaMA2-7B 上仅为 5.2M（对比 QLoRA 的 62.2M），在 LLaMA3-8B 上为 3.9M（对比 QLoRA 的 56.6M），最高减少约 27 倍；MoKA_s 进一步减少至 2.1M-4.2M，性能接近 MoKA。
*   **实验设置合理性**：实验覆盖指令微调和常识推理两大任务类型，使用 4 位量化的 LLaMA2-7B 和 LLaMA3-8B 模型，数据集（如 MMLU、BoolQ、PIQA 等）和评估指标（5-shot 和 zero-shot 准确率）符合领域标准，验证了方法的普适性。
*   **消融实验**：门控机制显著提升性能（LLaMA2-7B 上平均提升 0.51，LLaMA3-8B 上提升 0.81），证明动态加权的重要性。

## Further Thoughts

MoKA 的混合适配器设计启发了我思考是否可以进一步探索多种适配器结构的组合（如低秩、Kronecker、稀疏等），甚至通过元学习自动优化组合策略；门控机制的成功应用也让我考虑在多任务或多模态学习中引入类似自适应机制，根据任务或输入动态调整模型行为；此外，硬件优化的思路提醒我在设计算法时始终关注实际部署需求，特别是在边缘设备上的应用。