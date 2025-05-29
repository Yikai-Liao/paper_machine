---
title: "Parameter-Efficient Fine-Tuning with Column Space Projection"
pubDatetime: 2025-05-26T16:52:40+00:00
slug: "2025-05-pica-fine-tuning"
type: "arxiv"
id: "2505.20211"
score: 0.8442682516000996
author: "grok-3-latest"
authors: ["Junseo Hwang", "Wonguk Cho", "Taesup Kim"]
tags: ["LLM", "Parameter Efficiency", "Fine-Tuning", "Spectral Properties", "Weight Sharing"]
institution: ["Seoul National University"]
description: "本文提出 PiCa，一种基于谱特性的参数高效微调方法，通过梯度投影到预训练权重的低秩列子空间并结合权重共享，显著减少参数数量，同时实现接近 Full FT 的学习模式和优于现有 PEFT 方法的性能。"
---

> **Summary:** 本文提出 PiCa，一种基于谱特性的参数高效微调方法，通过梯度投影到预训练权重的低秩列子空间并结合权重共享，显著减少参数数量，同时实现接近 Full FT 的学习模式和优于现有 PEFT 方法的性能。 

> **Keywords:** LLM, Parameter Efficiency, Fine-Tuning, Spectral Properties, Weight Sharing

**Authors:** Junseo Hwang, Wonguk Cho, Taesup Kim

**Institution(s):** Seoul National University


## Problem Background

大型语言模型（LLMs）的完全微调（Full Fine-Tuning, Full FT）在计算和内存成本上极高，难以在资源受限环境中应用。
参数高效微调（PEFT）方法如 LoRA 虽然减少了参数更新量，但其学习行为与 Full FT 在谱特性上存在显著差异，导致性能差距。
本文旨在设计一种新的 PEFT 方法，使微调的学习模式更接近 Full FT，同时保持高参数效率。

## Method

*   **核心思想:** 提出 PiCa（Parameter-efficient Fine-tuning with Column Space Projection），通过将梯度投影到预训练权重的低秩列子空间中，使微调更新更接近 Full FT 的谱特性。
*   **理论基础:** 基于 Eckart-Young 定理和 Theorem 1，证明将权重更新投影到预训练权重顶部奇异向量构成的子空间，可以在 Frobenius 范数下获得近似最优的低秩逼近。
*   **具体实现:** 
    *   对预训练权重矩阵 *W_0* 进行奇异值分解（SVD），提取顶部 *r* 个左奇异向量 *U_r*，作为固定投影矩阵。
    *   引入一个可训练矩阵 *B*，权重更新定义为 *ΔW = U_r * B*，确保更新方向限制在谱上最重要的子空间内。
    *   在训练时，梯度更新被投影到 *U_r* 构成的列子空间中，等效于限制学习方向到最具信息量的方向，减少噪声影响。
*   **权重共享优化:** 
    *   将具有相同功能角色（如 query、key、value）的层共享同一个可训练矩阵 *B*，而每层的投影矩阵 *U_r* 仍基于各自预训练权重的 SVD 结果。
    *   这种策略显著减少可训练参数数量，同时保留层特定的预训练知识，允许使用更高秩 *r* 以提升表达能力。
*   **优势:** 相比 LoRA，PiCa 的更新在谱特性和学习模式上更接近 Full FT；相比 SVFT，PiCa 不需存储完整奇异向量，内存开销更低。

## Experiment

*   **性能提升:** 在高秩设置下，PiCa 在自然语言生成（GSM-8K, MATH）、常识推理（BoolQ, PIQA 等 8 个数据集）和自然语言理解（GLUE 基准）任务中，多数情况下取得最佳性能。例如，在 Gemma-7B 上，PiCa (r=128) 在常识推理平均准确率达 84.47%，比 LoRA (r=32) 高约 1 个百分点，比 SVFT 高约 0.5 个百分点；在 GSM-8K 上，PiCa (r=256) 准确率达 78.39%，显著优于其他 PEFT 方法。
*   **参数效率:** 通过权重共享，PiCa 显著减少可训练参数数量。例如，在 Gemma-7B 上，PiCa (r=128) 仅用 5.11M 参数，相比 LoRA (r=32) 的 68.8M 参数减少约 13 倍，但性能更优。
*   **内存效率:** 相比 SVFT，PiCa 的非可训练参数更少（113.25M vs 380.04M），训练时 GPU 内存占用减少约 25%（16.73GB vs 20.68GB）。
*   **实验设置合理性:** 实验覆盖多种任务（NLG, Commonsense Reasoning, NLU）和模型规模（Gemma-2B/7B, LLaMA-3-8B, DeBERTaV3-base），设置高低秩配置，与基线方法在相同条件下比较，确保公平性。消融研究验证了权重共享的有效性，显示其在参数预算相似的情况下性能不降反升。
*   **局限性:** PiCa 在推理时需额外进行 SVD 计算恢复投影矩阵，可能增加初始化开销，但可通过预存储投影矩阵解决。

## Further Thoughts

PiCa 利用预训练权重的谱结构指导微调更新，启发我们是否可以探索其他结构化特性（如稀疏性、对称性）设计更高效的 PEFT 方法；权重共享策略表明层间功能相似性可被有效利用，是否可以设计动态共享机制，根据任务需求自适应调整共享范围；此外，梯度投影方法是否能与其他优化技术（如正交约束、梯度裁剪）结合，进一步提升学习效率和稳定性？