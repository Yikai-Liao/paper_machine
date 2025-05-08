---
title: "SEFE: Superficial and Essential Forgetting Eliminator for Multimodal Continual Instruction Tuning"
pubDatetime: 2025-05-05T09:09:41+00:00
slug: "2025-05-sefe-forgetting-mitigation"
type: "arxiv"
id: "2505.02486"
score: 0.6132135881256661
author: "grok-3-latest"
authors: ["Jinpeng Chen", "Runmin Cong", "Yuzhi Zhao", "Hongzheng Yang", "Guangneng Hu", "Horace Ho Shing Ip", "Sam Kwong"]
tags: ["Multimodal Learning", "Continual Learning", "Instruction Tuning", "Forgetting Mitigation", "Parameter Efficiency"]
institution: ["City University of Hong Kong", "Shandong University", "The Chinese University of Hong Kong", "Xidian University", "Lingnan University"]
description: "本文提出 SEFE 方法，通过 ASD 范式和 RegLoRA 分别缓解多模态持续指令微调中的表面遗忘和本质遗忘，显著提升模型在持续学习中的性能。"
---

> **Summary:** 本文提出 SEFE 方法，通过 ASD 范式和 RegLoRA 分别缓解多模态持续指令微调中的表面遗忘和本质遗忘，显著提升模型在持续学习中的性能。 

> **Keywords:** Multimodal Learning, Continual Learning, Instruction Tuning, Forgetting Mitigation, Parameter Efficiency

**Authors:** Jinpeng Chen, Runmin Cong, Yuzhi Zhao, Hongzheng Yang, Guangneng Hu, Horace Ho Shing Ip, Sam Kwong

**Institution(s):** City University of Hong Kong, Shandong University, The Chinese University of Hong Kong, Xidian University, Lingnan University


## Problem Background

多模态持续指令微调（Multimodal Continual Instruction Tuning, MCIT）旨在让多模态大语言模型（MLLMs）在逐步学习新任务时避免灾难性遗忘（Catastrophic Forgetting）。
然而，现有方法未充分区分遗忘类型，导致性能不佳；论文创新性地将遗忘分为表面遗忘（Superficial Forgetting，指因后续任务回答风格影响而偏离旧任务预期格式）和本质遗忘（Essential Forgetting，指真正的知识丢失导致内容错误），并指出表面遗忘会掩盖模型真实知识状态，需优先解决以评估本质遗忘。

## Method

*   **核心思想:** 提出 SEFE（Superficial and Essential Forgetting Eliminator）方法，通过两个组件分别缓解表面遗忘和本质遗忘，确保 MLLMs 在持续学习中的性能稳定性。
*   **Answer Style Diversification (ASD) 范式:** 针对表面遗忘，将每个任务的数据集转换为五种问题类型（是/否问题、多选题、简答题、简要解释/描述题、详细解释/描述题），以减少模型对单一回答风格的偏见；具体操作是将部分数据（默认 20%）均匀转换为其他四种风格，保留原始风格的同时引入多样性，转换过程结合现有 MLLMs 生成内容和固定规则，并为每种问题类型添加响应格式提示（Response Format Prompt, RFP），确保模型适应不同任务格式要求。
*   **RegLoRA:** 针对本质遗忘，基于 LoRA（Low-Rank Adaptation）设计正则化方法；在每个任务学习后，识别 LoRA 权重更新矩阵 ΔW 中绝对值最大的关键元素（默认 top 2%），并在后续任务训练时对这些位置施加正则化损失，防止关键知识被覆盖；正则化目标是 ΔW 而非参数本身，通过 Hadamard 乘积和累积正则化掩码实现，确保在保留旧知识的同时保持学习新知识的灵活性。
*   **结合机制:** ASD 首先缓解格式偏差，使模型输出反映真实知识状态；RegLoRA 随后保护核心知识，减少内容错误，两者协同作用形成综合解决方案。

## Experiment

*   **有效性:** 实验在 CoIN 基准数据集（包含八个视觉-语言任务）上进行，SEFE 在 Truth Alignment (TA) 指标下显著优于现有方法（如 FFT、LoRA、O-LoRA、LoTA），即使这些方法结合 ASD 后性能仍低于 SEFE；ASD 单独应用时，平均提升 Mean Final Accuracy (MFN) 6.29%、Mean Average Accuracy (MAA) 20.18%、Backward Transfer (BWT) 8.36%，表明其有效缓解表面遗忘；加上 RegLoRA 后，MFN 进一步提升 10.69%，BWT 提升 9.81%，证明其对本质遗忘的缓解作用；Knowledge Capability (KC) 指标趋势一致。
*   **实验设置合理性:** 实验设计全面，涵盖多种任务类型和回答格式，引入 CoIN-ASD 基准进一步聚焦本质遗忘评估；消融研究分析了 ASD 数据转换比例（X=20 为最优）、RegLoRA 正则化元素比例（M=2 为最优）及正则化目标（ΔW 优于其他选项）的影响，参数选择基于实验优化；案例研究直观展示表面遗忘（如格式错误）和本质遗忘（如内容错误）的改进。
*   **局限性与开销:** KC 评估依赖 Qwen1.5-32B 可能存在评分偏差，实验主要基于 LLaVA-1.5 模型，未广泛验证其他 MLLMs 的泛化性；ASD 需额外数据转换，RegLoRA 增加正则化计算，但整体开销可控。

## Further Thoughts

论文将遗忘分解为表面和本质两种类型的视角启发我们，持续学习问题可能需要细化定义以设计更精准的解决方案，这一思路可扩展到其他领域如纯文本或图像任务的持续学习；ASD 范式通过数据风格多样化缓解格式偏见的策略，提示数据增强可能在跨领域迁移学习中减少风格依赖；RegLoRA 选择性保护关键参数更新的思想，可以应用于其他参数高效微调方法，探索新旧知识平衡的更通用机制。