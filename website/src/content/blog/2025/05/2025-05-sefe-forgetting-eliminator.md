---
title: "SEFE: Superficial and Essential Forgetting Eliminator for Multimodal Continual Instruction Tuning"
pubDatetime: 2025-05-05T09:09:41+00:00
slug: "2025-05-sefe-forgetting-eliminator"
type: "arxiv"
id: "2505.02486"
score: 0.6384562402881698
author: "grok-3-latest"
authors: ["Jinpeng Chen", "Runmin Cong", "Yuzhi Zhao", "Hongzheng Yang", "Guangneng Hu", "Horace Ho Shing Ip", "Sam Kwong"]
tags: ["LLM", "Multimodal Learning", "Continual Learning", "Instruction Tuning", "Regularization"]
institution: ["City University of Hong Kong", "Shandong University", "The Chinese University of Hong Kong", "Xidian University", "Lingnan University"]
description: "本文提出 SEFE 方法，通过 ASD 范式和 RegLoRA 分别解决多模态持续指令微调中的表面遗忘和本质遗忘问题，显著提升模型性能并实现最先进的遗忘缓解效果。"
---

> **Summary:** 本文提出 SEFE 方法，通过 ASD 范式和 RegLoRA 分别解决多模态持续指令微调中的表面遗忘和本质遗忘问题，显著提升模型性能并实现最先进的遗忘缓解效果。 

> **Keywords:** LLM, Multimodal Learning, Continual Learning, Instruction Tuning, Regularization

**Authors:** Jinpeng Chen, Runmin Cong, Yuzhi Zhao, Hongzheng Yang, Guangneng Hu, Horace Ho Shing Ip, Sam Kwong

**Institution(s):** City University of Hong Kong, Shandong University, The Chinese University of Hong Kong, Xidian University, Lingnan University


## Problem Background

多模态持续指令微调（MCIT）旨在让多模态大语言模型（MLLMs）逐步学习新任务而不丢失旧任务能力，但灾难性遗忘问题阻碍了这一目标。
本文创新性地将遗忘分为表面遗忘（回答风格偏离预期格式）和本质遗忘（知识内容真实丢失），指出表面遗忘会掩盖模型真实知识状态，需优先解决以准确评估本质遗忘，现有方法未充分识别这两种遗忘及其关系，导致性能不佳。

## Method

*   **核心思想:** 提出 SEFE 方法，通过两个组件分别解决表面遗忘和本质遗忘，实现 MCIT 中遗忘问题的全面缓解。
*   **Answer Style Diversification (ASD) 范式:** 针对表面遗忘，将每个任务的数据集转换为五种问题类型（是/否、多选、简短回答、简要解释/描述、详细解释/描述），以统一回答风格，减少风格偏移导致的偏见。具体方法是将部分数据（默认 20%）平均转换为其他四种格式，保留原始格式的同时引入多样性，转换过程结合规则和现有 MLLM 模型生成内容，确保训练数据风格多样化。
*   **RegLoRA:** 针对本质遗忘，基于 LoRA（低秩适应）设计正则化方法。在每个任务学习后，计算 LoRA 权重更新矩阵，识别绝对值较大的关键元素（默认 top 2%），并在后续任务训练时对这些位置施加正则化损失，防止关键知识被覆盖，同时允许其他参数自由更新以学习新信息。这种方法通过平衡知识保留和学习灵活性，有效减少本质遗忘。
*   **实现细节:** ASD 使用标准化转换流程和响应格式提示（RFP）辅助风格统一；RegLoRA 通过正则化损失与语言建模损失结合，确保训练目标兼顾新旧任务性能。

## Experiment

*   **有效性:** 在 CoIN 基准数据集（包含八个视觉-语言任务）上，SEFE 在 Truth Alignment (TA) 指标下显著优于现有方法（如 FFT、LoRA、O-LoRA、LoTA），即使这些方法结合 ASD 后也无法匹敌。ASD 单独应用时，平均最终准确率 (MFN) 提升 6.29%，平均平均准确率 (MAA) 提升 20.18%，后向转移 (BWT) 改善 8.36%；加上 RegLoRA 后，MFN 再提升 10.69%，MAA 提升 3.33%，BWT 改善 9.81%，显示对两种遗忘的显著控制。
*   **实验设置合理性:** 实验设计全面，包含多种对比方法、消融研究（如 ASD 数据转换比例 X 和 RegLoRA 正则化比例 M 的影响）以及案例分析。引入 CoIN-ASD 基准进一步聚焦本质遗忘评估。TA 指标作为主要依据，Knowledge Capability (KC) 指标因局限性（如对格式误判）仅作补充。
*   **显著性与局限:** 数据和案例研究均表明 SEFE 的提升显著，尤其在综合指标 MAA 和 BWT 上表现突出；但 KC 指标的局限性提示未来需更鲁棒的评估方式。

## Further Thoughts

将遗忘细分为表面和本质的视角可推广至其他持续学习场景，如纯文本 LLM 微调中是否也存在风格偏移问题，是否能通过类似 ASD 的数据多样化缓解？RegLoRA 的正则化思路启发动态关键参数识别方法，如基于任务相关性而非绝对值排序，或结合知识蒸馏增强效果。此外，CoIN-ASD 基准提示数据格式标准化可能是持续学习的重要方向，未来可设计通用转换框架适应多模态任务。