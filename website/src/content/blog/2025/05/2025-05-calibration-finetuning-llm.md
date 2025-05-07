---
title: "Restoring Calibration for Aligned Large Language Models: A Calibration-Aware Fine-Tuning Approach"
pubDatetime: 2025-05-04T05:42:51+00:00
slug: "2025-05-calibration-finetuning-llm"
type: "arxiv"
id: "2505.01997"
score: 0.8279014294555844
author: "grok-3-latest"
authors: ["Jiancong Xiao", "Bojian Hou", "Zhanliang Wang", "Ruochen Jin", "Qi Long", "Weijie J. Su", "Li Shen"]
tags: ["LLM", "Calibration", "Fine-Tuning", "Preference Alignment", "Overconfidence"]
institution: ["University of Pennsylvania"]
description: "本文通过校准感知微调方法（CFT 和 RCFT），结合理论状态划分和 EM 算法正则化，显著改善了偏好对齐后大型语言模型的校准性能，同时保持或提升模型准确率。"
---

> **Summary:** 本文通过校准感知微调方法（CFT 和 RCFT），结合理论状态划分和 EM 算法正则化，显著改善了偏好对齐后大型语言模型的校准性能，同时保持或提升模型准确率。 

> **Keywords:** LLM, Calibration, Fine-Tuning, Preference Alignment, Overconfidence

**Authors:** Jiancong Xiao, Bojian Hou, Zhanliang Wang, Ruochen Jin, Qi Long, Weijie J. Su, Li Shen

**Institution(s):** University of Pennsylvania


## Problem Background

大型语言模型（LLMs）在经过偏好对齐（如 RLHF 或 DPO）后，虽然更符合人类价值观，但校准性能显著下降，表现为预测概率与实际准确率不匹配的过自信问题。
这种现象源于偏好崩溃（preference collapse），即模型过度偏好某一选项（无论是否正确），尤其在多选题场景中导致预期校准误差（ECE）增加，影响模型在高风险领域（如医疗、法律）的可靠性。

## Method

*   **理论框架：** 论文提出了‘可校准’（calibratable）和‘不可校准’（non-calibratable）两种状态的概念，通过预期校准误差（ECE）的上下界和关键准确率阈值，分析模型校准的可能性，为后续方法设计提供了理论依据。
*   **校准感知微调（CFT）：** 针对可校准状态的模型，采用领域特定知识的监督微调（SFT），使用简单的损失函数（L_SFT1）基于输出（completion）计算损失，旨在学习模式而非具体知识，从而降低过自信而不显著改变准确率。
*   **正则化校准微调（RCFT）：** 针对过度微调后进入不可校准状态的模型，设计了一种基于 EM 算法的 ECE 正则化方法，结合复杂损失（L_SFT2，考虑输入问题和输出的联合概率）和 ECE 损失（L_ECE，使用均方误差衡量置信度与目标分布差异），在提升准确率的同时控制校准误差。
*   **实现细节：** 两种方法均基于量化低秩（QLoRA）技术进行高效微调，适应有限计算资源；EM 算法用于估计目标概率分布，通过离散化置信度区间和秩保持映射调整模型输出分布以接近理想校准状态。

## Experiment

*   **有效性：** 在四个开源模型（Llama-3.1-Tulu-8B, Vicuna-7B, Olmo2-7B, Mistral-7B）上，CFT 显著降低 ECE（如 Llama-3.1-Tulu-8B 的 conf-ECE 从 0.1953 降至 0.0239），接近完美校准，同时保持或略提升准确率；RCFT 优先提升准确率（如 Llama-3.1-Tulu-8B 从 0.6228 升至 0.8341），校准误差略高于 CFT 但仍优于基线。
*   **对比基线：** 相较于温度缩放（Temperature Scaling, TS），CFT 和 RCFT 不仅改善校准，还能提升语言能力（准确率和胜率），而 TS 仅调整置信度；DPO/RLHF 基线校准性能最差，验证了问题的普遍性。
*   **设置合理性：** 实验涵盖多种模型、对齐方法（RLHF, DPO）和数据集（MMLU, MedMCQA 等），域内与域外测试验证了方法的泛化能力；校准图直观展示改进效果，柱状图更接近完美校准对角线。
*   **局限性：** RCFT 在追求高准确率时校准性能有所折衷，反映校准与性能的权衡；计算开销虽通过 QLoRA 降低，但正则化仍需额外资源。

## Further Thoughts

论文提出的可校准与不可校准状态的理论划分启发了对模型性能与校准动态关系的思考，未来可探索更细化的状态分类或引入任务类型维度指导校准策略；CFT 通过领域知识微调缓解过自信，提示是否可设计自适应知识选择机制动态调整微调数据；RCFT 的 EM 算法正则化方法展示了在优化性能同时控制校准误差的潜力，是否可扩展至其他领域（如视觉模型）或任务（如生成任务）以实现广义校准？