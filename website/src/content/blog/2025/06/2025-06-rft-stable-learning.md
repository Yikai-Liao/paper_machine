---
title: "Reinforcement Fine-Tuning Enables MLLMs Learning Novel Tasks Stably"
pubDatetime: 2025-06-30T04:15:01+00:00
slug: "2025-06-rft-stable-learning"
type: "arxiv"
id: "2506.23508"
score: 0.782317323356401
author: "grok-3-latest"
authors: ["Zhihao Zhang", "Qiaole Dong", "Qi Zhang", "Jun Zhao", "Enyu Zhou", "Zhiheng Xi", "Senjie Jin", "Xiaoran Fan", "Yuhao Zhou", "Yanwei Fu", "Tao Ji", "Tao Gui", "Xuanjing Huang"]
tags: ["MLLM", "Post-Training", "Supervised Fine-Tuning", "Reinforcement Learning", "Catastrophic Forgetting"]
institution: ["Fudan University", "Shanghai Artificial Intelligence Laboratory"]
description: "本文通过拼图任务揭示强化微调（RFT）在多模态大语言模型中能稳定学习新任务并减少灾难性遗忘，强调数据分布而非算法是关键因素。"
---

> **Summary:** 本文通过拼图任务揭示强化微调（RFT）在多模态大语言模型中能稳定学习新任务并减少灾难性遗忘，强调数据分布而非算法是关键因素。 

> **Keywords:** MLLM, Post-Training, Supervised Fine-Tuning, Reinforcement Learning, Catastrophic Forgetting

**Authors:** Zhihao Zhang, Qiaole Dong, Qi Zhang, Jun Zhao, Enyu Zhou, Zhiheng Xi, Senjie Jin, Xiaoran Fan, Yuhao Zhou, Yanwei Fu, Tao Ji, Tao Gui, Xuanjing Huang

**Institution(s):** Fudan University, Shanghai Artificial Intelligence Laboratory


## Problem Background

多模态大语言模型（MLLMs）在后训练阶段（如监督微调SFT和强化微调RFT）能够有效适应下游任务，但其对预训练知识的保留问题尚未被充分研究。
论文关注‘灾难性遗忘’（Catastrophic Forgetting）现象，即模型在学习新任务时可能显著丢失已有知识。为此，作者引入拼图（Jigsaw Puzzles）作为全新任务，测试SFT和RFT在学习新知识时对已有知识的影响，旨在揭示两种方法在任务适应与知识保留间的权衡。

## Method

*   **核心思想:** 对比监督微调（SFT）和强化微调（RFT）在学习新任务时的表现，探索其对已有知识的影响，并通过数据分布和学习动力学解释差异。
*   **监督微调（SFT）:** 基于人工标注数据，通过教师强制（Teacher Forcing）方式直接优化模型输出概率，使模型快速学习新任务（如拼图）。其训练目标是最大化静态标注数据的似然，数据分布可能与模型原有输出空间不匹配，易导致遗忘。
*   **强化微调（RFT）:** 使用Group Relative Policy Optimization (GRPO)算法，通过奖励机制（如命中奖励、准确率奖励、格式奖励）驱动模型自我生成答案并强化正确输出。RFT在模型自身概率分布内采样（Rollouts），通过长期探索学习新任务，减少对已有知识的干扰。
*   **创新实验设计:** 将RFT生成的正确Rollouts作为SFT训练数据，发现能显著减少遗忘，表明数据分布是关键因素。
*   **理论分析:** 从学习动力学（Learning Dynamics）视角解释遗忘现象，指出RFT数据位于模型低困惑度（Low Perplexity）区域，与预训练知识兼容性更高，而SFT数据常位于低概率区域，易引发冲突。

## Experiment

*   **新任务学习效果:** 在拼图任务（3x3 Jigsaw Puzzles）上，SFT仅需1个epoch（约2738步）即可达到62%的准确率，而RFT（GRPO）需8-10个epoch（约2万步）达到66%，表明SFT学习速度更快，但RFT也能通过长期探索掌握新任务。
*   **已有知识保留:** SFT导致显著灾难性遗忘，尤其在Grounding任务（如RefCOCO）上性能下降高达12.6%，而在文档问答（DocVQA）和通用视觉问答（MME）上也有明显下降；RFT的下降幅度极小（RefCOCO下降约0.3-1.9%），表现出更好的稳定性。
*   **数据分布影响验证:** 使用RFT生成的正确Rollouts进行SFT训练后，遗忘大幅减少（如RefCOCO下降仅0.04%），证明数据分布而非算法是遗忘关键。
*   **实验设置合理性:** 实验基于Qwen2.5-VL-3B模型，覆盖多个能力维度（Grounding, OCR, VQA, Hallucination），数据集（如COCO 2014）和基准（如RefCOCO, MME）具有代表性，超参数（如学习率2e-5）消融研究也增强了结论可靠性。但局限性在于仅测试单一模型和3x3拼图任务，泛化性待验证。
*   **显著性:** 数据表明RFT在减少遗忘上的优势明显，尤其在高学习率下，SFT几乎导致任务性能归零，而RFT Rollouts训练的SFT仍保持较高稳定性。

## Further Thoughts

论文揭示数据分布比算法本身对遗忘的影响更大，RFT通过自我探索生成的数据与模型输出分布对齐，避免了对预训练知识的破坏。这一思路启发我们：是否可以通过设计更贴近模型分布的数据生成策略来优化SFT？此外，学习动力学的对称性分析为持续学习提供了理论基础，未来可探索如何利用这一特性设计更稳定的后训练方法，例如通过动态调整数据分布或奖励机制来平衡新任务学习与旧知识保留。