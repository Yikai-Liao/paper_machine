---
title: "RICL: Adding In-Context Adaptability to Pre-Trained Vision-Language-Action Models"
pubDatetime: 2025-08-04T05:01:11+00:00
slug: "2025-08-ricl-vla-adaptation"
type: "arxiv"
id: "2508.02062"
score: 0.6346588390865782
author: "grok-3-latest"
authors: ["Kaustubh Sridhar", "Souradeep Dutta", "Dinesh Jayaraman", "Insup Lee"]
tags: ["Vision-Language-Action Model", "In-Context Learning", "Retrieval-Augmented Generation", "Robotics", "Adaptation"]
institution: ["University of Pennsylvania", "University of British Columbia"]
description: "本文提出 RICL 方法，通过后训练和检索增强生成机制，为预训练视觉-语言-动作模型注入上下文学习能力，显著提升其在新任务上的适应性。"
---

> **Summary:** 本文提出 RICL 方法，通过后训练和检索增强生成机制，为预训练视觉-语言-动作模型注入上下文学习能力，显著提升其在新任务上的适应性。 

> **Keywords:** Vision-Language-Action Model, In-Context Learning, Retrieval-Augmented Generation, Robotics, Adaptation

**Authors:** Kaustubh Sridhar, Souradeep Dutta, Dinesh Jayaraman, Insup Lee

**Institution(s):** University of Pennsylvania, University of British Columbia


## Problem Background

视觉-语言-动作（Vision-Language-Action, VLA）模型作为机器人领域的通用基础模型，在新任务和新环境中表现出一定的零样本能力，但缺乏用户友好的改进方式。
相比之下，大型语言模型（LLMs）通过上下文学习（In-Context Learning, ICL）可以轻松通过少量示例学习新任务，而 VLA 模型由于采用模仿学习目标在较窄的数据集上训练，天然缺乏 ICL 能力。
论文旨在解决如何在预训练 VLA 模型中注入上下文学习能力，使得用户可以通过少量演示（10-20 个）快速提升模型在新任务上的表现，而无需参数微调。

## Method

*   **核心思想:** 通过‘再训练以实现上下文学习’（Retraining for In-Context Learning, RICL）方法，在预训练 VLA 模型上进行后训练（Post-Training），使其具备上下文学习（ICL）能力，并结合检索增强生成（Retrieval-Augmented Generation, RAG）机制，从少量演示数据中提取相关上下文信息，提升任务表现。
*   **具体实现步骤:**
    *   **后训练阶段:** 基于预训练 VLA 模型（如 π₀-FAST-DROID），使用一小部分‘启动’（Priming）演示数据（约 400 个演示，涵盖 20 个任务），对模型的语言模型部分（LLM）进行微调，而图像编码器保持冻结。训练输入包括查询状态（图像、语言指令、状态）和从其他演示中检索到的邻近状态-动作对，上下文按距离排序（最近的邻居在最左）。
    *   **检索与上下文构建:** 在部署时，使用现成的 DINO-v2 图像编码器对查询图像进行嵌入，通过 L2 距离从检索缓冲区（包含 10-20 个任务特定演示）中找到最相似的状态-动作对，构建上下文。
    *   **动作预测与插值:** 通过距离加权插值机制，结合检索到的动作和语言模型预测的动作，生成最终动作序列。插值公式为 π_RICL = e^(-λd) * one-hot(a') + (1 - e^(-λd)) * σ(π_θ)，其中 d 为查询与最近邻居的图像嵌入距离，λ 为超参数。
    *   **进一步微调（可选）:** 如果需要更高性能，可以在任务特定演示数据上进一步微调 RICL-VLA，仍然采用 RICL 目标（即检索增强的微调），以提升表现。
*   **关键特点:** 不改变模型架构，仅通过上下文调整实现适应；检索时间小于 1 毫秒，计算开销低（运行时间仅为基线模型的 1.33 倍）。

## Experiment

*   **有效性:** RICL-π₀-FAST-DROID 在整体任务成功率上从基线模型的 2.5% 提升至 31.25%，检查点完成率从 21.25% 提升至 83.75%，尤其在语言 grounding 和动作适应性上表现突出（如在‘pick up the pokeball’任务中正确识别目标对象）。
*   **进一步微调效果:** 在任务特定数据上微调后，成功率进一步提升至 61.67%，远超基线模型微调后的 31.67%，显示出更高的参数效率。
*   **实验设置合理性:** 实验任务设计全面，涵盖未见对象、新颖动作和新场景（如厨房水槽区域），测试包括 10 次随机初始位置和方向的 rollout，确保结果稳健性；消融实验表明至少需要 10 个演示才能有效提升性能。
*   **局限性:** RICL 无法处理与基线模型能力差距过大的任务（如网球正手击球），且对相机视角或场景重大变化适应性有限。

## Further Thoughts

RICL 将检索增强生成（RAG）与上下文学习（ICL）结合的思路非常具有启发性，不仅适用于 VLA 模型，也可能推广到其他需要少样本适应的领域（如强化学习或多模态任务）；动作插值机制平衡了上下文依赖和模型自主性，值得在其他动作预测场景中探索；此外，后训练注入新能力的思路为改进现有预训练模型提供了低成本、高效的范式。发散性思考：是否可以通过多模态嵌入（而不仅是图像嵌入）提升检索质量？是否可以结合人类视频数据减少对遥控演示的依赖？