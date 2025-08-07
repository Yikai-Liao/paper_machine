---
title: "RICL: Adding In-Context Adaptability to Pre-Trained Vision-Language-Action Models"
pubDatetime: 2025-08-04T05:01:11+00:00
slug: "2025-08-ricl-vla-adaptation"
type: "arxiv"
id: "2508.02062"
score: 0.6346588390865782
author: "grok-3-latest"
authors: ["Kaustubh Sridhar", "Souradeep Dutta", "Dinesh Jayaraman", "Insup Lee"]
tags: ["VLA Models", "In-Context Learning", "Retrieval-Augmented Generation", "Robotics", "Adaptation"]
institution: ["University of Pennsylvania", "University of British Columbia"]
description: "本文提出 RICL 方法，通过后训练将上下文学习能力注入预训练视觉-语言-动作（VLA）模型，使其通过少量示范即可适应新任务，并在进一步微调时显著提升性能。"
---

> **Summary:** 本文提出 RICL 方法，通过后训练将上下文学习能力注入预训练视觉-语言-动作（VLA）模型，使其通过少量示范即可适应新任务，并在进一步微调时显著提升性能。 

> **Keywords:** VLA Models, In-Context Learning, Retrieval-Augmented Generation, Robotics, Adaptation

**Authors:** Kaustubh Sridhar, Souradeep Dutta, Dinesh Jayaraman, Insup Lee

**Institution(s):** University of Pennsylvania, University of British Columbia


## Problem Background

视觉-语言-动作（VLA）模型作为机器人领域的通用基础模型，在新任务和环境中表现出一定性能，但缺乏简单的方式让终端用户通过少量示范提升其性能。
相比之下，大型语言模型（LLMs）通过上下文学习（In-Context Learning, ICL）能力可以快速适应新任务，而 VLA 模型由于采用模仿学习目标在较窄数据集上训练，天然不具备 ICL 能力。
本文旨在解决如何在预训练 VLA 模型中注入上下文学习能力，使其通过少量示范（10-20 个）快速适应新任务，包括处理未见过的对象、新颖动作和不同场景。

## Method

* **核心思想**：通过后训练（post-training）方法，将上下文学习（ICL）能力注入到预训练的 VLA 模型中，使其能够在不更新参数的情况下，通过检索增强生成（RAG）和上下文学习适应新任务。
* **后训练策略（RICL）**：基于一小部分‘启动’（priming）示范数据（约 400 个示范，涵盖 20 个任务），对预训练 VLA 模型（如 π₀-FAST-DROID）进行后训练。训练时，将查询信息（包括三视角图像、语言提示和本体感受状态）与从其他示范中检索到的邻近信息（图像、状态、动作片段）拼接为上下文输入，仅微调模型的语言模型部分（LLM），图像编码器保持冻结。
* **检索增强生成（RAG）**：在部署阶段，使用现成的图像编码器（DINO-v2）对查询图像进行嵌入，通过 L2 距离从示范缓冲区中检索最相关的 4 个邻近示范片段，放入上下文以实现上下文学习，示范按距离排序，最近的邻近示范对预测影响最大。
* **动作插值层**：预测动作时，通过距离加权插值结合检索到的动作和模型自身的输出（通过 Softmax 函数处理），确保生成的动作既利用了上下文信息，又保留了模型的预测能力，动作片段随后通过 FAST 解码器转换为可执行的机器人动作。
* **进一步微调**：如果允许参数更新，可在目标任务的示范数据上进一步微调 RICL-VLA，使用相同的交叉熵损失目标，优化查询与预测动作片段之间的匹配，以获得额外的性能提升。
* **关键特点**：RICL 不需要从头训练模型，仅通过后训练和检索机制即可实现 ICL 能力，同时保持模型的通用性和场景泛化能力。

## Experiment

* **有效性**：在多个新任务（涉及未见过的对象、新颖动作和不同场景）上，RICL-π₀-FAST-DROID 的整体任务成功率显著高于基线模型 π₀-FAST-DROID（31.25% vs 2.5%），任务检查点完成率也大幅提升（83.75% vs 21.25%），尤其在语言 grounding 和适应新动作方面表现优异。
* **进一步微调效果**：在目标任务的 20 个示范数据上进一步微调后，RICL-π₀-FAST-DROID 的成功率提升至 61.67%，远高于直接微调基线模型的 31.67%，表明 RICL 的上下文学习能力为后续微调提供了更好的起点。
* **实验设置合理性**：实验任务设计全面，涵盖语言理解、动作适应和场景泛化挑战，每个任务收集 10 个测试 rollout，随机化初始位置和方向，确保结果稳健；检索数据为每个任务提供 20 个示范，数量合理且可操作。
* **开销与局限**：RICL 的计算开销主要来自检索步骤和上下文 token 数量增加（运行时间约为基线模型的 1.33 倍），但通过 FAISS 等工具优化后影响较小；实验还验证了 RICL 不会导致基线模型能力的损失（在无任务特定示范时成功率仍为 80%）。

## Further Thoughts

RICL 通过后训练在预训练模型中注入上下文学习能力的思路，启发我们可以在其他领域探索类似方法，如在预训练模型中注入特定任务适应性或安全约束，而无需从头训练；
动作插值层结合检索与预测的机制，提示可以在其他模型中探索‘记忆’与‘推理’的动态平衡，尤其在数据稀疏场景下；
实验中观察到的潜在动作涌现（如预测不在检索数据中的动作），表明上下文学习可能激活模型隐性知识，值得进一步研究如何系统化挖掘这种能力；
此外，RICL 仅需少量示范即可适应的特性，启发我们探索利用人类示范视频或跨领域数据，进一步减少示范需求。