---
title: "Vision-Language-Action Model with Open-World Embodied Reasoning from Pretrained Knowledge"
pubDatetime: 2025-05-28T02:48:42+00:00
slug: "2025-05-chatvla-openworld-reasoning"
type: "arxiv"
id: "2505.21906"
score: 0.6413766886505003
author: "grok-3-latest"
authors: ["Zhongyi Zhou", "Yichen Zhu", "Junjie Wen", "Chaomin Shen", "Yi Xu"]
tags: ["LLM", "Proxy Model", "Distillation", "Sampling", "Reasoning", "Vision-Language Model", "Robotic Control", "Mixture of Experts", "Open-World Reasoning"]
institution: ["Midea Group", "East China Normal University"]
description: "本文提出 ChatVLA-2 模型，通过动态混合专家架构和两阶段训练策略，成功保留预训练 VLM 知识并将其转化为开放世界机器人任务中的推理和动作能力，显著提升了泛化性能。"
---

> **Summary:** 本文提出 ChatVLA-2 模型，通过动态混合专家架构和两阶段训练策略，成功保留预训练 VLM 知识并将其转化为开放世界机器人任务中的推理和动作能力，显著提升了泛化性能。 

> **Keywords:** LLM, Proxy Model, Distillation, Sampling, Reasoning, Vision-Language Model, Robotic Control, Mixture of Experts, Open-World Reasoning

**Authors:** Zhongyi Zhou, Yichen Zhu, Junjie Wen, Chaomin Shen, Yi Xu

**Institution(s):** Midea Group, East China Normal University


## Problem Background

Vision-Language-Action (VLA) 模型是机器人领域的下一代技术，但现有模型在针对特定机器人任务微调时，往往会丢失预训练 Vision-Language Models (VLMs) 的核心能力，如开放世界推理和空间理解能力。
论文旨在解决这一关键问题：如何构建一个 VLA 模型，既能保留 VLM 的预训练知识，又能将其有效转化为机器人控制中的可执行推理和动作，从而实现真正的通用化机器人基础模型。

## Method

*   **核心思想:** 提出 ChatVLA-2 模型，通过动态混合专家（Dynamic Mixture-of-Experts, MoE）架构和两阶段训练策略，保留预训练 VLM 知识，同时确保推理与机器人动作的对齐。
*   **动态 MoE 架构:** 利用动态路由策略，根据输入的视觉和文本特征选择专家模块，将多模态理解和机器人控制的特征空间解耦，避免参数冲突，同时保留共享特征（如空间推理），确保预训练知识不被破坏。
*   **推理跟随增强模块:** 通过将推理 token 投影为嵌入，并将其注入到动作专家的后半层（而非全部层），生成与推理一致的动作参数，特别针对开放世界中未见过的复杂推理场景，提升模型的泛化能力。
*   **两阶段训练策略:** 第一阶段通过图像-文本数据和机器人数据的联合训练，保留预训练多模态知识并建立与机器人动作的联系；第二阶段冻结 VLM 骨干，仅训练动作专家，确保动作输出与模型内部推理一致。
*   **实现细节:** 基于 Qwen2-VL 作为 VLM 骨干，采用 8 个专家并在推理时动态选择 2 个，训练数据包括 COCO、TextVQA 等图像-文本数据集及机器人任务数据。

## Experiment

*   **有效性:** 在数学匹配游戏的开放世界场景中，ChatVLA-2 取得了 OCR 得分 3.58/4、数学推理得分 1.73/2 和 82.7% 的任务成功率，远超其他模型（如 DexVLA 和 ChatVLA），后者在开放世界中几乎完全失败；在玩具放置任务中，对象识别得分 0.94、空间感知得分 0.88、成功率 81.4%，相比 DexVLA 提升了 3.52 倍，显示出显著的泛化能力。
*   **优越性:** 相比现有模仿学习方法（如 OpenVLA、DexVLA），ChatVLA-2 在开放世界场景中的推理和动作执行能力有质的飞跃，尤其是在处理未见过的数学方程和空间指令时。
*   **实验设置合理性:** 实验涵盖在域内和开放世界两种场景，评估指标包括成功率、推理得分和对象识别得分，数据采集通过遥操作设备完成，设置较为全面；但未在模拟环境中测试，可能限制了对模型在更复杂环境中的评估。
*   **开销:** 动态 MoE 引入了额外的计算开销，训练总成本为 340 GPU 小时，但通过动态选择专家减少了不必要的计算。

## Further Thoughts

动态 MoE 架构启发了对模型模块化的进一步思考：是否可以通过更精细的专家划分（如按任务类型或模态类型），提升模型在不同任务间的泛化能力？两阶段训练策略提示，预训练知识保留和任务特定能力学习可能需要分阶段优化，是否可以引入自适应机制，根据任务需求动态决定冻结或训练哪些模块？此外，推理跟随模块是否可以通过强化学习机制，在动作执行后根据反馈进一步优化推理与动作的对齐？