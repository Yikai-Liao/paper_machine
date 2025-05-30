---
title: "EasyDistill: A Comprehensive Toolkit for Effective Knowledge Distillation of Large Language Models"
pubDatetime: 2025-05-27T08:32:51+00:00
slug: "2025-05-easydistill-toolkit"
type: "arxiv"
id: "2505.20888"
score: 0.5968910439281802
author: "grok-3-latest"
authors: ["Chengyu Wang", "Junbing Yan", "Wenrui Cai", "Yuanhao Yue", "Jun Huang"]
tags: ["LLM", "Knowledge Distillation", "Data Synthesis", "Supervised Fine-Tuning", "Reinforcement Learning", "Preference Optimization", "Reasoning"]
institution: ["Alibaba Cloud Computing", "Shanghai Jiao Tong University"]
description: "本文提出 *EasyDistill*，一个全面的知识蒸馏工具包，简化了大型语言模型的蒸馏过程，支持多种算法和工业应用场景，并通过开源模型和数据集推动了社区发展。"
---

> **Summary:** 本文提出 *EasyDistill*，一个全面的知识蒸馏工具包，简化了大型语言模型的蒸馏过程，支持多种算法和工业应用场景，并通过开源模型和数据集推动了社区发展。 

> **Keywords:** LLM, Knowledge Distillation, Data Synthesis, Supervised Fine-Tuning, Reinforcement Learning, Preference Optimization, Reasoning

**Authors:** Chengyu Wang, Junbing Yan, Wenrui Cai, Yuanhao Yue, Jun Huang

**Institution(s):** Alibaba Cloud Computing, Shanghai Jiao Tong University


## Problem Background

大型语言模型（LLMs）在自然语言处理领域取得了显著成功，但其高计算成本和能耗限制了广泛应用。
知识蒸馏（Knowledge Distillation, KD）是一种将大型教师模型的知识转移到小型学生模型的方法，以降低资源需求，但现有工具不足以支持多样化的 KD 场景，尤其是在工业应用中，缺乏易用性和全面性。

## Method

*   **核心思想:** 开发一个全面的工具包 *EasyDistill*，简化 LLM 的知识蒸馏过程，支持黑箱和白箱场景下的多种算法和工业应用。
*   **数据合成与增强:** 利用教师模型生成合成数据，增强种子数据集的体积和多样性，特别针对指令数据和链式思维（Chain-of-Thought, CoT）数据，支持 System 1（快速直觉）和 System 2（慢速分析）模型的蒸馏。
*   **训练算法:** 
    *   **监督微调（SFT）:** 在黑箱场景下，将教师模型输出作为 ground truth 训练学生模型；在白箱场景下，通过最小化教师和学生模型 logits 分布的差异（如 Kullback-Leibler 散度）提升效果，并优化计算效率（仅考虑 top-k logits）。
    *   **强化学习（RL）:** 通过教师模型反馈训练奖励模型，采用 Proximal Policy Optimization (PPO) 和 Group Relative Policy Optimization (GRPO) 等算法优化学生模型，增强泛化能力。
    *   **偏好排名优化:** 引入 Direct Preference Optimization (DPO) 和 Cognitive Preference Optimization (CogPO) 方法，稳定训练过程并提升 System 2 模型的推理能力。
*   **命令行接口:** 提供用户友好的 JSON 配置方式，简化 KD 流程，支持多种推理和训练加速技术（如 vLLM 和 DeepSpeed）。
*   **实用解决方案:** 通过 *EasyDistill-Recipes* 提供通用和领域特定（例如代码生成）的蒸馏方案，并发布多个数据集和模型（如 *DistilQwen* 系列）。
*   **云平台集成:** 将工具包集成到阿里巴巴云的 AI 平台（PAI），支持大规模部署。

## Experiment

*   **有效性:** 蒸馏后的 *DistilQwen* 系列模型在指令跟随和推理任务上显著优于原始 Qwen 模型，例如 *DistilQwen2.5-1.5B-Instruct* 在 AlpacaEval 2.0 上从 6.69 提升到 13.69，*DistilQwen-ThoughtX-32B* 在 AIME2024 上从 16.67 提升到 80.00。
*   **领域特定任务:** 在代码生成任务（LiveCodeBench V2）上，蒸馏模型性能提升明显（如 *Qwen2.5-7B-Code* 从 30.72 提升到 35.32），同时保持推理速度提升（2.3x）。
*   **实验设置:** 实验覆盖多种模型规模（0.5B 到 32B）和任务类型（指令跟随、推理、代码生成），数据集（如 *DistilQwen-100K*, *OmniThought*）开源，设置全面合理，但缺乏对不同算法（如 RL 和 DPO）具体贡献的消融实验。

## Further Thoughts

论文中 *EasyDistill-Recipes* 的模块化设计启发了我，是否可以通过定制化蒸馏流程，针对不同行业（如医疗、金融）设计专属小模型？此外，System 1 和 System 2 模型的区分让我思考是否可以设计混合蒸馏策略，在快速响应和深度推理间动态切换；数据合成和 CoT 增强的结合也提示是否可以通过生成高质量合成数据解决低资源语言或专业领域的数据稀缺问题。