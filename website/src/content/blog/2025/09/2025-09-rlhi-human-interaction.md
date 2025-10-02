---
title: "The Era of Real-World Human Interaction: RL from User Conversations"
pubDatetime: 2025-09-29T17:50:31+00:00
slug: "2025-09-rlhi-human-interaction"
type: "arxiv"
id: "2509.25137"
score: 0.7447197899721094
author: "grok-3-latest"
authors: ["未明确列出"]
tags: ["LLM", "Personalization", "Human Interaction", "Preference Optimization", "Reinforcement Learning"]
institution: ["未明确列出"]
description: "本文提出RLHI框架，通过从真实用户交互中提取监督信号，结合用户画像和即时反馈，显著提升语言模型的个性化、指令跟随和推理能力。"
---

> **Summary:** 本文提出RLHI框架，通过从真实用户交互中提取监督信号，结合用户画像和即时反馈，显著提升语言模型的个性化、指令跟随和推理能力。 

> **Keywords:** LLM, Personalization, Human Interaction, Preference Optimization, Reinforcement Learning

**Authors:** 未明确列出

**Institution(s):** 未明确列出


## Problem Background

当前语言模型的后训练主要依赖静态的专家标注数据，这些数据缺乏真实用户的长期目标、动态需求和个性化偏好，难以适应真实使用场景中的多样性和上下文相关性；论文旨在解决如何从‘野外’用户交互中提取监督信号，以实现模型的持续学习和个性化对齐。

## Method

* **核心思想**：提出Reinforcement Learning from Human Interaction (RLHI)框架，通过真实用户交互中的自然反馈和长期用户画像，实现模型的个性化对齐和持续改进。
* **具体实现**：
  * **RLHI with User-Guided Rewrites**：针对用户对话中不满意的模型输出，利用用户的自然语言反馈（如请求更多细节）进行修订，形成偏好对（原输出为不优，修订后为优）；结合从用户长期对话历史中提取的用户画像（persona），通过persona-conditioned Direct Preference Optimization (DPO)进行偏好优化，确保输出符合用户特定需求。
  * **RLHI with User-Based Rewards**：针对无反馈的初始请求，生成多个候选响应，并利用基于用户画像的奖励模型对候选进行评分，选取最高分和最低分形成偏好对；同样通过persona-conditioned DPO（离线或在线）进行优化，适应无明确反馈的场景。
* **辅助机制**：引入质量过滤机制（如基于奖励模型的筛选）处理交互数据的噪声，确保偏好对的高质量；同时结合多轮对话上下文和用户画像，实现长期偏好与即时需求的联合建模。
* **技术细节**：训练基于Llama-3.1-8B-Instruct模型，采样参数（如温度T=0.6, top-p=0.9）和学习率经过调优，奖励模型采用Athene-RM-8B，确保评分与用户画像一致。

## Experiment

* **有效性**：在WildChat UserEval用户评价中，RLHI with User-Guided Rewrites在个性化维度提升24.3个百分点，整体用户偏好提升22.4个百分点；RLHI with User-Based Rewards在指令跟随上提升14.1个百分点，人类研究胜率达72.6%-74.0%。
* **标准基准表现**：在AlpacaEval 2.0上，RLHI with User-Based Rewards达到77.9%的长度控制胜率，超越所有基线；在ArenaHard上与最强基线持平，显示出通用任务竞争力。
* **推理能力**：在四个推理基准（OlympiadBench, Minerva, GPQA, MMLU-Pro）上，RLHI with User-Guided Rewrites将平均准确率从26.5提升至31.8，表明从轻量反馈中学习推理能力的潜力，且泛化到非训练领域。
* **实验设置合理性**：实验覆盖个性化、指令跟随和推理任务，数据来源包括真实用户交互（WildChat）和合成推理对话（基于PRM800K）；消融研究验证了用户多样性、质量过滤和RL优于SFT的重要性，设置全面且对比充分。
* **显著性与局限**：提升效果显著，尤其在个性化维度，但多轮交互中对后续轮次反馈的处理仍有改进空间；计算开销主要来自候选生成和奖励模型评分，但未详细报告具体成本。

## Further Thoughts

论文提出的从‘野外’用户交互中持续学习的理念启发我们思考在线学习循环的潜力，即模型在部署后通过实时交互不断改进，而非依赖固定训练数据；此外，结合用户画像和即时反馈的偏好优化方法，为处理多用户、多偏好场景提供了新思路，未来可探索如何在隐私保护的前提下更高效地建模用户画像；质量过滤对噪声数据的处理也值得关注，可应用于其他用户生成内容的学习场景，如社交媒体或客服对话。