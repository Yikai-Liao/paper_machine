---
title: "PsyMem: Fine-grained psychological alignment and Explicit Memory Control for Advanced Role-Playing LLMs"
pubDatetime: 2025-05-19T07:45:09+00:00
slug: "2025-05-psymem-role-playing"
type: "arxiv"
id: "2505.12814"
score: 0.7450662137321731
author: "grok-3-latest"
authors: ["Xilong Cheng", "Yunxiao Qin", "Yuting Tan", "Zhengnan Li", "Ye Wang", "Hongjiang Xiao", "Yuan Zhang"]
tags: ["LLM", "Role Playing", "Psychological Modeling", "Memory Alignment", "Fine-Tuning"]
institution: ["Communication University of China", "State Key Laboratory of Media Convergence and Communication"]
description: "本文提出 PsyMem 框架，通过细粒度心理对齐和显式记忆控制，显著提升大型语言模型在角色扮演任务中的真实性和一致性，尤其在记忆对齐和人性化方面表现突出。"
---

> **Summary:** 本文提出 PsyMem 框架，通过细粒度心理对齐和显式记忆控制，显著提升大型语言模型在角色扮演任务中的真实性和一致性，尤其在记忆对齐和人性化方面表现突出。 

> **Keywords:** LLM, Role Playing, Psychological Modeling, Memory Alignment, Fine-Tuning

**Authors:** Xilong Cheng, Yunxiao Qin, Yuting Tan, Zhengnan Li, Ye Wang, Hongjiang Xiao, Yuan Zhang

**Institution(s):** Communication University of China, State Key Laboratory of Media Convergence and Communication


## Problem Background

现有基于大型语言模型（LLM）的角色扮演系统在建模角色属性和记忆一致性方面存在不足：
* **过简化的角色刻画**：依赖基本文本描述或狭窄指标，未能全面捕捉角色的内在心理特质和外在行为模式。
* **弱记忆控制**：依赖模型隐式知识或简单检索增强生成（RAG），无法动态对齐角色记忆与对话响应，尤其在普通或未知角色上表现不佳。
这些问题限制了角色扮演 LLM 在可信社会模拟、互动游戏和 AI 咨询等应用中的可靠性。

## Method

* **核心思想**：提出 PsyMem 框架，通过细粒度的心理属性建模和显式记忆控制，提升角色扮演的真实性和一致性。
* **细粒度心理属性建模**：基于现代心理学理论（如 Big Five Personality Model、Schwartz’s Theory of Basic Values），设计 26 个量化指标，涵盖内在心理属性（个性、价值观）和外在行为模式（社交与领导风格、行为决策），并结合少量文本描述，全面刻画角色。
* **显式记忆控制与对齐训练**：受海马体和认知映射机制启发，构建基于知识图谱的记忆结构，将角色、事件和关系组织为图谱；在训练阶段引入记忆对齐训练，强制模型学习基于检索到的记忆生成响应，而非依赖隐式知识；同时通过引入无关记忆噪声和角色特定知识边界（如拒绝回答超出角色认知范围的问题），增强记忆一致性和合理性。
* **两阶段训练策略**：
  * **第一阶段**：基于角色属性数据进行基础角色扮演能力训练，使用不含记忆的子数据集，建立模型对角色特质的初步理解。
  * **第二阶段**：结合记忆增强数据和角色化通用监督微调（SFT）数据，进一步训练模型，提升记忆一致性和通用能力，通过加权损失函数平衡角色特异性和通用语言理解能力，避免灾难性遗忘。
* **关键创新**：将心理学框架系统性地融入 LLM 训练，并通过显式记忆对齐提升动态适应性，同时设计角色化 SFT 数据缓解通用能力下降问题。

## Experiment

* **有效性**：在角色保真度（Character Fidelity）方面，PsyMem-Qwen 相较基线模型 Qwen2.5-7B-Instruct 提升约 3.5%，尤其在记忆对齐维度提升高达 6.2%；在角色无关能力（Character-independent Capabilities）方面，人性化（Human-likeness）指标从 64.4% 提升至 87.6%，表现接近人类对话风格。
* **优越性**：PsyMem-Qwen（仅 7B 参数）在角色保真度上超越所有基线（包括 GPT-4o 和专用角色扮演模型如 CharacterGLM），在社交与领导风格、行为决策等维度表现最佳；PsyMem-LLama 在价值观维度得分最高（81.93%）。
* **实验设置**：数据集从 539 本小说中提取 5414 个角色和 38962 段对话，规模远超现有数据集；评估数据选用 2024 年 6 月后出版的小说，避免预训练知识偏差；采用‘LLM as Judges’方法，使用 GPT-4o 评分，覆盖 500 个场景、每个场景 15 轮对话，设置全面合理。
* **消融实验**：验证了记忆对齐训练显著提升记忆一致性（从 44.2% 到 90.2%），角色化 SFT 数据有效缓解通用能力遗忘问题（如一致性从 82.2% 提升至 95.6%）。
* **开销**：训练在 4 张 A800 80G GPU 上进行，序列长度 8192，批大小 128，采用 LoRA 微调，计算成本较高，但推理时主要增加记忆检索和对齐的轻量计算。

## Further Thoughts

论文启发我们可以在 LLM 中进一步探索心理学与神经科学的结合，例如引入 Ebbinghaus 遗忘曲线设计动态记忆更新策略，提升长上下文任务的表现；此外，是否可以通过多模态数据（如图像、语音）补充角色属性建模，增强角色扮演的沉浸感？两阶段训练策略也提示我们可以在领域适应或任务迁移中设计类似方法，平衡特异性与通用性。