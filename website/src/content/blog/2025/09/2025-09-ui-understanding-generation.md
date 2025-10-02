---
title: "UI-UG: A Unified MLLM for UI Understanding and Generation"
pubDatetime: 2025-09-29T06:59:09+00:00
slug: "2025-09-ui-understanding-generation"
type: "arxiv"
id: "2509.24361"
score: 0.3870041016241289
author: "grok-3-latest"
authors: ["Hao Yang", "Weijie Qiu", "Ru Zhang", "Zhou Fang", "Ruichao Mao", "Xiaoyu Lin", "Maji Huang", "Zhaosong Huang", "Teng Guo", "Shuoyang Liu", "Hai Rao"]
tags: ["Multimodal LLM", "UI Understanding", "UI Generation", "Reinforcement Learning", "Domain-Specific Language"]
institution: ["Ant Group", "Beijing University of Posts and Telecommunications", "Zhejiang University"]
description: "UI-UG 是一个统一的MLLM，在现代复杂UI理解任务上达到SOTA性能，并在较低计算成本下实现了与更大模型相当的UI生成质量。"
---

> **Summary:** UI-UG 是一个统一的MLLM，在现代复杂UI理解任务上达到SOTA性能，并在较低计算成本下实现了与更大模型相当的UI生成质量。 

> **Keywords:** Multimodal LLM, UI Understanding, UI Generation, Reinforcement Learning, Domain-Specific Language

**Authors:** Hao Yang, Weijie Qiu, Ru Zhang, Zhou Fang, Ruichao Mao, Xiaoyu Lin, Maji Huang, Zhaosong Huang, Teng Guo, Shuoyang Liu, Hai Rao

**Institution(s):** Ant Group, Beijing University of Posts and Telecommunications, Zhejiang University


## Problem Background

随着移动互联网的发展，用户界面（UI）在用户体验和商业指标中的作用日益重要，但现代UI设计复杂性增加（高分辨率、密集元素、商业内容），给UI理解和生成带来挑战；现有通用多模态大语言模型（MLLMs）在UI领域的准确性和稳定性不足，而专门的UI模型往往只关注单一任务，因此需要一个统一的模型同时提升对复杂UI的细粒度理解能力和高质量生成能力。

## Method

* **核心思想**：构建一个统一的MLLM（UI-UG），通过联合训练UI理解和生成任务，利用两者的协同效应提升性能，同时保持较低计算成本。
* **数据准备**：收集超过3万张现代移动应用截图，设计细粒度UI分类类别（包括弹窗、功能图标等），并开发JSON格式的领域特定语言（DSL）支持UI生成，包含UI类型、层次结构和样式描述。
* **两阶段训练**：
  * **监督微调（SFT）**：基于Qwen2.5-VL-7B模型，使用视觉问答（VQA）数据集进行微调，覆盖理解任务（指代、定位）和生成任务，优化空间感知和格式一致性。
  * **强化学习优化**：对理解任务采用Group Relative Policy Optimization (GRPO)，通过分类准确性和IoU奖励提升指代和定位性能；对生成任务采用Direct Preference Optimization (DPO)，基于偏好数据集优化生成质量和视觉稳定性。
* **DSL渲染与交互**：DSL支持动态数据绑定和实时渲染，结合Tailwind CSS样式描述，确保生成UI的可执行性和交互性。
* **关键点**：不修改基础模型结构，仅通过数据增强和训练策略优化，同时利用任务间协同效应提升整体效果。

## Experiment

* **有效性**：在UI理解任务上，UI-UG达到最先进（SOTA）性能，指代任务分类准确率高达0.974，定位任务mAP为0.559，显著优于通用MLLMs（如GPT-4o、Claude 3.7 Sonnet）和同类UI专用模型（如Ferret-UI2）；在UI生成任务中，评分（42.02）接近更大模型（如Qwen2.5-VL-72B的42.15），但计算成本大幅降低。
* **优越性**：通过强化学习（GRPO和DPO），理解任务性能提升明显（mAP提升4.6%），生成质量提升显著（评分提升15.9%）；推理速度快，平均5.2秒完成生成，量化后低于2秒，适合实时应用。
* **实验设置**：涵盖现代复杂UI数据集、多种任务和评价指标（分类准确率、IoU、视觉相似度），并通过消融研究验证联合训练和强化学习的有效性；零样本测试显示一定泛化能力，但部分任务（如captioned grounding）性能下降，提示改进空间。
* **开销**：基于7B参数模型，训练和推理成本远低于更大模型（如72B），在两块NVIDIA L20 GPU上即可高效运行。

## Further Thoughts

联合训练UI理解和生成任务的思路启发我们在其他多模态领域探索多任务协同训练范式，利用任务间知识共享提升性能；此外，DSL支持动态渲染和实时交互的设计提示我们可以在AI驱动的设计工具中引入类似机制，直接生成可执行代码，进一步提升开发效率和用户体验。