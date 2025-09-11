---
title: "Uncovering Scaling Laws for Large Language Models via Inverse Problems"
pubDatetime: 2025-09-09T16:53:21+00:00
slug: "2025-09-scaling-laws-inverse"
type: "arxiv"
id: "2509.07909"
score: 0.7041142536930818
author: "grok-3-latest"
authors: ["Arun Verma", "Zhaoxuan Wu", "Zijian Zhou", "Xiaoqiang Lin", "Zhiliang Chen", "Rachael Hwee Ling Sim", "Rui Qiao", "Jingtan Wang", "Nhung Bui", "Xinyuan Niu", "Wenyang Hu", "Gregory Kang Ruey Lau", "Zi-Yu Khoo", "Zitong Zhao", "Xinyi Xu", "Apivich Hemachandra", "See-Kiong Ng", "Bryan Kian Hsiang Low"]
tags: ["LLM", "Scaling Laws", "Inverse Problems", "Data Selection", "Inference Optimization"]
institution: ["Singapore-MIT Alliance for Research and Technology", "National University of Singapore", "Agency for Science, Technology and Research", "Institute of Data Science, National University of Singapore", "SAP", "CNRS@CREATE", "AI Singapore"]
description: "本文提出通过逆问题框架揭示大型语言模型的缩放定律，倡导从数据选择、推理优化和机器遗忘等角度反推最优输入成分，以更低的成本实现期望性能。"
---

> **Summary:** 本文提出通过逆问题框架揭示大型语言模型的缩放定律，倡导从数据选择、推理优化和机器遗忘等角度反推最优输入成分，以更低的成本实现期望性能。 

> **Keywords:** LLM, Scaling Laws, Inverse Problems, Data Selection, Inference Optimization

**Authors:** Arun Verma, Zhaoxuan Wu, Zijian Zhou, Xiaoqiang Lin, Zhiliang Chen, Rachael Hwee Ling Sim, Rui Qiao, Jingtan Wang, Nhung Bui, Xinyuan Niu, Wenyang Hu, Gregory Kang Ruey Lau, Zi-Yu Khoo, Zitong Zhao, Xinyi Xu, Apivich Hemachandra, See-Kiong Ng, Bryan Kian Hsiang Low

**Institution(s):** Singapore-MIT Alliance for Research and Technology, National University of Singapore, Agency for Science, Technology and Research, Institute of Data Science, National University of Singapore, SAP, CNRS@CREATE, AI Singapore


## Problem Background

大型语言模型（LLMs）由于其在数据和计算上的空前规模，训练和优化成本极高（如GPT-4成本超过1亿美元），使得通过暴力试错改进模型变得不可行。
作者提出，亟需揭示指导LLM构建的缩放定律（Scaling Laws），以在资源受限的情况下实现期望性能，解决成本效益问题。

## Method

*   **核心思想:** 将LLM优化问题形式化为逆问题（Inverse Problems），即从观测数据（如性能指标）反推最优输入成分（如数据、模型架构、训练和推理策略），以揭示缩放定律，指导高效模型构建。
*   **具体框架:** 论文从理论层面倡导以下子问题的逆向解决方式：
    *   **数据选择（Data Selection）:** 针对多模态LLM（MLLMs）和非可微评估指标（如BLEU分数），研究如何选择最具信息量的数据子集以提升性能；探索不同训练阶段（如预训练、微调、对齐）的联合数据优化策略；提出使用强化学习（如REINFORCE算法）作为非可微指标的代理梯度方法。
    *   **推理优化（Inference Optimization）:** 关注推理阶段的输入设计，如提示优化（使用黑箱优化方法如NeuralUCB）、模型配置选择（如Mixture-of-Experts路由优化）和计算资源分配（如Chain of Thought推理的计算扩展）；强调联合优化训练和推理数据以提升整体性能。
    *   **机器遗忘（Machine Unlearning, MU）:** 设计遗忘验证指标（如基于水印的检测方法）和遗忘技术（如推理时调整或利用MoE架构的稀疏性隔离数据影响），以移除有害数据或满足隐私需求，同时保持对保留数据的性能。
*   **关键点:** 逆问题方法避免了昂贵的正向训练过程，通过反向推导寻找最优输入组合，理论上适用于资源受限场景。

## Experiment

*   **有效性:** 作为立场性论文，本文未提供具体实验数据，而是通过文献综述支持其观点；例如，引用DeepSeek V3以560万美元成本实现顶尖性能，说明优化输入成分比单纯增加规模更具成本效益。
*   **合理性:** 提出的逆问题框架在理论上合理，针对的问题（如非可微指标下的数据选择、推理时计算扩展）与实际需求高度相关；但由于缺乏直接实验验证，方法的实际效果和可行性尚待后续研究确认。
*   **局限性:** 论文未提供具体实验设置或数据分析，更多是理论倡导，全面性和实践指导性有待补充。

## Further Thoughts

论文中联合优化（Joint Optimization）的概念非常启发性，提示我们是否可以通过跨阶段的动态资源分配（如预训练时分配更多数据资源，推理时分配更多计算资源）进一步提升效率；此外，基于水印的机器遗忘验证让我联想到是否可以设计‘数据指纹’机制，追踪数据在模型中的影响路径，实现更精准的遗忘或性能优化；逆问题框架是否还能扩展到其他AI领域，如强化学习中的策略优化，也是一个值得探索的方向。