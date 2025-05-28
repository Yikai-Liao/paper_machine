---
title: "VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use"
pubDatetime: 2025-05-25T18:23:39+00:00
slug: "2025-05-vtool-multimodal-reasoning"
type: "arxiv"
id: "2505.19255"
score: 0.48527388061007404
author: "grok-3-latest"
authors: ["Mingyuan Wu", "Jingcheng Yang", "Jize Jiang", "Meitang Li", "Kaizhuo Yan", "Hanchao Yu", "Minjia Zhang", "Chengxiang Zhai", "Klara Nahrstedt"]
tags: ["LLM", "VLM", "Reinforcement Learning", "Multimodal Reasoning", "Tool Use"]
institution: ["University of Illinois Urbana-Champaign", "University of Michigan Ann Arbor", "Independent Researcher"]
description: "VTool-R1 首次通过强化学习微调训练视觉-语言模型生成多模态推理链，显著提升了结构化图像理解任务中的推理能力。"
---

> **Summary:** VTool-R1 首次通过强化学习微调训练视觉-语言模型生成多模态推理链，显著提升了结构化图像理解任务中的推理能力。 

> **Keywords:** LLM, VLM, Reinforcement Learning, Multimodal Reasoning, Tool Use

**Authors:** Mingyuan Wu, Jingcheng Yang, Jize Jiang, Meitang Li, Kaizhuo Yan, Hanchao Yu, Minjia Zhang, Chengxiang Zhai, Klara Nahrstedt

**Institution(s):** University of Illinois Urbana-Champaign, University of Michigan Ann Arbor, Independent Researcher


## Problem Background

大型语言模型（LLMs）通过强化学习微调（RFT）在文本推理、自我纠错和工具使用方面取得了显著进展，但视觉-语言模型（VLMs）在多模态推理上仍局限于静态图像输入，生成的推理链仅为文本形式，缺乏真正的多模态推理能力（即在推理中动态生成和利用视觉中间步骤）；本文旨在解决这一问题，通过训练让 VLMs 学会在推理过程中交织文本和视觉推理步骤，提升多模态任务性能。

## Method

* **核心思想**：通过强化学习微调（RFT）训练 VLMs 使用外部视觉编辑工具，生成中间视觉推理步骤，并将其与文本推理链交织，形成多模态推理能力。
* **推理与工具调用**：在推理阶段，模型基于系统提示和任务目标生成初始推理（Thought 0），决定是否调用视觉编辑工具（如高亮、遮罩、绘制边界框等）生成修改后的图像；若调用工具，则通过外部 Python 环境执行工具操作，将修改后的图像作为额外输入，进入第二阶段推理，基于原始和修改图像生成最终答案；若不调用工具，则直接生成答案。
* **训练策略**：采用基于结果的奖励（outcome-based reward）设计，仅根据最终任务准确性给予奖励，避免过程奖励导致的奖励黑客问题；使用 Group Relative Policy Optimization (GRPO) 方法，通过采样一组响应计算相对优势，优化模型策略，同时通过 KL 散度正则化确保训练稳定性；训练过程模拟推理时的两阶段迭代，确保模型学会根据上下文自主决定工具使用。
* **工具集设计**：工具基于 Python 实现，专注于结构化图像理解任务，包括高亮行列、遮罩无关区域、绘制边界框等操作，模拟人类视觉注意力机制；当前实验限制为单轮工具调用。
* **关键创新**：将工具使用和多模态推理嵌入训练过程，而非仅依赖推理时提示，让模型自主学习‘何时’和‘如何’使用工具支持推理。

## Experiment

* **有效性**：在结构化图像理解任务（如 ChartQA 和 TableVQA）上，VTool-R1 显著提升了模型性能；例如，Qwen2.5-VL 3B 模型在 ChartQA 上的准确率从纯推理的 51.8% 和提示工具使用的 24.6% 提升到 64.0%，7B 模型提升到 76.2%，接近甚至部分超越 GPT-4o 的表现。
* **工具使用行为**：训练后，模型展现出上下文敏感的工具使用行为，工具调用频率和成功率在训练中波动，表明模型逐渐学会在必要时选择性使用工具，而非过度依赖或盲目调用。
* **奖励设计验证**：实验对比基于结果和基于过程的奖励，发现后者易导致模型规避工具或利用奖励漏洞，而基于结果的奖励更稳定，能有效引导推理能力提升。
* **实验设置合理性**：数据集（如 VWTQ、ChartQA）覆盖真实和合成数据，避免训练数据泄露；工具集设计针对性强，适合结构化任务；训练配置公开透明，实验可重复性高；但局限在于仅关注单轮工具调用，未探索多轮迭代，且工具集功能有限。

## Further Thoughts

VTool-R1 的多模态推理链训练思路启发了我思考是否可以将类似框架扩展到其他模态（如视频、3D 场景）或跨模态任务（如语音与图像结合），训练模型在不同模态间动态切换推理方式；此外，基于结果的奖励设计是否适用于其他强化学习任务（如自动驾驶），仅关注最终目标而非中间步骤优化；最后，模型自适应工具使用行为是否能迁移到开放环境，训练模型自主调用复杂 API 或生成代码解决未知问题。