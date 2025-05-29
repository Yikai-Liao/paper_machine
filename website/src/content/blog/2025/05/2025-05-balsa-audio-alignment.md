---
title: "From Alignment to Advancement: Bootstrapping Audio-Language Alignment with Synthetic Data"
pubDatetime: 2025-05-26T16:08:41+00:00
slug: "2025-05-balsa-audio-alignment"
type: "arxiv"
id: "2505.20166"
score: 0.6796585890353606
author: "grok-3-latest"
authors: ["Chun-Yi Kuan", "Hung-yi Lee"]
tags: ["LLM", "Audio-Language Alignment", "Synthetic Data", "Contrastive Learning", "Multi-Modal Learning"]
institution: ["National Taiwan University"]
description: "BALSa框架通过骨干LLM生成合成数据实现音频-语言对齐，显著提升音频感知大型语言模型的理解和推理能力，同时减少幻觉和灾难性遗忘，且数据效率极高。"
---

> **Summary:** BALSa框架通过骨干LLM生成合成数据实现音频-语言对齐，显著提升音频感知大型语言模型的理解和推理能力，同时减少幻觉和灾难性遗忘，且数据效率极高。 

> **Keywords:** LLM, Audio-Language Alignment, Synthetic Data, Contrastive Learning, Multi-Modal Learning

**Authors:** Chun-Yi Kuan, Hung-yi Lee

**Institution(s):** National Taiwan University


## Problem Background

音频感知大型语言模型（ALLMs）在从文本基础的大型语言模型（LLMs）适配到音频任务时，面临两大关键问题：一是灾难性遗忘，即在音频训练后丢失文本能力（如指令遵循），甚至出现音频幻觉（错误识别不存在的声音）；二是跨模态对齐依赖大规模任务特定问答数据，数据收集成本高昂。
论文的出发点是开发一种高效、可扩展的方法，利用骨干LLM生成合成数据以实现音频-语言对齐，减少对外部数据依赖，同时缓解灾难性遗忘和幻觉问题。

## Method

*   **核心思想：** 提出BALSa框架（Bootstrapping Audio-Language Alignment via Synthetic Data Generation from Backbone LLMs），利用骨干LLM基于音频元数据生成合成对齐数据，实现音频-语言对齐，同时避免修改骨干LLM参数以减少灾难性遗忘。
*   **数据生成：** 设计简单生成提示（如‘重复音频内容’或‘列出不存在的声音’），让骨干LLM根据音频元数据（如声音事件标签或人工标注描述）生成三种类型样本：正样本（描述音频中存在的声音）、负样本（描述不存在的声音）和组合样本（同时描述存在和不存在的声音）。
*   **训练策略：** 仅训练音频模态适配器（Audio Modality Adapter），将音频编码器（如Whisper）提取的特征映射到LLM输入空间，骨干LLM和音频编码器参数保持冻结；采用下一token预测损失进行端到端优化。
*   **LISTEN方法：** 引入对比式训练策略，通过合成负样本帮助模型区分存在和不存在的声音，减少音频幻觉。
*   **多音频扩展（BALSa-MA）：** 扩展到多音频场景，生成音频间差异解释或统一描述，增强模型对多音频输入的理解和推理能力。
*   **技术细节：** 使用LLaMA-3.1-8B-Instruct作为骨干LLM，Whisper-small作为音频编码器，适配器采用Qformer架构提取并对齐音频特征；训练采用渐进学习策略，先单音频预训练，再多音频微调。

## Experiment

*   **性能提升：** 在音频问答任务（如ClothoAQA、NonSpeech AQA）上，BALSa模型与主流基线（如Qwen2-Audio-Instruct）性能相当或更优，尤其在零样本任务（如EDANSA AQA）上提升显著；在音频推理任务（如Synonym-Hypernym Test）上，F1分数高达82.89%，领先多数基线；在音频幻觉检测上，通过负样本训练加权F1分数提升约20%；指令遵循率（IFRate）高达91.02%，遗忘率接近0，远优于其他ALLMs。
*   **数据效率：** BALSa仅使用Qwen2-Audio-Instruct所需训练数据时长的12%，却取得相当或更好性能，显示出极高效率。
*   **实验设置：** 实验涵盖多种数据集（如AudioSet-20K、AudioCaps）和基准测试（音频问答、推理、幻觉、指令遵循），评估维度全面；对比多个基线模型（如Qwen-Audio、SALMONN），并通过消融研究验证骨干LLM生成数据和负样本训练的重要性，设置合理且结果可信。
*   **多音频效果：** BALSa-MA（多音频扩展）进一步提升性能，如在SAKURA多跳推理任务上准确率提升近10%，表明多音频训练对理解和推理能力有显著促进作用。

## Further Thoughts

BALSa框架利用骨干LLM生成合成数据的思路启发我们可以在其他多模态任务（如图像-语言对齐）中尝试自包含的数据生成策略，减少对外部标注数据的依赖；负样本训练减少音频幻觉的方法提示在视觉或其他模态中引入类似对比学习机制可能有效增强模型对‘不存在’特征的辨别；多音频训练（BALSa-MA）提升推理能力的发现表明多实例联合学习可能是多模态模型训练的一个重要方向，尤其在需要比较或综合推理的任务中。