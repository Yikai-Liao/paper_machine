---
title: "ATGen: A Framework for Active Text Generation"
pubDatetime: 2025-06-29T17:27:48+00:00
slug: "2025-06-atgen-active-learning"
type: "arxiv"
id: "2506.23342"
score: 0.46612819391391663
author: "grok-3-latest"
authors: ["Akim Tsvigun", "Daniil Vasilev", "Ivan Tsvigun", "Ivan Lysenko", "Talgat Bektleuov", "Aleksandr Medvedev", "Uliana Vinogradova", "Nikita Severin", "Mikhail Mozikov", "Andrey Savchenko", "Rostislav Grigorev", "Ramil Kuleev", "Fedor Zhdanov", "Artem Shelmanov", "Ilya Makarov"]
tags: ["LLM", "Active Learning", "Text Generation", "Annotation Efficiency", "Fine-Tuning"]
institution: ["Research Center of the Artificial Intelligence Institute, Innopolis University", "HSE University", "Independent Researcher", "T-Technologies", "Robotics Center", "AIRI", "SB-AI-Lab", "Royal Holloway University of London", "MBZUAI"]
description: "ATGen 框架通过整合主动学习策略和支持人工与 LLM 自动标注，为自然语言生成任务提供了一个降低标注成本和提升效率的统一平台。"
---

> **Summary:** ATGen 框架通过整合主动学习策略和支持人工与 LLM 自动标注，为自然语言生成任务提供了一个降低标注成本和提升效率的统一平台。 

> **Keywords:** LLM, Active Learning, Text Generation, Annotation Efficiency, Fine-Tuning

**Authors:** Akim Tsvigun, Daniil Vasilev, Ivan Tsvigun, Ivan Lysenko, Talgat Bektleuov, Aleksandr Medvedev, Uliana Vinogradova, Nikita Severin, Mikhail Mozikov, Andrey Savchenko, Rostislav Grigorev, Ramil Kuleev, Fedor Zhdanov, Artem Shelmanov, Ilya Makarov

**Institution(s):** Research Center of the Artificial Intelligence Institute, Innopolis University, HSE University, Independent Researcher, T-Technologies, Robotics Center, AIRI, SB-AI-Lab, Royal Holloway University of London, MBZUAI


## Problem Background

自然语言生成（NLG）任务在特定领域中受限于标注数据的高成本和大型语言模型（LLMs）的生成质量不足，而主动学习（AL）作为减少标注成本的方法在 NLG 任务中应用有限，缺乏统一框架来支持和评估 AL 策略。
因此，论文提出 ATGen 框架，旨在通过 AL 减少人工和 LLM 自动标注的成本，并为 NLG 任务提供一个开发和测试 AL 策略的平台。

## Method

*   **框架目标与设计**：ATGen 是一个综合性框架，旨在将主动学习（AL）应用于 NLG 任务，通过选择最具信息量的样本进行标注，减少标注成本，同时支持人工和自动标注场景。
*   **AL 策略集成**：框架实现了多种针对 NLG 任务的 AL 策略，包括 HUDS（结合不确定性和度量学习）、HADAS（针对文本摘要的幻觉感知评分）、IDDS（基于语义相似度选择样本）、Facility Location（基于子模块函数的实验设计）等，以及不确定性采样策略如归一化序列概率和平均 token 熵，用于从无标注数据池中选择最有价值的样本。
*   **标注模式支持**：提供 Web GUI 界面支持人工标注，推荐使用实验设计（ED）策略以减少迭代延迟；同时支持 LLM 自动标注，集成主流 API 服务（如 OpenAI、Anthropic）及本地模型，优化批量 API 调用以降低成本。
*   **高效计算支持**：支持参数高效微调（PEFT）方法，包括 LoRA（低秩适应）、QLoRA（量化低秩适应）、DoRA（权重分解低秩适应），以减少大模型微调的计算资源需求；集成高效推理框架如 vLLM（基于 PagedAttention 优化）、SGLang（基于 RadixAttention 前缀缓存）、Unsloth（内存高效内核），加速 AL 循环中的模型更新和样本评估。
*   **数据与模型兼容性**：与 HuggingFace 深度集成，支持多种 NLG 任务数据集和模型，允许用户上传自定义 CSV 或 JSON 格式数据。
*   **评估与基准测试**：提供多种评估指标，包括自动化指标（如 BLEU、ROUGE）、开源 LLM 指标（如 BERTScore、AlignScore）及专有 LLM 指标（如 DeepEval），并提供基准测试脚本以评估和比较不同 AL 策略的性能。

## Experiment

*   **实验设置**：在多个 NLG 任务（如 TriviaQA、GSM8K、RACE、AESLC）上测试 AL 策略，采用模拟 AL 循环设置，覆盖人工标注模拟和 LLM 自动标注（使用 DeepSeek-R1）两种场景，使用 Qwen3-1.7B 作为获取模型，评估指标包括 Exact Match、ROUGE-2、AlignScore 等。
*   **有效性**：结果显示 HUDS、HADAS 和 Facility Location 等 AL 策略显著优于随机采样，例如在 TriviaQA 上，AL 策略仅需标注 4% 数据即可达到随机采样 12% 数据时的性能，标注量减少约 3 倍。
*   **成本节约**：在 LLM 自动标注场景中，AL 策略可将 API 调用成本降低 2-4 倍，同时保持性能。
*   **局限性与合理性**：在 GSM8K 数学推理任务中，LLM 标注导致整体性能下降，表明其质量不足以完全替代人工标注；实验覆盖多种任务和场景，设置较为全面，但未探讨 AL 可能引入的数据分布偏差。

## Further Thoughts

ATGen 框架将 AL 与 LLM 自动标注结合以降低 API 调用成本的思路具有实际价值，未来可探索结合领域知识的提示工程提升 LLM 标注质量；此外，框架对 PEFT 和高效推理框架的集成启发我们在资源受限环境下部署 AL 的可能性，例如在边缘设备上运行；同时，AL 可能引入的数据分布偏差问题提示我们设计更鲁棒的采样策略，如结合对抗性采样或分布校正方法。