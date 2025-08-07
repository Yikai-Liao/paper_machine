---
title: "VFLAIR-LLM: A Comprehensive Framework and Benchmark for Split Learning of LLMs"
pubDatetime: 2025-08-05T05:20:33+00:00
slug: "2025-08-split-learning-llm"
type: "arxiv"
id: "2508.03097"
score: 0.43957257701920915
author: "grok-3-latest"
authors: ["Zixuan Gu", "Qiufeng Fan", "Long Sun", "Yang Liu", "Xiaojun Ye"]
tags: ["LLM", "Split Learning", "Privacy Protection", "Fine-Tuning", "Resource Efficiency"]
institution: ["Tsinghua University", "The Hong Kong Polytechnic University", "Shanghai Artificial Intelligence Laboratory"]
description: "本文提出VFLAIR-LLM框架，通过分割学习实现资源受限环境下的LLM隐私保护适配，并提供全面的攻击与防御基准测试，为实际应用提供实用指导。"
---

> **Summary:** 本文提出VFLAIR-LLM框架，通过分割学习实现资源受限环境下的LLM隐私保护适配，并提供全面的攻击与防御基准测试，为实际应用提供实用指导。 

> **Keywords:** LLM, Split Learning, Privacy Protection, Fine-Tuning, Resource Efficiency

**Authors:** Zixuan Gu, Qiufeng Fan, Long Sun, Yang Liu, Xiaojun Ye

**Institution(s):** Tsinghua University, The Hong Kong Polytechnic University, Shanghai Artificial Intelligence Laboratory


## Problem Background

随着大型语言模型（LLMs）在各领域的广泛应用，用户希望利用私有领域数据适配模型，但直接使用外部LLM API存在数据泄露风险，而本地部署LLM对计算资源需求极高，这对小型组织或个人而言几乎不可行。
公共数据的枯竭进一步加剧了私有数据使用的需求，但隐私问题成为主要瓶颈。
关键挑战在于如何在资源受限环境下实现LLM的隐私保护适配，同时应对分割学习（Split Learning, SL）本身面临的隐私攻击风险。

## Method

*   **框架设计核心:** 提出VFLAIR-LLM，一个轻量级、可扩展的分割学习框架，通过将LLM模型分割到数据方和模型方，实现资源高效和隐私保护的协同学习。
*   **模型分割策略:** 提供两种分割设置：
    *   **Head-Tail (HT):** 模型分为头部（数据方，包含嵌入层和少量编码器/解码器层）和尾部（模型方，包含大部分层和输出层），适用于基本隐私保护场景。
    *   **Head-Body-Tail (HBT):** 模型分为头部（数据方）、主体（模型方）和尾部（数据方），通过将输入和输出控制在数据方，进一步保护标签和推理结果。
*   **微调方法:** 支持多种微调策略，包括全模型微调（Full-Tuning）和本地微调（Local-Tuning），并集成参数高效微调（PEFT）方法如LoRA、AdaLoRA等，以减少计算开销。
*   **攻击与防御模块:** 内置多种隐私攻击（3种模型反演攻击MIA和2种标签推断攻击LIA）及9种防御策略（包括基于扰动的如差分隐私DP、Sparsification SP，和基于学习的如互信息防御MID、对抗训练AT），便于评估隐私风险和防御效果。
*   **支持范围与扩展性:** 支持16种LLM类型、3种任务类型（分类、生成、问答）和18个数据集，提供单机模拟和分布式部署两种工作模式，用户可自定义数据集和模型分割点。

## Experiment

*   **微调效果:** 测试了HT和HBT两种分割设置下的4种微调策略（Full-Vanilla, Full-LoRA, Local-Vanilla, Local-LoRA），结果显示LoRA在小型模型（如BERT）上与Vanilla微调效果相当，但在大型模型（如GPT-2）上准确率略降；HBT在本地微调时表现优于HT，因其可训练参数更多；实验覆盖多种任务和模型，设置全面。
*   **攻击与防御基准:** 评估了多种攻击和9种防御策略，BiSR和VMI等攻击在简单任务（如SST2）上成功率高（AP>0.6），但在复杂任务（如GSM8K）上效果较差；基于学习的防御（如MID）在隐私-实用性权衡上优于基于扰动的防御（如DP），尤其在复杂任务中表现突出；模型头部越大，隐私保护效果越好，但资源需求增加；LoRA微调在防御隐私攻击时更具鲁棒性。
*   **效率评估:** 分布式部署与单机模拟对比显示，分布式模式吞吐量较低（如Llama3-8B从23.27 token/s降至15.41 token/s），通信开销是主要瓶颈，需进一步优化。
*   **合理性与局限:** 实验设置覆盖多种场景、模型和参数配置，数据支持结论，但通信效率问题和复杂任务中防御的性能下降表明仍有改进空间。

## Further Thoughts

论文提出的HT和HBT分割方式启发了对模型架构动态调整的思考，是否可以根据任务隐私敏感度或资源限制设计自适应分割策略？此外，LoRA微调在隐私保护中的优越性提示，是否可以通过定制参数高效微调方法进一步增强模型抗攻击能力？DCS指标的综合评估思路也值得扩展，未来可加入计算和通信成本等多维度评估隐私-实用性权衡。