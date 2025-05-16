---
title: "DeepMath-Creative: A Benchmark for Evaluating Mathematical Creativity of Large Language Models"
pubDatetime: 2025-05-13T16:58:05+00:00
slug: "2025-05-deepmath-creative-benchmark"
type: "arxiv"
id: "2505.08744"
score: 0.6917769964554747
author: "grok-3-latest"
authors: ["Xiaoyang Chen", "Yuting Gao", "Xiang Jiang", "Xiangnan Li"]
tags: ["LLM", "Mathematical Reasoning", "Creativity Evaluation", "Benchmark Design"]
institution: ["Tongji University", "Fudan University", "Tsinghua University", "Tianjin University", "University of Hong Kong"]
description: "本文提出 DeepMath-Creative 基准，专注于评估大型语言模型的数学创造力，通过构造性任务揭示当前模型在创新性问题解决上的显著局限性。"
---

> **Summary:** 本文提出 DeepMath-Creative 基准，专注于评估大型语言模型的数学创造力，通过构造性任务揭示当前模型在创新性问题解决上的显著局限性。 

> **Keywords:** LLM, Mathematical Reasoning, Creativity Evaluation, Benchmark Design

**Authors:** Xiaoyang Chen, Yuting Gao, Xiang Jiang, Xiangnan Li

**Institution(s):** Tongji University, Fudan University, Tsinghua University, Tianjin University, University of Hong Kong


## Problem Background

大型语言模型（LLMs）在数学推理任务上表现出较强的能力，尤其是在基础到本科水平的数学问题上，但现有数据集和基准主要关注推理技能，对模型的数学创造力（Mathematical Creativity）评估不足，相关数据集稀缺。
论文提出，数学创造力是模型能力的重要维度，涉及新概念生成、新方法发明以及新例子的构建（如反例），而当前模型在这方面的表现尚未被充分探索。
因此，本文旨在设计一个专门评估数学创造力的基准——DeepMath-Creative，并通过该基准揭示模型在创造性问题解决上的局限性，解决如何定义和评估数学创造力的问题。

## Method

*   **核心目标：** 构建一个高质量的基准数据集 DeepMath-Creative，用于评估大型语言模型在数学创造力方面的能力，聚焦于创新性和构造性（Constructiveness）。
*   **设计原则：** 强调问题设计的创新性，覆盖数学核心分支（如代数、拓扑、分析），旨在测试模型是否能超越记忆模式，展现独立探索和创新解决能力。
*   **问题类型：** 包含两种探究式问题：（1）需要形式证明的问题，要求模型构建数学对象以验证命题；（2）需要反例的问题，要求模型构建反例以推翻命题。这两种类型共同构成对创造力的全面评估框架。
*   **数据收集：** 由数学领域专家（教授和研究生）设计和标注问题，确保逻辑严谨性和数学正确性，数据集包含 179 个问题，分布于代数（50%）、拓扑（15%）、分析（35%）等领域，难度涵盖本科（60%）和硕士（40%）水平。
*   **评估框架：** 结合定量和定性指标，定量上使用‘方向准确性’和‘过程准确性’评分（0、0.5、1 三档），分别对应错误方向、部分正确但有缺陷、完全正确；定性上由专家手动评分，关注逻辑严谨性、表达清晰度和原创性。
*   **实验流程：** 通过统一 API 接口测试多个主流模型，采用标准化提示格式生成响应，并根据评分准则进行人工评审，确保评估的客观性和公平性。

## Experiment

*   **有效性：** 实验评估了五个主流模型（GPT O3-mini, Claude-3-7-Sonnet, Gemini-2.0-Flash, DeepSeek R1, Qwen QwQ-32B），结果显示即使采用宽松评分标准（仅关注核心解题要素，忽略小错误），最佳模型 O3 Mini 仅达到 70% 准确率，且主要集中在本科水平的基础构造任务上。
*   **局限性：** 随着任务难度增加，模型性能显著下降，尤其在开放性问题上，模型未能提出有意义的策略，表现更多依赖记忆模式的组合，而非真正创造性洞察。
*   **实验设置：** 实验涵盖不同模型规模和架构，测试环境统一，确保公平性；数据分布（本科 60%，硕士 40%；证明 40%，反例 60%）合理，能较好反映创造力需求；但论文未提出具体改进模型的方法，仅提供基准和评估，未能直接提升模型性能。
*   **常见问题：** 模型在测试中表现出错误解题方向（如应构建反例却尝试证明）、构造过程有缺陷、输出冗长且无明确结论等问题。

## Further Thoughts

论文提出的数学创造力三维定义（新概念生成、新方法发明、新例子构建）不仅适用于数学领域，也可能推广到其他需要创造力的任务，如科学发现或工程设计，启发我们思考如何通过构造性任务设计测试模型在其他领域的创新能力；此外，论文提到未来使用强化学习（RL）训练 DeepMath-Creative 模型，这提示创造力可能需要通过与环境交互的动态学习来培养，而非单纯依赖预训练数据，或许可以探索结合人类反馈和探索式学习进一步提升模型创造力。