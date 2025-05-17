---
title: "Customizing a Large Language Model for VHDL Design of High-Performance Microprocessors"
pubDatetime: 2025-05-14T17:58:40+00:00
slug: "2025-05-vhdl-llm-customization"
type: "arxiv"
id: "2505.09610"
score: 0.5186716229315338
author: "grok-3-latest"
authors: ["Nicolas Dupuis", "Ravi Nair", "Shyam Ramji", "Sean McClintock", "Nishant Chauhan", "Priyanka Nagpal", "Bart Blaner", "Ken Valk", "Leon Stok", "Ruchir Puri"]
tags: ["LLM", "Hardware Design", "Code Explanation", "Pre-Training", "Post-Training", "Domain Adaptation", "Secure Deployment", "Evaluation Strategy"]
institution: ["IBM Research, Yorktown Heights, NY, USA", "IBM Infrastructure"]
description: "本文通过扩展预训练、指令微调和安全基础设施，定制大型语言模型以解释 VHDL 代码，显著提升高性能微处理器设计生产力，并为工业环境中的 AI 应用提供了创新评估和部署策略。"
---

> **Summary:** 本文通过扩展预训练、指令微调和安全基础设施，定制大型语言模型以解释 VHDL 代码，显著提升高性能微处理器设计生产力，并为工业环境中的 AI 应用提供了创新评估和部署策略。 

> **Keywords:** LLM, Hardware Design, Code Explanation, Pre-Training, Post-Training, Domain Adaptation, Secure Deployment, Evaluation Strategy

**Authors:** Nicolas Dupuis, Ravi Nair, Shyam Ramji, Sean McClintock, Nishant Chauhan, Priyanka Nagpal, Bart Blaner, Ken Valk, Leon Stok, Ruchir Puri

**Institution(s):** IBM Research, Yorktown Heights, NY, USA, IBM Infrastructure


## Problem Background

随着高性能微处理器设计复杂性的增加，设计周期长且对领域专家依赖严重，VHDL 作为一种广泛使用的硬件描述语言，在技能缺失和知识传承方面面临挑战。
论文旨在通过定制大型语言模型（LLM）来辅助 VHDL 代码解释，帮助新手设计师快速上手，保留组织内部的设计知识，同时应对工业环境中的数据隐私和安全需求。

## Method

*   **基础设施建设**：在 IBM Cloud 上部署安全虚拟私有云（VPC）和多区域对象存储桶，确保敏感设计数据的隐私和安全，所有训练和推理均在安全环境中进行。
*   **数据收集与预处理**：从内部资源（如 GitHub 项目、Wiki、课程材料、架构文档等）收集 VHDL 相关数据，通过过滤、去重、格式转换和分词步骤，确保数据质量，构建领域特定训练数据集。
*   **扩展预训练（EPT）**：基于 Granite 基础模型，使用 VHDL 代码和文档数据（约 162M 代码 token 和 14.6M 文档 token）进行扩展预训练，避免灾难性遗忘通过加入基础模型的回放数据，训练在 8 个 NVIDIA H100 GPU 上进行，上下文窗口为 8192 token。
*   **指令微调（IT）**：在扩展预训练后，使用 1.1M 文档数据集进一步微调模型，提升指令遵循能力，训练在 32 个 NVIDIA A100 GPU 上进行，优化学习率和批量大小。
*   **模型合并**：通过球面线性插值（SLERP）技术，将扩展预训练模型与指令微调的基础模型权重合并，以低成本获得类似指令微调的效果，仅需 10 分钟在 2 个 L40S GPU 上完成。
*   **模型评估与 LLM-as-a-Judge**：设计代码解释和多选题测试集，专家评估采用 Likert 量表，覆盖正确性、完整性等维度；由于专家资源有限，引入 LLM-as-a-Judge 模拟专家评分，通过提示模板评估模型输出，与专家评分相关性高达 0.99。
*   **束搜索（Beam Search）**：在代码解释任务中应用束搜索，探索不同束宽度（Beam Width）对生成质量的影响，优化 token 路径选择以提升解释质量。

## Experiment

*   **代码解释任务效果**：基础模型专家评分仅为 43%，通过两轮扩展预训练提升至 69%（EPT2.2 模型），指令微调后进一步提升至 71%，束搜索（Beam Width=2 或 5）使评分达到 77%；新基础模型预测评分高达 85%，显示未来潜力。
*   **多选题测试效果**：知识获取任务上，模型性能提升较小（从 34% 到 36%），但新基础模型（14B 参数）准确率达 61%，表明推理能力对硬件设计任务的重要性。
*   **实验设置合理性**：测试集分为代码解释（80 题）和多选题（263 题）两类，由专家构建并审查，覆盖数字设计、VHDL 语法等多个主题，评估维度细致（正确性、完整性等），设置全面合理。
*   **资源开销与反馈**：指令微调需 18 小时（32 个 A100 GPU），模型合并仅需 10 分钟（2 个 L40S GPU），部署后用户反馈显示 70% 正面评价，但反馈率低于 50%，提示工程已带来进一步改进。

## Further Thoughts

论文提出的领域特定定制化方法（扩展预训练与指令微调结合）可推广至其他工业场景，如金融或医疗领域的知识密集型任务；LLM-as-a-Judge 的评估方式为专家资源有限的情况提供了新思路，未来可通过多模型投票或结合人类反馈提升评判准确性；安全环境下的 AI 部署策略为敏感领域应用提供了参考；新基础模型在推理能力上的突破表明，结合外部知识库或更强推理模型可能进一步提升硬件设计任务（如调试和验证）的性能。