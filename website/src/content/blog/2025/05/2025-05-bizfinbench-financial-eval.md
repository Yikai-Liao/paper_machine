---
title: "BizFinBench: A Business-Driven Real-World Financial Benchmark for Evaluating LLMs"
pubDatetime: 2025-05-26T03:23:02+00:00
slug: "2025-05-bizfinbench-financial-eval"
type: "arxiv"
id: "2505.19457"
score: 0.48479344456064555
author: "grok-3-latest"
authors: ["Guilong Lu", "Xuntao Guo", "Rongjunchen Zhang", "Wenqiao Zhu", "Ji Liu"]
tags: ["LLM", "Financial Benchmark", "Evaluation Framework", "Reasoning", "Numerical Computation"]
institution: ["HiThink Research", "Harbin Institute of Technology"]
description: "本文提出 BizFinBench，一个业务驱动的金融基准，结合真实场景数据和 IteraJudge 迭代评估框架，揭示了大型语言模型在金融任务中的能力差距，为未来研究提供了严谨的测试平台。"
---

> **Summary:** 本文提出 BizFinBench，一个业务驱动的金融基准，结合真实场景数据和 IteraJudge 迭代评估框架，揭示了大型语言模型在金融任务中的能力差距，为未来研究提供了严谨的测试平台。 

> **Keywords:** LLM, Financial Benchmark, Evaluation Framework, Reasoning, Numerical Computation

**Authors:** Guilong Lu, Xuntao Guo, Rongjunchen Zhang, Wenqiao Zhu, Ji Liu

**Institution(s):** HiThink Research, Harbin Institute of Technology


## Problem Background

大型语言模型（LLMs）在通用任务中表现出色，但在逻辑密集、精度要求高的金融领域，其可靠性和鲁棒性评估仍面临挑战。
现有金融基准多将任务简化为通用文档问答，缺乏结构化输入和业务导向的推理能力，与真实金融场景需求存在显著差距。
BizFinBench 旨在通过构建一个业务驱动的真实世界金融基准，解决 LLMs 在复杂金融任务（如多步推理、时间敏感性、对抗性上下文）中的评估不足问题。

## Method

*   **基准设计 (BizFinBench):** 提出了一个包含 6,781 个中文查询的金融基准，覆盖五个关键维度：数值计算、推理、信息提取、预测识别和知识问答，细分为九个类别（如异常事件归因、金融时间推理）。
    *   数据来源于真实用户查询（如 iwencai APP），通过 GPT-4o 清洗和分类，并由三位资深金融专家进行多轮标注，确保数据质量和业务相关性。
    *   强调上下文复杂性和对抗性鲁棒性，例如在异常事件归因任务中引入误导性信息，测试模型在噪声环境下的细粒度推理能力。
    *   数据集结合内部金融数据库和外部来源，包含股票价格、历史交易数据、新闻等，增强任务的现实性。
*   **评估框架 (IteraJudge):** 提出了一种迭代校准的评估方法，通过三个核心机制提升评估可靠性：
    *   维度解耦：将评估分解为多个独立维度（如因果一致性、计算准确性），逐一分析模型输出。
    *   顺序校正生成：通过提示 LLM 逐步改进初始答案，形成可解释的改进轨迹。
    *   参考对齐评估：以最终改进答案作为质量基准，通过对比评估初始输出的不足。
*   **关键创新:** 结合业务场景设计任务，注重真实世界适用性，同时通过 IteraJudge 减少 LLM 作为评判者时的偏见，提供更可靠的评估结果。

## Experiment

*   **评估范围与结果:** 评估了 25 个模型（包括闭源和开源模型），结果显示无单一模型在所有任务中占据主导地位。
    *   数值计算任务：Claude-3.5-Sonnet (63.18) 和 DeepSeek-R1 (64.04) 表现最佳，小型模型如 Qwen2.5-VL-3B (15.92) 显著落后。
    *   推理任务：闭源模型优势明显，ChatGPT-o3 (83.58) 和 Gemini-2.0-Flash (81.15) 领先，开源模型落后高达 19.49 分。
    *   信息提取任务：性能差距最大，DeepSeek-R1 (71.46) 领先，Qwen3-1.7B 仅得 11.23。
    *   预测识别任务：性能方差较小，顶级模型得分在 39.16 至 50.00 之间。
*   **实验设置合理性:** 任务设计贴近业务场景，涵盖多种模型规模和类型（从 1.7B 到 671B 参数），评估维度多样，数据来源于真实用户查询，增强现实性。
*   **IteraJudge 效果:** 消融实验表明 IteraJudge 有效减少评估偏见，Spearman 相关性提升最高达 17.24%（在金融数据描述任务中），最低也有 3.09% 提升。
*   **局限性:** 评估未完全覆盖多轮交互和实时数据处理场景，模型性能可能受训练数据和推理策略影响，存在一定实验偏差。

## Further Thoughts

BizFinBench 的业务驱动基准设计启发我们可以在其他高风险领域（如法律、医疗）构建类似真实场景的评估框架，以提升模型的现实适用性。
IteraJudge 的迭代校准机制提示我们可以在评估中引入多阶段改进和专家反馈，甚至结合强化学习动态调整评估维度权重，进一步减少偏见。
论文揭示的模型能力差异（如数值计算与情感识别的差距）启发我们在开发领域特定模型时，应针对性优化薄弱环节，而非追求全面能力提升；此外，是否可以探索跨领域迁移能力，例如将金融推理能力应用于医疗决策支持？