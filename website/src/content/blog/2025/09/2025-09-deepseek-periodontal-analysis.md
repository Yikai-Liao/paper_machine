---
title: "DeepSeek performs better than other Large Language Models in Dental Cases"
pubDatetime: 2025-09-02T07:26:20+00:00
slug: "2025-09-deepseek-periodontal-analysis"
type: "arxiv"
id: "2509.02036"
score: 0.6687848775522678
author: "grok-3-latest"
authors: ["Hexian Zhang", "Xinyu Yan", "Yanqi Yang", "Lijian Jin", "Ping Yang", "Junwen Wang"]
tags: ["LLM", "Clinical Reasoning", "Domain Adaptation", "Evaluation Metrics", "Open-Source Model"]
institution: ["The University of Hong Kong", "Mayo Clinic"]
description: "本文通过对比四个大型语言模型在牙周病学案例分析中的表现，证明 DeepSeek V3 在忠实度、相关性和临床准确性上显著优于其他模型，为 LLMs 在临床教育和决策支持中的应用提供了重要证据。"
---

> **Summary:** 本文通过对比四个大型语言模型在牙周病学案例分析中的表现，证明 DeepSeek V3 在忠实度、相关性和临床准确性上显著优于其他模型，为 LLMs 在临床教育和决策支持中的应用提供了重要证据。 

> **Keywords:** LLM, Clinical Reasoning, Domain Adaptation, Evaluation Metrics, Open-Source Model

**Authors:** Hexian Zhang, Xinyu Yan, Yanqi Yang, Lijian Jin, Ping Yang, Junwen Wang

**Institution(s):** The University of Hong Kong, Mayo Clinic


## Problem Background

大型语言模型（LLMs）在医疗和牙科领域的应用潜力日益凸显，尤其是在处理复杂临床案例时展现推理能力。
本文聚焦牙周病学（Periodontology），由于其结构化的临床数据和标准化的诊断治疗流程，成为测试 LLMs 的理想场景。
研究的关键问题是：现有 LLMs 是否能够准确理解和回答基于牙周病学纵向案例的开放性问题？
这一问题至关重要，因为牙周病管理中约一半的决策错误源于对多源患者信息的整合不足，而 LLMs 有望通过智能分析支持临床决策和教育培训。

## Method

*   **研究设计:** 本研究从 Wiley-Blackwell 出版的《Clinical Cases in Periodontics》中提取了 34 个牙周病学纵向案例，生成 258 个开放性问答对，并随机选取 30%（78 个问题）进行测试。
*   **测试对象:** 评估了四个主流大型语言模型：OpenAI 的 GPT-4o、Google 的 Gemini 2.0 Flash、Microsoft 的 Copilot 和 DeepSeek V3。
*   **交互框架:** 采用三步对话框架：首先通过提示词设定模型为‘牙周病学导师’角色；其次输入完整的案例背景以建立上下文；最后提出开放性临床问题要求模型直接回答。
*   **评估方式:** 性能评估结合了自动化指标和专家评估。自动化指标包括忠实度（Faithfulness，衡量生成答案与参考答案的事实一致性）、答案相关性（Answer Relevancy，衡量答案与问题的语义相关性）和可读性（Readability，通过 Flesch-Kincaid 等级计算）。专家评估由两名持证牙医基于五级 Likert 量表进行盲评，重点关注临床准确性。
*   **数据污染检测:** 通过 MELD 评分检测模型是否可能提前接触过测试数据，确保结果基于学习能力而非记忆。
*   **补充实验:** 对 GPT-4o 进行微调实验，探索通过领域数据训练提升性能的可能性。

## Experiment

*   **有效性:** DeepSeek V3 在多项指标上显著优于其他模型。自动化评估中，其忠实度中位数为 0.528，高于 GPT-4o（0.402）、Gemini 2.0 Flash（0.457）和 Copilot（0.367）；答案相关性中位数为 0.946，与其他模型接近但分布更稳定；可读性中位数为 12.8，略低于 Copilot（11.9）但仍属较高水平。专家评估中，DeepSeek V3 的临床准确性中位数为 4.5/5，明显高于其他模型的 4.0/5。
*   **统计显著性:** 通过 Friedman 测试和 Wilcoxon 配对检验，DeepSeek V3 的优势在统计上显著（p < 0.05）。
*   **实验设置合理性:** 实验设计较为全面，结合自动化指标和专家评估双重维度，确保结果的客观性和临床相关性；通过 MELD 评分（均值低于 0.28）排除数据污染可能性，增强了结果可信度。
*   **局限性:** 实验未纳入图像数据，可能限制了对复杂案例的全面分析；参考答案由作者单方面定义，可能存在偏见，导致自动化评估与专家评分偶有不一致。
*   **微调效果:** 对 GPT-4o 的微调实验显示忠实度从 0.421 提升至 0.457，但专家评分略降（4.19 至 3.96），表明微调优化了答案呈现和可读性，但未显著提升临床准确性。

## Further Thoughts

DeepSeek V3 的混合专家（MoE）架构通过动态查询路由机制显著提升了领域特定知识的处理能力，这启发我们未来可以通过类似架构优化或领域微调进一步提升 LLMs 在医疗领域的性能；此外，DeepSeek 作为开源模型的表现优于闭源模型，提示开源模型在医疗领域的可定制性和成本效益潜力；最后，自动化评估指标（如忠实度）与专家评估的高度一致性表明，未来可利用自动化指标作为初步筛选工具，降低人工评估成本。