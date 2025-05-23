---
title: "EasyMath: A 0-shot Math Benchmark for SLMs"
pubDatetime: 2025-05-20T19:31:52+00:00
slug: "2025-05-easymath-slm-benchmark"
type: "arxiv"
id: "2505.14852"
score: 0.6367411985597355
author: "grok-3-latest"
authors: ["Drishya Karki", "Michiel Kamphuis", "Angelecia Frey"]
tags: ["Small Language Models", "Mathematical Reasoning", "Benchmark Design", "Zero-Shot Learning", "Chain of Thought"]
institution: ["Assistantslab"]
description: "本文提出 EasyMath，一个针对小型语言模型的实用数学基准测试，通过覆盖日常问题和多层次评估，揭示了 SLMs 在零样本数学推理中的能力和局限性。"
---

> **Summary:** 本文提出 EasyMath，一个针对小型语言模型的实用数学基准测试，通过覆盖日常问题和多层次评估，揭示了 SLMs 在零样本数学推理中的能力和局限性。 

> **Keywords:** Small Language Models, Mathematical Reasoning, Benchmark Design, Zero-Shot Learning, Chain of Thought

**Authors:** Drishya Karki, Michiel Kamphuis, Angelecia Frey

**Institution(s):** Assistantslab


## Problem Background

随着小型语言模型（Small Language Models, SLMs）在移动应用和聊天助手等资源受限场景中的广泛应用，评估其日常数学推理能力变得至关重要。
现有数学基准测试（如 GSM8K、MathQA）要么过于复杂（对 SLMs 几乎不可完成），要么采用多选题形式，无法真实反映模型在开放性问题上的解决能力。
EasyMath 旨在填补这一空白，专注于设计一个贴近实际需求的基准测试，覆盖从基础算术到多步推理的实用数学问题，揭示 SLMs 的能力与局限性。

## Method

*   **数据集设计:** EasyMath 包含13个数学类别，涵盖基础算术、运算顺序、百分比、字面题、代数表达式、多步推理及边缘案例等，避免过于专业的领域（如线性代数）。
    *   问题由人工手动编写，确保与现有数据集无重叠，并为每个问题提供 SymPy 可计算的参考解法，注重实用性和真实性。
*   **评估流程:** 采用多层次评估流水线：
    *   **响应处理与正则化:** 使用正则表达式提取模型回答中的数学表达式，并标准化格式（如将符号转换为统一表示）。
    *   **类别特定评估:** 根据问题类别选择评估方式，基础类别（如算术）采用严格匹配，其他类别（如代数、字面题）采用等价性检查。
    *   **多重匹配方法:** 包括直接字符串匹配、数值评估、符号等价性检查（借助 SymPy）和差值简化，确保评估的灵活性和准确性。
    *   **结果判定:** 只要任一匹配方法成功即判定为正确，否则标记为错误；边缘案例类别特别要求明确回答‘undefined’。
*   **辅助分析:** 研究模型训练（如在现有数学数据集上微调）和推理策略（如链式思维提示）对性能的影响，探索提升 SLMs 数学能力的方法。

## Experiment

*   **测试范围与设置:** 测试了23个模型（参数规模从14M到4B），采用零样本（0-shot）设置，覆盖标准指令微调模型、数学专用模型及推理模型。
*   **性能表现:** 模型准确率与参数规模呈正相关，例如 Qwen2.5-0.5B 达88.31%，AceMath-1.5B 达95.69%，而 SmolLM2-135M 仅28.77%；较大模型（≥1.5B）在代数表达式和多步推理上表现显著优于小型模型（<200M）。
*   **推理策略效果:** 链式思维（Chain-of-Thought, CoT）提示对 SLMs 性能提升有限，例如 Pythia-1.4B 从16.77%提升至22.92%，远不如大型语言模型中的效果。
*   **一致性分析:** 通过多轮测试发现，较大模型（如 Gemma-3-4B）一致性更高（标准差±0.30%），小型模型（如 SmolLM2-135M）波动较大（标准差±1.71%）。
*   **评估鲁棒性:** 计算意外提及正确答案概率（约4%），并据此调整准确率（如 Llama-3.2-1B 从74.92%调整至73.87%），显示评估的严谨性。
*   **对比其他基准:** EasyMath 相较 GSM8K（SLMs 上几乎全为0分）和 MathQA（多选题易受猜测影响）能更好区分 SLMs 性能，实验设置全面合理。

## Further Thoughts

EasyMath 揭示了 SLMs 在数学推理上的局限性，尤其是在复杂任务和推理策略（如链式思维）上的表现不如 LLMs，这启发我们探索更适合 SLMs 的轻量级推理框架或混合方法（如结合符号求解器与神经模块）。
此外，论文强调通过高质量数据集（如 Orca-word-problems）微调可显著提升性能，提示未来可以研究如何通过数据选择和结构化训练进一步优化 SLMs，甚至探索跨领域基准设计（如逻辑推理、常识问答）以贴近实际应用需求。