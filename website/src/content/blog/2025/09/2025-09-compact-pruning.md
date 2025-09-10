---
title: "COMPACT: Common-token Optimized Model Pruning Across Channels and Tokens"
pubDatetime: 2025-09-08T16:07:06+00:00
slug: "2025-09-compact-pruning"
type: "arxiv"
id: "2509.06836"
score: 0.664314519852267
author: "grok-3-latest"
authors: ["Eugene Kwek", "Wenpeng Yin"]
tags: ["LLM", "Model Compression", "Pruning", "Vocabulary Optimization", "Efficiency"]
institution: ["Penn State University"]
description: "COMPACT提出了一种训练无关的剪枝框架，通过词汇剪枝和常见Token加权的FFN剪枝，实现跨尺度鲁棒性和高效压缩，同时保持标准Transformer架构，显著提升LLM部署友好性。"
---

> **Summary:** COMPACT提出了一种训练无关的剪枝框架，通过词汇剪枝和常见Token加权的FFN剪枝，实现跨尺度鲁棒性和高效压缩，同时保持标准Transformer架构，显著提升LLM部署友好性。 

> **Keywords:** LLM, Model Compression, Pruning, Vocabulary Optimization, Efficiency

**Authors:** Eugene Kwek, Wenpeng Yin

**Institution(s):** Penn State University


## Problem Background

大型语言模型（LLMs）因参数规模庞大导致内存占用、推理延迟和能耗成本高，限制了在边缘设备和交互式应用中的部署。
现有剪枝方法存在局限：深度剪枝移除整个层导致性能骤降，宽度剪枝常破坏标准Transformer架构或需定制推理代码，且缺乏跨尺度鲁棒性。
作者通过分析参数分布（小模型嵌入层占比高，大模型FFN占比高）和语言学特性（稀有Token贡献小），提出新的剪枝框架以解决这些问题。

## Method

*   **核心思想:** COMPACT是一种训练无关的剪枝框架，通过联合词汇剪枝和基于常见Token加权的前馈网络（FFN）剪枝，减少模型参数和内存占用，同时保持标准Transformer架构和下游任务性能。
*   **词汇剪枝（Vocabulary Pruning）:** 基于自然语言的Zipf分布规律，识别并移除词汇表中稀有Token对应的嵌入和解嵌入矩阵行，直接减少参数量，尤其对小模型有效。此步骤无需校准数据或前向推理，操作简单高效，且不引入额外超参数。
*   **常见Token加权的FFN剪枝（Common-Token-Weighted FFN Pruning）:** 针对FFN中间通道，提出改进的激活值评分方法‘Common act²’，通过校准数据集计算通道重要性，但仅考虑常见Token（即剪枝后仍有效的Token）的激活值，而非传统方法（如act²）对所有Token一视同仁，确保剪枝优化针对常见Token分布。
*   **联合剪枝流程:** 先识别稀有Token集合，用于指导FFN剪枝的激活值加权计算，最后同时移除词汇参数和FFN参数，确保两部分剪枝协同优化，而非孤立执行。
*   **优势:** 方法规模自适应（通过调整词汇和FFN剪枝比例适应不同模型规模）、架构无关（保持标准Transformer结构，兼容现有推理框架如Huggingface、vLLM）、训练无关（无需额外训练，剪枝时间短）。

## Experiment

*   **有效性:** COMPACT在小模型（如Qwen 2.5-0.5B）上，35%剪枝比例下平均得分达35.3（恢复至原模型的70.4%），显著优于基线方法（如SliceGPT、ShortGPT），后者在10%比例下性能已接近随机；在较大模型（如LLaMA 3.1-70B）上，35%比例下平均得分63.7（恢复至80.2%），同样达到最优。
*   **性能下降平滑性:** 与深度剪枝方法（如ShortGPT）常见的性能骤降不同，COMPACT展现平滑下降，尤其在困难任务（如MMLU、GSM8K）上仍高于随机猜测，体现更强的鲁棒性。
*   **效率提升:** 内存使用显著减少（如LLaMA 3.1-8B分类任务降至原模型64%），推理吞吐量提升（分类任务1.37倍，生成任务1.38倍），虽不及深度剪枝但优于其他宽度剪枝；剪枝时间短（如8B模型32秒，70B模型2分17秒），接近深度剪枝效率。
*   **实验设置合理性:** 实验覆盖0.5B-70B规模的多个模型家族（Qwen、LLaMA、Gemma），测试7个下游任务（多选题与生成任务兼顾），指标全面（性能、效率、内存）；因词汇剪枝影响困惑度指标，故未报告，属合理取舍；不足之处是部分基线未能在某些模型（如Gemma 3）上运行，略影响对比完整性。

## Further Thoughts

COMPACT基于参数分布和语言学特性的剪枝思路启发我思考是否可以结合语义信息（如Token的上下文依赖性）进一步指导剪枝，而非仅依赖频率分布；此外，词汇与FFN的协同剪枝提示未来可探索注意力层与FFN的联合优化，平衡不同模块冗余性；训练无关性也让我联想到结合轻量级后训练（如自数据蒸馏）动态调整剪枝比例，以适应边缘设备资源限制。