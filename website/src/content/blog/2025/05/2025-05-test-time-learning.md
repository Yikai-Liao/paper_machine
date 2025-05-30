---
title: "Test-Time Learning for Large Language Models"
pubDatetime: 2025-05-27T02:18:59+00:00
slug: "2025-05-test-time-learning"
type: "arxiv"
id: "2505.20633"
score: 0.8319750922063133
author: "grok-3-latest"
authors: ["Jinwu Hu", "Zitian Zhang", "Guohao Chen", "Xutao Wen", "Chao Shuai", "Wei Luo", "Bin Xiao", "Yuanqing Li", "Mingkui Tan"]
tags: ["LLM", "Adaptation", "Self-Supervised Learning", "Efficiency", "Domain Shift"]
institution: ["South China University of Technology, School of Software Engineering", "Pazhou Laboratory", "Zhejiang University", "South China Agricultural University", "Chongqing University of Posts and Telecommunications", "Key Laboratory of Big Data and Intelligent Robot, Ministry of Education"]
description: "本文提出了一种测试时学习方法 TLM，通过输入困惑度最小化、样本高效学习策略和 LoRA 参数更新，显著提升了大型语言模型在分布偏移下的性能，同时降低了计算成本。"
---

> **Summary:** 本文提出了一种测试时学习方法 TLM，通过输入困惑度最小化、样本高效学习策略和 LoRA 参数更新，显著提升了大型语言模型在分布偏移下的性能，同时降低了计算成本。 

> **Keywords:** LLM, Adaptation, Self-Supervised Learning, Efficiency, Domain Shift

**Authors:** Jinwu Hu, Zitian Zhang, Guohao Chen, Xutao Wen, Chao Shuai, Wei Luo, Bin Xiao, Yuanqing Li, Mingkui Tan

**Institution(s):** South China University of Technology, School of Software Engineering, Pazhou Laboratory, Zhejiang University, South China Agricultural University, Chongqing University of Posts and Telecommunications, Key Laboratory of Big Data and Intelligent Robot, Ministry of Education


## Problem Background

大型语言模型（LLMs）通过大规模预训练展现了强大能力，但在实际部署中常面临分布偏移（distribution shifts）问题，导致在特定领域知识和语言多样性上的性能下降。
论文指出，现有方法如微调、检索增强生成（RAG）和测试时适应（TTA）存在局限性，例如需要大量标注数据、忽略自回归依赖性或计算开销高昂，因此提出了一种测试时学习（Test-Time Learning, TTL）方法，仅使用无标注测试数据动态适应目标领域。

## Method

*   **核心思想：** 提出一种测试时学习方法 TLM（Test-Time Learning for LLMs），通过自监督方式在测试时动态适应分布偏移，提升模型性能，同时避免灾难性遗忘和降低计算开销。
*   **输入困惑度最小化（Input Perplexity Minimization）：** 基于观察到输入困惑度与输出困惑度呈正相关，作者将测试时学习目标定义为最小化输入数据的困惑度（perplexity），即通过优化模型对输入序列的预测概率来间接提升对目标分布的生成能力。这种自监督目标无需标注数据，适用于测试时场景。
*   **样本高效学习策略（Sample Efficient Learning Strategy）：** 针对不同测试样本对模型更新的贡献差异，作者发现高困惑度样本更具信息量，因此设计了一种基于困惑度的加权方案，通过计算样本的困惑度值并设置阈值，优先选择高困惑度样本进行反向传播，从而减少不必要的计算开销并提升适应效率。
*   **低秩适应（Low-Rank Adaptation, LoRA）：** 为避免灾难性遗忘和降低计算成本，作者采用 LoRA 进行参数更新，通过低秩矩阵分解仅调整模型参数的一小部分（而非全参数更新），实现轻量级训练，同时保留模型的原始知识，确保适应过程的稳定性。
*   **实现细节：** TLM 方法通过算法流程整合上述组件，在测试时对每个输入样本计算困惑度，依据加权方案选择更新样本，并通过 LoRA 进行参数优化，整个过程无需访问训练数据或外部知识库。

## Experiment

*   **有效性：** 在作者构建的 AdaptEval 基准数据集上（包括领域知识、指令跟随和推理任务），TLM 在多种 LLM 架构（如 Llama3.2-3B-Instruct, Llama3-8B-Instruct, Qwen2.5-7B-Instruct）上均显著优于原始模型，性能提升至少 20%。例如，在 DomainBench 的 Geography 数据集上，TLM 相较于 Llama3.2-3B-Instruct 提升了 20.79%。
*   **对比优势：** 相较于基线方法（如 Tent, EATA, COME），TLM 在领域知识适应、指令任务和推理任务上表现更优。例如，在 Qwen2.5-7B-Instruct 的 Agriculture 数据集上，TLM 相较于 EATA 提升了 37.32%。
*   **实验设置合理性：** 实验覆盖了多种模型规模和任务类型，AdaptEval 基准设计考虑了分布偏移的多样性（垂直领域偏移和非特定领域偏移），通过消融研究验证了每个组件的贡献（如输入困惑度最小化提升了 30% 以上，样本高效策略进一步提升约 2%）。
*   **计算开销：** 通过样本高效策略和 LoRA，TLM 显著降低了计算成本，例如在线设置下反向传播次数减少了 69.7%，同时在量化模型（如 4-bit Llama3-8B-Instruct）上也表现出色，性能提升至少 25%。

## Further Thoughts

输入困惑度最小化作为自监督目标的创新应用，为测试时学习提供了一种无需标注数据的新思路，未来可以探索其他自监督目标（如对比学习）在类似场景中的潜力；高困惑度样本优先策略提示我们可以进一步研究动态样本选择机制，例如结合任务难度或领域特性设计更精细的加权方案；LoRA 在测试时学习中的成功应用启发我们思考是否可以设计更高效的参数更新方法，甚至结合模型剪枝或量化技术以适应资源受限环境。