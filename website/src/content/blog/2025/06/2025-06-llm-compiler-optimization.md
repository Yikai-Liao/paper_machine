---
title: "Compiler Optimization via LLM Reasoning for Efficient Model Serving"
pubDatetime: 2025-06-02T07:02:46+00:00
slug: "2025-06-llm-compiler-optimization"
type: "arxiv"
id: "2506.01374"
score: 0.707542006589878
author: "grok-3-latest"
authors: ["Sujun Tang", "Christopher Priebe", "Rohan Mahapatra", "Lianhui Qin", "Hadi Esmaeilzadeh"]
tags: ["LLM", "Compiler Optimization", "Monte Carlo Tree Search", "Contextual Reasoning", "Sample Efficiency"]
institution: ["University of California San Diego"]
description: "本文提出 REASONING COMPILER 框架，通过将 LLM 的上下文推理与 MCTS 结合，显著提升了神经网络编译器优化的采样效率和模型服务性能。"
---

> **Summary:** 本文提出 REASONING COMPILER 框架，通过将 LLM 的上下文推理与 MCTS 结合，显著提升了神经网络编译器优化的采样效率和模型服务性能。 

> **Keywords:** LLM, Compiler Optimization, Monte Carlo Tree Search, Contextual Reasoning, Sample Efficiency

**Authors:** Sujun Tang, Christopher Priebe, Rohan Mahapatra, Lianhui Qin, Hadi Esmaeilzadeh

**Institution(s):** University of California San Diego


## Problem Background

大型语言模型（LLM）和其他神经网络模型在推理阶段的高计算成本成为模型服务广泛应用和快速创新的重大障碍。
传统编译器在优化神经网络工作负载时面临挑战，因为可能的程序变换空间（如平铺、融合、布局调整）极其庞大且高度相互依赖，导致现有随机搜索技术（如进化算法）采样效率低，未能充分利用编译决策的结构化上下文。
论文旨在解决如何在不重新训练 LLM 的情况下，利用其上下文感知能力显著提高编译器优化的采样效率。

## Method

*   **核心思想:** 将编译器优化问题建模为序列化决策过程，利用大型语言模型（LLM）的上下文推理能力结合蒙特卡洛树搜索（MCTS）进行高效探索。
*   **问题形式化:** 将优化问题定义为有限时间范围的马尔可夫决策过程（MDP），每个状态对应一个程序变体，每个动作对应一个变换操作（如平铺、并行化），目标是找到最大化性能指标（如延迟、功耗）的变换序列。
*   **LLM 上下文推理:** LLM 作为提案生成器，通过精心设计的提示（Prompt）接收当前程序、父节点和祖父节点的代码、变换历史及性能成本信息，采用链式思维（Chain-of-Thought, CoT）推理方式，分析变换对性能的影响并提出硬件感知的变换序列建议。提案经过验证后应用于程序生成新变体。
*   **MCTS 结构化搜索:** MCTS 通过 UCT（Upper Confidence Bound for Trees）准则平衡探索与利用，选择有前景的节点进行扩展，结合 LLM 提案生成新节点，并通过模拟（使用硬件成本模型评估未来潜在路径）估计长期影响，最后反向传播更新树统计信息以指导后续搜索。
*   **硬件成本模型:** 使用学习的代理模型近似真实硬件性能，避免昂贵的实际硬件运行，确保搜索过程的高效性。
*   **关键创新:** 无需微调或训练 LLM，直接利用其预训练能力嵌入优化流程，结合结构化搜索框架，在高维、交互性强的优化空间中实现上下文感知的高效探索。

## Experiment

*   **有效性:** LLM-Guided MCTS 在多个基准测试（如 Llama3-8B 自注意力层、DeepSeek-R1 MoE 层）上表现出显著加速，例如在 Llama3-8B 上仅用 36 个样本达到 7.08 倍加速，而进化搜索（Evolutionary Search）需 72 个样本。
*   **采样效率:** 在 FLUX 自注意力层上，LLM-Guided MCTS 用 36 个样本达到 2 倍加速，而进化搜索需超 600 个样本，效率提升高达 16 倍，特别是在低采样预算下表现突出。
*   **收敛速度:** 方法在搜索初期快速收敛，对资源受限场景（如实时编译）具有实际意义。
*   **实验设置合理性:** 实验覆盖多种模型和算子类型（自注意力、卷积、MoE），在固定硬件环境（Intel Core i9, Apache TVM v0.20.0）下重复 20 次以确保统计稳定性；消融研究分析了 LLM 选择、历史轨迹深度和分支因子的影响，验证了方法的鲁棒性。
*   **局限性:** 实验未充分探讨方法在不同硬件平台上的泛化性，也未详细分析 LLM 提案的计算开销对整体效率的影响。

## Further Thoughts

论文中 LLM 与 MCTS 结合的上下文感知优化框架启发了我思考如何将通用预训练模型直接应用于其他序列决策问题，如自动代码生成或硬件设计自动化；此外，是否可以通过多模型协作（一个 LLM 负责提案生成，另一个负责评估）或动态调整历史轨迹深度来进一步提升效率和适应性？这种结构化搜索与生成模型结合的范式也可能适用于更广泛的组合优化问题，如资源调度或路径规划。