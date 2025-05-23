---
title: "Scaling Laws for State Dynamics in Large Language Models"
pubDatetime: 2025-05-20T20:38:21+00:00
slug: "2025-05-state-dynamics-scaling"
type: "arxiv"
id: "2505.14892"
score: 0.6952533697068858
author: "grok-3-latest"
authors: ["Jacob X Li", "Shreyas S Raman", "Jessica Wan", "Fahad Samman", "Jazlyn Lin"]
tags: ["LLM", "State Dynamics", "World Model", "Residual Stream", "Mechanistic Interpretability"]
institution: ["Brown University"]
description: "本文通过多任务、多规模实验和机制解释性分析，系统研究了大型语言模型在状态动态建模中的能力和局限，揭示了规模、任务复杂度和内部机制间的关系。"
---

> **Summary:** 本文通过多任务、多规模实验和机制解释性分析，系统研究了大型语言模型在状态动态建模中的能力和局限，揭示了规模、任务复杂度和内部机制间的关系。 

> **Keywords:** LLM, State Dynamics, World Model, Residual Stream, Mechanistic Interpretability

**Authors:** Jacob X Li, Shreyas S Raman, Jessica Wan, Fahad Samman, Jazlyn Lin

**Institution(s):** Brown University


## Problem Background

大型语言模型（LLM）被视为一种文本领域的‘世界模型’，能够通过自回归生成模拟状态转移动态（State Dynamics）。
然而，LLM 在复杂规划和状态跟踪任务中表现不佳，尤其是在状态空间较大或转移约束复杂时，存在显著局限性。
本文旨在探索 LLM 使用残差流捕捉状态动态的限制、预测失败的因果解释，以及是否存在特定机制建模顺序状态动态。

## Method

*   **研究目标与框架**：系统性研究基于 Transformer 的 LLM 如何表示和更新内部状态，聚焦于状态动态建模能力的规模限制和内部机制。
*   **任务设计**：设计了三个任务领域，均建模为确定性有限自动机（DFA），以测试不同维度的状态跟踪能力：
    *   **Box Tracking**：测试简单实体位置更新的动态状态跟踪能力，通过初始配置和移动指令预测最终位置。
    *   **Abstract DFA Sequences**：测试在明确定义但高度受限的转移动态下，模型对大组合状态空间的跟踪能力，需预测序列中最后动作后的状态。
    *   **Complex Text Games**：测试在自然语言情境下的复杂组合推理能力，通过分配和转移线索预测结果，涉及多变量状态更新和隐式约束推断。
*   **实验设置**：在不同状态数量（|Q|）和转移数量（|T|）的组合下测试 LLM 性能，评估其在规模扩展时的状态跟踪和转移历史建模能力。
*   **模型选择**：选用不同参数规模的 Transformer 模型家族（TinyStories, GPT-2, Pythia），从小型（8M 参数）到大型（1.5B 参数），以探索规模对性能的影响。
*   **评估方法**：
    *   **准确性评估**：通过下一状态预测准确性衡量 LLM 的表示能力，比较模型预测与 DFA 真实状态转移的匹配度。
    *   **激活补丁（Activation Patching）**：识别对状态跟踪行为贡献最大的模型组件（如层级、注意力头），通过清洁和损坏提示对的对比分析关键信息处理位置。
    *   **注意力模式分析**：聚合关键注意力头的注意力概率，研究是否存在专门处理状态历史或状态-动作依赖的‘状态动态头’，揭示信息在残差流中的传播机制。
*   **创新点**：结合任务复杂度、模型规模和机制解释性分析，从行为表现到内部机制多维度剖析 LLM 的状态动态建模能力。

## Experiment

*   **性能与规模关系**：实验表明模型规模与状态跟踪性能呈正相关，较大模型（如 GPT-2 XL, Pythia 1B）在 Box Tracking 和 Abstract DFA 任务中准确性更高，但在 Complex Text Games 任务中所有模型表现较差，显示语言复杂性和组合推理的挑战。
*   **状态与转移数量影响**：在状态数量少、转移数量多的场景下，模型准确性较高；而在状态数量多、转移数量少的场景下，模型难以推断转移动态，准确性显著下降，表明 LLM 对未充分探索的状态空间推断能力有限。
*   **机制解释性结果**：激活补丁和注意力模式分析显示状态跟踪并非由单一注意力头主导，而是多个注意力头（如 Induction Heads 和 Name-Mover Heads）协同作用，特定头在特定层级对状态历史和状态-动作依赖信息表现出强关注。
*   **实验设置评价**：实验设计全面，覆盖了不同任务复杂度、模型规模和状态-转移组合，数据清晰呈现规模效应和任务难度效应；但未深入探讨训练数据分布对性能的影响，对复杂任务中语言噪声的具体影响分析也较有限。

## Further Thoughts

将 LLM 视为世界模型的视角非常新颖，不仅限于文本生成，还可扩展至强化学习和决策任务，启发我们探索如何利用自回归能力模拟更广泛的环境动态；此外，状态跟踪依赖多注意力头协同的发现提示未来可通过针对性干预（如增强特定注意力头功能）提升模型性能；规模效应虽显著但非线性增长，尤其在复杂任务中受限，启发思考是否能通过高效训练方法或架构设计弥补规模不足。