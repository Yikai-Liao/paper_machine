---
title: "Goedel-Prover-V2: Scaling Formal Theorem Proving with Scaffolded Data Synthesis and Self-Correction"
pubDatetime: 2025-08-05T16:28:22+00:00
slug: "2025-08-theorem-proving-self-correction"
type: "arxiv"
id: "2508.03613"
score: 0.5720483436653248
author: "grok-3-latest"
authors: ["Yong Lin", "Shange Tang", "Bohan Lyu", "Ziran Yang", "Jui-Hui Chung", "Haoyu Zhao", "Lai Jiang", "Yihan Geng", "Jiawei Ge", "Jingruo Sun", "Jiayun Wu", "Jiri Gesi", "Ximing Lu", "David Acuna", "Kaiyu Yang", "Hongzhou Lin", "Yejin Choi", "Danqi Chen", "Sanjeev Arora", "Chi Jin"]
tags: ["LLM", "Theorem Proving", "Self-Correction", "Data Synthesis", "Model Averaging"]
institution: ["Princeton University", "NVIDIA", "Tsinghua University", "Stanford University", "Meta FAIR", "Amazon", "Shanghai Jiao Tong University", "Peking University"]
description: "本文提出Goedel-Prover-V2，通过验证器引导的自我修正、脚手架式数据合成和模型平均策略，在较小模型规模和计算预算下显著提升自动化定理证明性能，超越现有开源和部分闭源系统。"
---

> **Summary:** 本文提出Goedel-Prover-V2，通过验证器引导的自我修正、脚手架式数据合成和模型平均策略，在较小模型规模和计算预算下显著提升自动化定理证明性能，超越现有开源和部分闭源系统。 

> **Keywords:** LLM, Theorem Proving, Self-Correction, Data Synthesis, Model Averaging

**Authors:** Yong Lin, Shange Tang, Bohan Lyu, Ziran Yang, Jui-Hui Chung, Haoyu Zhao, Lai Jiang, Yihan Geng, Jiawei Ge, Jingruo Sun, Jiayun Wu, Jiri Gesi, Ximing Lu, David Acuna, Kaiyu Yang, Hongzhou Lin, Yejin Choi, Danqi Chen, Sanjeev Arora, Chi Jin

**Institution(s):** Princeton University, NVIDIA, Tsinghua University, Stanford University, Meta FAIR, Amazon, Shanghai Jiao Tong University, Peking University


## Problem Background

自动化定理证明（Automated Theorem Proving, ATP）是人工智能领域的一个重大挑战，要求AI系统在形式化语言（如Lean）中构建严谨的、可验证的证明。
现有方法（如DeepSeek-Prover-V2）虽在数学竞赛基准（如MiniF2F, PutnamBench）上取得了进展，但往往依赖大规模模型（数百亿参数）和高计算成本的推理过程，限制了效率和可访问性。
本文旨在探索如何在较小的模型规模和较低的测试时计算预算下，实现与大型模型相当甚至更优的性能，同时提升开源定理证明器的能力。

## Method

*   **Verifier-Guided Self-Correction（验证器引导的自我修正）**：
    *   核心思想是将Lean编译器的反馈（如错误信息）整合到模型输入中，使模型能够迭代修正其生成的证明。
    *   具体实现上，模型在初始证明尝试后，解析验证失败的信息并将其作为指导输入，生成修复后的证明，形成迭代自我修正过程。
    *   创新点在于将长链式推理（Chain-of-Thought, CoT）与验证器反馈结合，特别适用于复杂定理证明任务。
*   **Scaffolded Data Synthesis（脚手架式数据合成）**：
    *   目标是通过生成难度逐步增加的合成数学问题，为模型提供更好的学习信号。
    *   包括两种方法：一是形式化方法，利用Lean系统提取未解决子目标作为较简单的问题；二是非形式化方法，利用大型语言模型（LLM）生成自然语言中的简单或复杂变体问题，随后形式化为Lean语句。
    *   数据合成还包括质量过滤和难度评估，确保生成的问题既正确又具有适当挑战性。
*   **Model Averaging（模型平均）**：
    *   针对训练后期模型输出多样性下降的问题，通过合并基础模型和微调模型的参数（按一定比例加权平均），提升高采样预算下的性能（如pass@N）。
    *   具体操作是在监督微调（SFT）和强化学习（RL）阶段完成后，分别进行模型平均，确保最终模型兼具准确性和多样性。
*   **整体训练管道**：
    *   结合监督微调（SFT）、专家迭代（Expert Iteration）和强化学习（RL），基于DeepSeek-Prover-V2等模型生成初始数据集，逐步优化模型性能。
    *   RL采用多任务设置，同时优化完整证明生成和自我修正能力，并通过动态采样策略聚焦于中等难度问题。

## Experiment

*   **性能提升显著**：Goedel-Prover-V2-32B在MiniF2F基准上的pass@32准确率为88.1%，启用自我修正后提升至90.4%，超越DeepSeek-Prover-V2-671B（82.4%）和Kimina-Prover-72B（84.0%），尽管参数规模仅为前者的1/20。
    *   更小规模的Goedel-Prover-V2-8B也达到84.6%，几乎与70B模型相当，显示出方法在小模型上的高效性。
*   **效率优势**：在PutnamBench上，32B模型在pass@32下解决43个问题（自我修正后为57个），在pass@184下解决86个问题，远超DeepSeek-Prover-V2-671B（pass@1024下47个），表明其在低采样预算下的推理效率更高。
*   **实验设置全面**：实验覆盖多个基准（MiniF2F, PutnamBench, MathOlympiadBench），测试了不同模型规模（8B和32B）、不同采样预算（pass@32至pass@8192）以及自我修正的效果。
    *   消融研究验证了自我修正（提升约2个百分点）和模型平均（优化pass@N）的有效性。
    *   扩展上下文长度（至128k token）和增加修正轮次（至5轮）后，MiniF2F的pass@32准确率进一步提升至92.7%，展现了迭代修正的潜力。
*   **局限性**：论文未详细探讨方法在极高难度问题上的表现，也未充分披露训练所需的计算资源和时间，可能影响可重复性评估。

## Further Thoughts

自我修正机制的潜力令人印象深刻，验证器反馈与长链式推理的结合不仅适用于定理证明，还可能推广至其他需要迭代优化的任务（如代码生成或多步推理问题），是否可以通过更精细的反馈解析（如提取错误模式或优先级）进一步提升效率？
脚手架式数据合成提供了一种引导模型学习的通用思路，是否可以应用于自然语言推理或知识图谱构建等领域，通过自动化难度评估优化学习路径？
模型平均策略在缓解多样性下降方面的成功，启发我们思考是否可以通过动态参数融合（如基于任务难度的自适应权重）进一步提升性能，尤其是在多任务或多领域场景中。