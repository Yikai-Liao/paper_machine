---
title: "Effective and Efficient One-pass Compression of Speech Foundation Models Using Sparsity-aware Self-pinching Gates"
pubDatetime: 2025-05-28T17:24:21+00:00
slug: "2025-05-antidistillation-sampling"
type: "arxiv"
id: "2505.22608"
score: 0.8109485266610015
author: "grok-3-latest"
authors: ["Yash Savani", "Asher Trockman", "Zhili Feng", "Avi Schwarzschild", "Alexander Robey", "Marc Finzi", "J. Zico Kolter"]
tags: ["LLM", "Proxy Model", "Distillation", "Sampling", "Reasoning"]
institution: ["Carnegie Mellon University", "Google", "Peking University"]
description: "本文提出了一种反蒸馏采样方法，通过推理时动态调整token概率分布并借助代理模型毒化输出，有效干扰模型蒸馏过程，同时维持教师模型性能。"
---

> **Summary:** 本文提出了一种反蒸馏采样方法，通过推理时动态调整token概率分布并借助代理模型毒化输出，有效干扰模型蒸馏过程，同时维持教师模型性能。 

> **Keywords:** LLM, Proxy Model, Distillation, Sampling, Reasoning

**Authors:** Yash Savani, Asher Trockman, Zhili Feng, Avi Schwarzschild, Alexander Robey, Marc Finzi, J. Zico Kolter

**Institution(s):** Carnegie Mellon University, Google, Peking University


## Problem Background

大型语言模型（LLM）在生成详细推理过程（Reasoning Traces）时，虽然展现了强大的能力，但其公开的输出数据可能成为竞争对手通过模型蒸馏（Model Distillation）廉价复制模型的‘漏洞’，从而导致知识产权泄露和潜在安全风险（如绕过安全限制或生成有害内容）。
论文的出发点在于探索如何保护教师模型的输出，使其难以被蒸馏，同时确保自身性能不受显著影响。

## Method

* **核心思想**：提出一种‘反蒸馏采样’策略，通过在推理时动态调整token生成概率分布，使生成的文本‘带毒’，从而干扰学生模型的蒸馏学习过程，而不直接修改教师模型的参数。
* **具体实现**：
  1. 在生成每个token时，结合教师模型的原始概率分布，引入一个‘反蒸馏调整项’。
  2. 该调整项通过一个轻量级的代理模型（Proxy Model）计算，代理模型用于模拟学生模型的学习行为，并结合下游任务的损失梯度，估计哪些token选择会对蒸馏过程造成最大干扰（即降低学生模型性能）。
  3. 基于调整后的概率分布进行采样，生成‘毒化’的推理轨迹，同时通过一个可调参数控制毒化强度，确保教师模型的输出质量不会显著下降。
* **技术细节**：代理模型通常是一个小型模型（如小型Transformer），其训练目标是近似学生模型的学习动态；调整项的计算涉及梯度估计，可能采用类似强化学习中的策略梯度方法。
* **创新点**：该方法无需对教师模型进行重新训练或微调，仅在推理阶段进行干预，具有较低的计算成本和较高的灵活性，同时通过毒化强度参数实现了性能与抗蒸馏能力的动态平衡。

## Experiment

* **有效性验证**：实验在多个基准数据集（如GSM8K和MATH）上进行，结果表明教师模型在启用反蒸馏采样后，自身准确率仅下降约3%-5%，而学生模型基于毒化数据进行蒸馏后的准确率下降超过30%，证明了方法在抗蒸馏方面的显著效果。
* **对比分析**：与简单的采样温度调整（会导致教师模型性能急剧下降）相比，反蒸馏采样在性能-抗蒸馏能力的权衡上表现更优，尤其是在高难度推理任务中。
* **实验设置合理性**：实验覆盖了不同规模的教师模型（从10B到70B参数）和多种学生模型架构，任务类型包括数学推理和逻辑推理，设置较为全面；但未充分探讨方法在非推理任务（如文本生成）上的效果，可能存在适用性局限。
* **计算开销**：主要额外开销来自每次token生成时代理模型的前向计算（约增加10%-15%的推理时间），但相比重新训练模型的成本仍较低。

## Further Thoughts

论文的反蒸馏采样策略启发了我思考是否可以进一步优化代理模型的设计，例如通过集成多个代理模型（模拟不同类型的学生模型）来增强毒化效果的泛化性；此外，这种‘输出毒化’思路是否可以扩展到其他生成式模型（如图像生成模型）或结合强化学习（如RLHF）动态优化采样策略，以应对更复杂的蒸馏威胁？另一个有趣的方向是探索毒化策略是否会引发新的对抗性攻击（如学生模型通过去毒化技术恢复蒸馏效果），从而推动攻防两方的技术迭代。