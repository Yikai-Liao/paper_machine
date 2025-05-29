---
title: "DOGe: Defensive Output Generation for LLM Protection Against Knowledge Distillation"
pubDatetime: 2025-05-26T04:31:38+00:00
slug: "2025-05-defensive-output-generation"
type: "arxiv"
id: "2505.19504"
score: 0.6416418539223867
author: "grok-3-latest"
authors: ["Pingzhi Li", "Zhen Tan", "Huaizhi Qu", "Huan Liu", "Tianlong Chen"]
tags: ["LLM", "Proxy Model", "Distillation", "Sampling", "Reasoning"]
institution: ["The University of North Carolina at Chapel Hill", "Arizona State University"]
description: "本文提出Defensive Output Generation（DOGe）方法，通过对抗性微调最终线性层，使大型语言模型输出对合法用户保持有用性，同时显著误导知识蒸馏过程，保护模型知识产权。"
---

> **Summary:** 本文提出Defensive Output Generation（DOGe）方法，通过对抗性微调最终线性层，使大型语言模型输出对合法用户保持有用性，同时显著误导知识蒸馏过程，保护模型知识产权。 

> **Keywords:** LLM, Proxy Model, Distillation, Sampling, Reasoning

**Authors:** Pingzhi Li, Zhen Tan, Huaizhi Qu, Huan Liu, Tianlong Chen

**Institution(s):** The University of North Carolina at Chapel Hill, Arizona State University


## Problem Background

大型语言模型（LLMs）作为重要的知识产权，其公开API输出的文本可能被竞争对手通过知识蒸馏（Knowledge Distillation, KD）廉价模仿，导致模型能力泄露和开发者竞争优势受损。
现有防御方法如水印仅能事后检测，而其他主动防御策略往往假设蒸馏涉及内部logits匹配，无法应对仅基于输出文本的攻击，因此需要在API访问限制下，设计一种方法生成对合法用户有用但对蒸馏具有误导性的输出。

## Method

*   **核心思想:** 提出Defensive Output Generation（DOGe），通过对抗性训练调整教师模型的输出行为，使其对合法用户保持准确和有用，但对试图进行蒸馏的学生模型产生误导。
*   **具体实现:** 
    *   **训练目标:** 使用结合监督微调损失（L_SFT）和对抗性损失（L_adv）的总损失函数（L_total = L_SFT + λ·L_adv），其中L_SFT通过交叉熵损失确保教师模型任务性能，L_adv通过最大化教师输出分布与一组代理学生模型（Proxy Student Models）预测分布的KL散度，使输出分布难以被模仿。
    *   **推理感知掩码（Reasoning-Aware Masking）:** 在训练中对中间推理步骤的token应用对抗性损失，而对最终答案token仅应用SFT损失，确保最终输出正确性，同时让推理过程对蒸馏具有误导性。
    *   **高效微调:** 仅对教师模型的最终线性层（LM Head）进行参数高效微调（PEFT），保持底层模型权重冻结，减少训练和内存开销。
    *   **部署优势:** 防御特性嵌入模型权重中，与解码策略（如greedy, sampling）无关，适用于API场景和开源模型发布。
*   **关键创新:** 不依赖推理时干预，通过模型输出分布的对抗性调整实现防御，同时兼顾效率和实用性。

## Experiment

*   **有效性:** 防御性教师模型在多个任务上保持甚至提升性能（如DeepSeek-R1在GSM8K上提升1.5%，Qwen3-8B提升2.1%），而从防御性教师模型蒸馏的学生模型性能显著下降（如Llama-3.2-1B在GSM8K下降12.9%，CSQA下降高达39.0%），证明DOGe在保护模型能力上的显著效果。
*   **实验设置合理性:** 实验覆盖数学推理（GSM8K, MATH）和常识推理（ARC, CSQA）任务，包含域内和域外数据集，验证了跨领域泛化性；使用多种教师模型（DeepSeek-R1, Qwen3-8B）和学生模型（Llama-3.2-1B, Gemma-3-1b-it），结果一致性高。
*   **消融研究:** 调整对抗性损失系数λ显示性能与防御效果的权衡，λ=3e-5为最优；单一代理模型与多代理模型效果接近，计算开销可控；不同训练数据集（GSM8K vs Tulu）均有效，Tulu因多样性带来更强学生性能下降。
*   **局限性:** 教师模型在某些常识任务（如ARC, CSQA）性能略有下降，可能因对抗性训练干扰非目标任务推理。

## Further Thoughts

推理感知掩码的设计非常启发性，通过区分中间推理步骤和最终答案token，针对性调整输出分布，未来是否可以根据推理步骤的重要性或复杂性动态调整对抗性损失强度？此外，仅微调LM Head的思路是否可扩展到其他层（如注意力层）以实现更深层次防御？另一个思考是，面对攻击者可能的数据清洗或对抗性训练反制，防御与攻击的博弈是否能进一步通过多轮迭代优化防御策略？