---
title: "TuCo: Measuring the Contribution of Fine-Tuning to Individual Responses of LLMs"
pubDatetime: 2025-06-29T23:08:36+00:00
slug: "2025-06-tuco-fine-tuning-contribution"
type: "arxiv"
id: "2506.23423"
score: 0.7724588936390374
author: "grok-3-latest"
authors: ["Felipe Nuti", "Tim Franzmeyer", "João Henriques"]
tags: ["LLM", "Fine-Tuning", "Model Decomposition", "Safety Analysis", "Inference Time"]
institution: ["University of Oxford"]
description: "本文提出 TuCo 方法，通过分解模型为预训练和微调组件，量化微调对大型语言模型单个提示输出的贡献，并在解释性和安全性分析中展现显著价值。"
---

> **Summary:** 本文提出 TuCo 方法，通过分解模型为预训练和微调组件，量化微调对大型语言模型单个提示输出的贡献，并在解释性和安全性分析中展现显著价值。 

> **Keywords:** LLM, Fine-Tuning, Model Decomposition, Safety Analysis, Inference Time

**Authors:** Felipe Nuti, Tim Franzmeyer, João Henriques

**Institution(s):** University of Oxford


## Problem Background

大型语言模型（LLMs）经过微调（fine-tuning）后，其输出和行为受到微调的显著影响，但现有研究缺乏对单个提示（prompt）级别上微调贡献的量化方法。
论文旨在解决这一问题，特别是在安全性和指令遵循能力方面，因为微调常用于增强这些特性，而越狱攻击（jailbreak attacks）可能通过削弱微调效果来绕过安全限制，造成潜在风险。

## Method

*   **核心思想:** 提出一种名为 Tuning Contribution (TuCo) 的方法，通过将微调后的模型分解为预训练组件（Pre-Training Component, PTC）和微调组件（Fine-Tuning Component, FTC），量化微调对单个提示输出的贡献。
*   **具体实现:** 
    *   **模型分解:** 基于 Transformer 的残差结构，将每一层输出分解为预训练模型的输出（PTC）以及微调带来的增量输出（FTC，即微调模型输出与预训练输出的差值）。
    *   **贡献计算:** TuCo 定义为 FTC 在所有层累积输出的幅度与 PTC 和 FTC 总幅度的比值，特别关注最后一个 token 的隐藏状态（直接影响输出），以捕捉微调对最终输出的影响。
    *   **理论依据:** 通过离散 Grönwall 不等式推导了一个界限，证明 FTC 的相对幅度可以限制预训练和微调模型最终隐藏状态的差异，为 TuCo 的定义提供理论支持。
    *   **行为调控实验:** 提出 FTC α-Scaling 方法，通过调整 FTC 的幅度（即乘以一个缩放因子 α），在推理时动态控制模型行为和性能，验证 TuCo 的有效性和解释力。
*   **优势与特点:** 该方法无需修改模型参数，仅在推理时计算，适用于大规模模型；同时，TuCo 提供了一个直观的百分比解释，方便理解微调对输出的影响程度。

## Experiment

*   **有效性:** TuCo 成功量化了微调对模型输出的影响，通过 FTC α-Scaling 调整 FTC 幅度，模型在 MMLU 基准测试上的性能提升高达 5%，在特定行为（如基督教信仰认同）上的表现变化高达 24%。
*   **输入类型区分:** 在预训练数据（OpenWebText）和聊天数据（HH-RLHF）上，TuCo 值差异显著（AUC 得分高达 1.0），表明其能有效区分预训练和微调的影响。
*   **越狱攻击分析:** 三种越狱攻击（GCG, Conjugate Prompting, Many-Shot Jailbreaking）显著降低 TuCo 值，尤其在攻击成功时 TuCo 更低（AUC 得分高达 0.94），验证了攻击通过削弱微调影响来绕过安全机制的假设。
*   **实验设置合理性:** 实验覆盖多个开源模型（Llama 2, Llama 3, Vicuna, Mistral 等，参数规模至 13B），涉及多种任务和攻击类型，数据量充足（如 MWE 包含 1000 个问题），并采用交叉验证确保结果稳健；唯一的局限是需要访问预训练和微调模型，但对开源模型和开发者而言可行。

## Further Thoughts

TuCo 的分解思想（将模型输出分解为预训练和微调组件）可以扩展到分析不同训练阶段或数据集的影响；FTC α-Scaling 提示推理时动态干预可能成为增强安全性的手段，是否可以设计自适应机制根据 TuCo 值实时调整模型行为？此外，越狱攻击削弱微调影响的发现启发我们可以在微调阶段引入对抗性训练，提升模型对低资源语言或多示例攻击的鲁棒性。