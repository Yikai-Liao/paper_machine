---
title: "Your Language Model Can Secretly Write Like Humans: Contrastive Paraphrase Attacks on LLM-Generated Text Detectors"
pubDatetime: 2025-05-21T10:08:39+00:00
slug: "2025-05-contrastive-paraphrase-attack"
type: "arxiv"
id: "2505.15337"
score: 0.6825727880075236
author: "grok-3-latest"
authors: ["Hao Fang", "Jiawei Kong", "Tianqu Zhuang", "Yixiang Qiu", "Kuofeng Gao", "Bin Chen", "Shu-Tao Xia", "Yaowei Wang", "Min Zhang"]
tags: ["LLM", "Paraphrase Attack", "Text Detection", "Contrastive Decoding", "Sampling"]
institution: ["Tsinghua Shenzhen International Graduate School, Tsinghua University", "Harbin Institute of Technology, Shenzhen", "Pengcheng Laboratory"]
description: "本文提出对比改写攻击（CoPA），通过对比人类和机器风格分布动态调整LLM输出概率，无需训练即可生成更具人类风格的文本，显著提高对文本检测器的愚弄率。"
---

> **Summary:** 本文提出对比改写攻击（CoPA），通过对比人类和机器风格分布动态调整LLM输出概率，无需训练即可生成更具人类风格的文本，显著提高对文本检测器的愚弄率。 

> **Keywords:** LLM, Paraphrase Attack, Text Detection, Contrastive Decoding, Sampling

**Authors:** Hao Fang, Jiawei Kong, Tianqu Zhuang, Yixiang Qiu, Kuofeng Gao, Bin Chen, Shu-Tao Xia, Yaowei Wang, Min Zhang

**Institution(s):** Tsinghua Shenzhen International Graduate School, Tsinghua University, Harbin Institute of Technology, Shenzhen, Pengcheng Laboratory


## Problem Background

大型语言模型（LLMs）的滥用，如学术剽窃和虚假信息生成，促使了检测器的发展以识别LLM生成的文本。然而，攻击者通过改写攻击（paraphrase attacks）绕过检测，现有的改写方法需要大量数据和计算资源训练专用模型，且在面对先进检测算法时效果下降。因此，论文提出了一种无需训练的攻击方法，利用现成的LLM生成更接近人类风格的文本以欺骗检测器。

## Method

* **核心思想**：通过对比人类风格和机器风格的词分布，动态调整LLM在解码过程中的输出概率，生成更具人类风格的文本以绕过检测器。
* **具体实现步骤**：
  * **人类风格提示设计**：精心设计输入提示（prompt），引导LLM生成接近人类写作的词分布（p_h'），以减少机器特征。
  * **机器风格提示设计**：设计另一组提示，引导LLM生成高度机器风格的词分布（p_m），作为负面参考，突出机器特征。
  * **对比调整机制**：在解码的每一步，利用对比公式 p_c ∝ exp((1+λ)*f_θ(·|x_h, ...) - λ*f_θ(·|x_m, ...))，从人类风格分布中减去机器风格特征，生成调整后的分布（p_c）。其中，λ为调节参数，控制对比强度。
  * **自适应截断**：引入自适应截断机制，限制候选词范围，确保生成文本的语义一致性和连贯性，避免因概率调整导致不合理词的选择。
* **理论支持**：通过KL散度分析，证明调整后的分布p_c比单纯的人类风格分布p_h'更接近真实人类分布p_h，验证了方法的有效性。
* **优势**：无需训练专用改写模型，仅通过现成LLM和提示设计即可实现攻击，显著降低计算成本和资源需求。

## Experiment

* **攻击有效性**：CoPA在多个数据集（XSum, SQuAD, LongQA）和8种检测算法（如Fast-DetectGPT, TOCSIN）上显著降低了真阳性率（TPR），例如在GPT-3.5-turbo生成的文本上，针对Fast-DetectGPT的愚弄率平均提升57.72%（FPR=5%），远超基线方法Dipper和Raidar-A。
* **语义一致性**：生成的改写文本保持了高语义相似度（平均超过90%），确保攻击的同时保留原文含义。
* **鲁棒性与普适性**：CoPA对不同源LLM（如GPT-4, Claude-3.5）生成的文本均表现出色，且在不同改写模型（如Qwen2.5-72B）上效果稳定。
* **实验设置合理性**：实验覆盖多种风格数据集和检测方法（零样本和训练型），并通过消融研究验证了参数λ和多次改写的效果，设置全面合理。
* **不足与局限**：实验未探讨多语言适用性，对比机制引入额外推理延迟（两次前向计算），可能限制实时应用。

## Further Thoughts

CoPA通过对比人类和机器风格分布调整LLM输出的思路，不仅适用于改写攻击，还可能启发其他生成任务（如风格迁移或多样性提升）；其无需训练的攻击范式提示我们探索类似方法在其他对抗性任务中的应用；动态解码调整机制也可能用于优化LLM推理时的其他特性（如减少偏见或提高准确性）；此外，论文揭示检测器脆弱性，激励开发结合上下文或多模态特征的更鲁棒检测方法。