---
title: "Your Language Model Can Secretly Write Like Humans: Contrastive Paraphrase Attacks on LLM-Generated Text Detectors"
pubDatetime: 2025-05-21T10:08:39+00:00
slug: "2025-05-contrastive-paraphrase-attack"
type: "arxiv"
id: "2505.15337"
score: 0.6825727880075236
author: "grok-3-latest"
authors: ["Hao Fang", "Jiawei Kong", "Tianqu Zhuang", "Yixiang Qiu", "Kuofeng Gao", "Bin Chen", "Shu-Tao Xia", "Yaowei Wang", "Min Zhang"]
tags: ["LLM", "Paraphrase Attack", "Text Detection", "Contrastive Learning", "Token Distribution"]
institution: ["Tsinghua Shenzhen International Graduate School, Tsinghua University", "Harbin Institute of Technology, Shenzhen", "Pengcheng Laboratory"]
description: "本文提出对比改写攻击（CoPA），通过对比人类和机器风格分布调整 LLM 解码概率，高效欺骗文本检测器，同时保持语义一致性，显著揭示当前检测器脆弱性。"
---

> **Summary:** 本文提出对比改写攻击（CoPA），通过对比人类和机器风格分布调整 LLM 解码概率，高效欺骗文本检测器，同时保持语义一致性，显著揭示当前检测器脆弱性。 

> **Keywords:** LLM, Paraphrase Attack, Text Detection, Contrastive Learning, Token Distribution

**Authors:** Hao Fang, Jiawei Kong, Tianqu Zhuang, Yixiang Qiu, Kuofeng Gao, Bin Chen, Shu-Tao Xia, Yaowei Wang, Min Zhang

**Institution(s):** Tsinghua Shenzhen International Graduate School, Tsinghua University, Harbin Institute of Technology, Shenzhen, Pengcheng Laboratory


## Problem Background

大型语言模型（LLMs）的滥用，如学术剽窃和虚假信息生成，促使了检测器的发展以识别其生成的文本；然而，攻击者通过改写攻击（paraphrase attacks）试图绕过检测，现有的方法需要大量资源训练专用改写模型，且对先进检测算法效果有限，因此本文提出了一种无需训练的高效攻击方法，揭示检测器脆弱性并推动更鲁棒技术发展。

## Method

* **核心思想**：通过对比人类风格和机器风格的词分布，利用现成的大型语言模型（LLM）动态调整解码过程中的 token 概率分布，生成更接近人类写作的文本以欺骗检测器。
* **具体实现**：
  * **提示词设计**：设计两类提示词，分别引导 LLM 生成人类风格（human-like）和机器风格（machine-like）的文本分布。
  * **对比调整**：在生成每个 token 时，计算人类风格分布（p_h）和机器风格分布（p_m），通过公式 p_c ∝ exp((1+λ)*f_θ(human) - λ*f_θ(machine)) 调整最终概率分布，其中 λ 调节对比强度，削弱机器特征。
  * **自适应截断**：引入自适应截断机制，仅从人类风格分布的高置信度 token 池中采样，确保生成文本的连贯性和语义一致性。
  * **理论支持**：基于 KL 散度的分析证明，对比调整后的分布 p_c 比单纯的人类风格分布更接近真实人类写作分布。
* **关键优势**：无需训练专用模型，仅通过现成 LLM 的两次前向推理实现，显著降低计算成本，同时保持攻击效果和文本质量。

## Experiment

* **攻击效果**：CoPA 在三个数据集（XSum, SQuAD, LongQA）上针对 8 种检测算法（如 Fast-DetectGPT）测试，平均欺骗率提升 57.72%（FPR=5%），显著优于基线方法 Dipper 和 Raidar-A，尤其对先进检测器效果突出。
* **语义一致性**：改写后文本语义相似度平均超过 90%，表明攻击未牺牲文本质量。
* **鲁棒性与全面性**：CoPA 对多种源 LLM（如 GPT-4, Claude-3.5）均有效，在不同 FPR（如 1%）下保持优越性能；实验覆盖多种数据集、模型和检测算法，设置合理且具代表性。
* **消融研究**：验证了对比系数 λ 的最佳值（0.5）及单次改写的优越性，避免多次改写导致的语义漂移。
* **计算开销**：每次 token 生成需两次前向推理，增加推理延迟，可能限制实时应用。

## Further Thoughts

CoPA 的对比思想（human-like vs. machine-like）可扩展至其他生成任务以增强多样性或风格逼真度；其无需训练的攻击范式启发我们在对抗样本生成等安全领域利用预训练模型潜力；此外，CoPA 揭示的检测器局限性提示未来可通过对抗训练增强检测鲁棒性。