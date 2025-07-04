---
title: "GPT, But Backwards: Exactly Inverting Language Model Outputs"
pubDatetime: 2025-07-02T13:20:30+00:00
slug: "2025-07-input-reconstruction-soda"
type: "arxiv"
id: "2507.01693"
score: 0.6024198500461546
author: "grok-3-latest"
authors: ["Adrians Skapars", "Edoardo Manino", "Youcheng Sun", "Lucas C. Cordeiro"]
tags: ["LLM", "Input Reconstruction", "Optimization", "Privacy", "Auditing"]
institution: ["University of Manchester", "Mohamed bin Zayed University of Artificial Intelligence", "Federal University of Amazonas"]
description: "本文提出 SODA 算法，通过连续松弛和改进的 Adam 优化，从 LLM 输出（尤其是 logits）中高效重建输入，为取证审计提供了新工具，并揭示了输出信息暴露对隐私的潜在威胁。"
---

> **Summary:** 本文提出 SODA 算法，通过连续松弛和改进的 Adam 优化，从 LLM 输出（尤其是 logits）中高效重建输入，为取证审计提供了新工具，并揭示了输出信息暴露对隐私的潜在威胁。 

> **Keywords:** LLM, Input Reconstruction, Optimization, Privacy, Auditing

**Authors:** Adrians Skapars, Edoardo Manino, Youcheng Sun, Lucas C. Cordeiro

**Institution(s):** University of Manchester, Mohamed bin Zayed University of Artificial Intelligence, Federal University of Amazonas


## Problem Background

大型语言模型（LLM）在广泛应用中表现出强大能力，但其安全性和隐私问题日益凸显，尤其是在输出可能泄露输入信息的情况下。
本文从取证视角出发，聚焦于精确输入重建（Exact Input Reconstruction）问题，即从已知的 LLM 输出（如文本或 logits）中重建导致该输出的原始输入，以支持事后分析、隐私泄露检测及假输出报告的验证。
这一问题在隐私保护机器学习（防止输入逆向推导）和闭源 LLM API（防止系统提示泄露）中尤为关键，而现有审计技术（如对抗攻击、越狱攻击）无法实现精确重建，因为它们通常寻找任意触发输出的输入，而非原始输入。

## Method

*   **问题形式化**：将输入重建定义为离散优化问题，目标是找到输入 x，使得模型输出 f(x) 与目标输出 y 的距离最小，并确保在原始输入处有唯一全局最小值。
*   **连续松弛策略**：由于直接在离散输入空间（词汇表大小的指数级空间）搜索不可行，提出将输入的 one-hot 编码松弛为连续概率分布，通过 SoftMax 函数对辅助变量 z 进行参数化，从而在连续空间中进行优化。
*   **SODA 算法**：提出 Sparse One-hot Discrete Adam（SODA）算法，基于 Adam 优化器迭代更新辅助变量 z，使用权重衰减（weight decay）防止过拟合，并通过周期性重启（periodic resets）避免陷入局部最优；优化过程中定期将当前解离散化为输入 token，若损失接近零则终止搜索。
*   **目标函数设计**：定义两种目标函数，基于纯文本输出（Φ_text）仅比较输出 token 序列，基于 logits（Φ_logit）则利用输出 token 的概率分布信息以提高重建精度。
*   **关键创新**：通过连续松弛将离散问题转化为可微优化问题，避免暴力搜索的计算不可行性，同时改进 Adam 优化以适应输入重建任务。

## Experiment

*   **实验设置**：在多个 LLM（如 TinyStories-33M, GPT-2 系列, Qwen-2.5 系列）上测试，数据集包括随机输入（Random）、自然语言输入（NL ID 和 OOD）及隐私数据（Privacy），对比算法包括 GCG 和黑盒反演模型；主要指标为精确匹配率（Exact Match）和部分匹配率（Partial Match）。
*   **有效性**：SODA 在 logits 条件下表现优异，对短输入（1-3 token）的 Random 数据集精确匹配率达 79.5%，对自然语言 ID 数据集高达 98.1%，且无假阳性；但在纯文本输出条件下效果较差（仅 3.6%）。
*   **优越性**：相比 GCG（11.8%）和黑盒反演模型（3.9%），SODA 在 logits 条件下的精确匹配率显著更高；消融实验验证了权重衰减和周期性重启等组件的重要性。
*   **局限性**：对长输入（15+ token）重建效果不佳，尤其在隐私数据上 PII 匹配率仅 3.0%；计算成本较高，随模型大小和迭代次数增加（总计 5400 GPU 小时）。
*   **合理性**：实验设置全面，覆盖不同模型、输入类型和长度，数据量充足（10K 样本/数据集）；结果表明当前部署实践（如隐藏 logits）对防止恶意反演有一定保护作用。

## Further Thoughts

论文启发了我对 LLM 隐私保护和审计的新思考：首先，连续松弛策略不仅适用于输入重建，还可能推广到提示优化或对抗样本生成等领域；其次，输出 logits 暴露的信息量直接影响隐私泄露风险，提示我们在设计 LLM API 时应严格控制输出粒度（如避免暴露 top-k logits）；最后，从取证视角进行审计的思路为责任追溯和安全检测提供了新方向，例如通过类似 SODA 的方法检测诽谤攻击（slander attacks），这可能进一步推动‘反向不可逆’模型架构（如输出噪声）的设计。