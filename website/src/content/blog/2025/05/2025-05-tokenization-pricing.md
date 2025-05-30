---
title: "Is Your LLM Overcharging You? Tokenization, Transparency, and Incentives"
pubDatetime: 2025-05-27T18:02:12+00:00
slug: "2025-05-tokenization-pricing"
type: "arxiv"
id: "2505.21627"
score: 0.7397562436170254
author: "grok-3-latest"
authors: ["Ander Artola Velasco", "Stratis Tsirtsis", "Nastaran Okati", "Manuel Gomez-Rodriguez"]
tags: ["LLM", "Tokenization", "Pricing Mechanism", "Transparency", "Incentive Compatibility"]
institution: ["Max Planck Institute for Software Systems"]
description: "本文通过委托-代理模型揭示了 LLM 云服务中按 token 计费机制的道德风险，提出高效启发式算法证明用户过收费漏洞，并设计按字符计费机制以消除提供商误报动机。"
---

> **Summary:** 本文通过委托-代理模型揭示了 LLM 云服务中按 token 计费机制的道德风险，提出高效启发式算法证明用户过收费漏洞，并设计按字符计费机制以消除提供商误报动机。 

> **Keywords:** LLM, Tokenization, Pricing Mechanism, Transparency, Incentive Compatibility

**Authors:** Ander Artola Velasco, Stratis Tsirtsis, Nastaran Okati, Manuel Gomez-Rodriguez

**Institution(s):** Max Planck Institute for Software Systems


## Problem Background

大型语言模型（LLM）云服务中普遍采用的按 token 计费（pay-per-token）定价机制存在信息不对称问题：用户无法观察模型生成输出的完整过程，而服务提供商可以通过误报（misreport）输出的 token 数量（例如将单词拆分为多个单字符 token）来多收费，造成道德风险（moral hazard）。
论文旨在揭示这一经济激励问题，分析其对用户和服务提供商的影响，并提出解决方案以保护用户利益。

## Method

*   **理论建模：** 作者将问题建模为委托-代理问题（principal-agent problem），其中用户为委托人，服务提供商为代理人，分析了按 token 计费机制如何激励提供商报告更长的 token 序列以增加收入。
*   **计算难度分析：** 证明了在透明性要求下（即提供商需公开生成过程中的概率分布），找到最长的看似合理（plausible）的 token 序列是一个 NP-Hard 问题，无法在多项式时间内最优解决。
*   **启发式算法：** 提出了一种高效的启发式算法（Algorithm 1），通过迭代拆分高索引 token（如基于词汇表索引选择拆分点），在不引起怀疑的情况下找到比真实输出更长的合理 token 序列，证明用户在当前机制下的脆弱性。
*   **替代机制设计：** 提出并理论证明了一种按字符计费（pay-per-character）的定价机制，该机制为激励相容（incentive-compatible），即提供商无法通过误报 token 序列获利，因为计费仅依赖于输出字符串的字符数，与 token 划分无关。

## Experiment

*   **有效性验证：** 实验表明，在按 token 计费机制下，若提供商将每个字符报告为单独 token，可过收费约 3 倍（过收费比例为 308%-345%，基于 Llama、Gemma、Ministral 模型在 LMSYS Chatbot Arena 平台 400 个提示上的测试）。
*   **算法表现：** 使用启发式算法时，过收费比例显著，例如在 Ministral-8B 模型（温度=1.3，top-p=0.99）上可达 13%，且算法在高温度和宽松采样参数下效果更佳。
*   **实验设置合理性：** 实验覆盖多个开源 LLM 模型（Llama、Gemma、Ministral 系列）、不同温度值和 top-p 参数，重复 5 次以提供 90% 置信区间，数据来源（LMSYS 平台）具有一定代表性；硬件配置明确（A100 GPU），生成过程细节（如输出长度 200-300 token）清晰。
*   **局限性：** 未测试专有模型，LMSYS 提示可能不完全代表真实用户分布，但整体实验设计全面，支持了理论分析和算法效果的验证。

## Further Thoughts

论文中透明性与隐私之间的权衡是一个值得关注的点：要求提供商公开生成过程可能泄露模型内部机制，损害商业利益；这启发我思考是否可以通过第三方审计或加密技术（如零知识证明）验证 token 序列合理性，而无需暴露模型细节。此外，按字符计费机制在多语言场景下可能面临字符定义和价值不一致的问题（例如中文字符与拉丁字母），这为未来研究提供了方向。