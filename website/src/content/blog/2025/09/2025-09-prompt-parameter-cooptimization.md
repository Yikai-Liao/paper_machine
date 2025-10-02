---
title: "Prompt and Parameter Co-Optimization for Large Language Models"
pubDatetime: 2025-09-29T03:38:25+00:00
slug: "2025-09-prompt-parameter-cooptimization"
type: "arxiv"
id: "2509.24245"
score: 0.8455486624738704
author: "grok-3-latest"
authors: ["Xiaohe Bo", "Rui Li", "Zexu Sun", "Quanyu Dai", "Zeyu Zhang", "Zihang Tian", "Xu Chen", "Zhenhua Dong"]
tags: ["LLM", "Prompt Optimization", "Fine-Tuning", "Joint Optimization", "Knowledge Sharing"]
institution: ["Gaoling School of Artificial Intelligence, Renmin University of China", "Huawei Noah’s Ark Lab"]
description: "本文提出 MetaTuner 框架，通过联合优化提示和参数，显著提升大型语言模型的任务性能，并通过监督正则化损失解决离散-连续优化难题。"
---

> **Summary:** 本文提出 MetaTuner 框架，通过联合优化提示和参数，显著提升大型语言模型的任务性能，并通过监督正则化损失解决离散-连续优化难题。 

> **Keywords:** LLM, Prompt Optimization, Fine-Tuning, Joint Optimization, Knowledge Sharing

**Authors:** Xiaohe Bo, Rui Li, Zexu Sun, Quanyu Dai, Zeyu Zhang, Zihang Tian, Xu Chen, Zhenhua Dong

**Institution(s):** Gaoling School of Artificial Intelligence, Renmin University of China, Huawei Noah’s Ark Lab


## Problem Background

大型语言模型（LLMs）的性能提升主要依赖提示优化和微调两种策略，但两者通常独立研究，提示优化通过外部输入上下文激活模型能力，难以适应大规模任务特定数据，而微调通过内部参数更新适应任务，却对输入提示的选择极为敏感，次优提示可能导致性能下降；论文旨在设计一个统一框架，结合提示优化和微调，互补彼此的弱点，提升整体任务表现。

## Method

* **核心思想**：提出 MetaTuner 框架，通过联合优化提示和参数，充分利用两者的互补优势，提升 LLMs 的任务性能。
* **框架设计**：
  * 使用一个共享的元编码器（Meta Encoder）对输入查询进行编码，提取底层特征表示。
  * 编码后的特征分别输入到两个解码器：提示解码器（Prompt Decoder）生成离散的自然语言提示，参数解码器（Parameter Decoder）生成连续的下游模型参数（如基于 LoRA 的低秩更新）。
  * 共享参数机制（Shared-Private Parameterization）：元编码器的参数在两个解码器间共享，促进知识共享，同时保留各自的私有参数以保持优化灵活性。
* **优化策略**：
  * 针对提示生成的离散优化和参数生成的连续优化之间的矛盾，设计监督正则化损失（Supervised Regularization Loss），通过专家数据集（包含优质提示）和任务奖励指导提示生成器的优化。
  * 优化分为两种模式：MetaTuner-I（交替优化两个损失项）和 MetaTuner-J（统一优化整体损失），以适应不同任务需求。
* **实现细节**：
  * 提示生成器（G）基于大型语言模型（如 Qwen2.5-7B），将前 k 层作为元编码器，后续层作为提示解码器。
  * 参数生成器（F）基于 LoRA 技术，通过超网络生成下游模型的参数更新，使用矩阵乘法和 ReLU 激活函数处理隐藏状态。
* **关键创新**：将离散-连续混合优化问题转化为连续优化问题，通过共享知识和监督信号实现提示和参数的协同优化。

## Experiment

* **有效性**：在 MATH, GSM8K, HotpotQA 和 CosmosQA 四个数据集上，MetaTuner 显著优于单独的提示优化方法（如 RLPrompt, BPO）和微调方法（如 SFT, DPO），也优于其他混合方法（如 BetterTogether），在 Qwen2.5-7B 和 3B 模型下分别实现平均 10.15% 和 17.08% 的相对性能提升。
* **组件贡献**：消融实验表明，提示优化、参数优化和共享参数机制均对最终性能有贡献，去除任一组件都会导致性能下降（平均下降约 0.99%-1.12%），验证了联合优化的必要性。
* **泛化能力**：跨数据集测试（如在 MATH 等数据集训练，在 GSM8K 测试）显示 MetaTuner 仍优于基线，表明其对未见任务的适应能力较强。
* **效率分析**：参数生成增加了额外计算成本，但超网络规模较小，实际开销可控（如在 A100 GPU 上每查询平均生成时间为 0.03-0.53 秒）。
* **实验设置合理性**：数据集覆盖数学推理和问答任务，指标（如 EM 和 F1）与任务需求一致，基线选择全面（包括零样本、提示优化、微调和混合方法），实验设计合理且结果一致。

## Further Thoughts

MetaTuner 的共享-私有参数机制启发了我思考如何在其他混合优化问题中应用类似设计，例如多模态模型中输入和参数的协同优化；监督正则化损失的思路可以扩展到其他非可微优化场景，通过任务奖励和专家数据提供直接指导；此外，为每个查询动态生成定制化提示和参数的个性化适应策略，可能对提升模型在复杂动态任务中的表现有深远影响，值得探索如何结合 Mixture of Experts（MoE）范式进一步优化效率。