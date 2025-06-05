---
title: "Rethinking Post-Unlearning Behavior of Large Vision-Language Models"
pubDatetime: 2025-06-03T07:28:22+00:00
slug: "2025-06-post-unlearning-behavior"
type: "arxiv"
id: "2506.02541"
score: 0.5262055590039437
author: "grok-3-latest"
authors: ["Minsung Kim", "Nakyeong Yang", "Kyomin Jung"]
tags: ["LVLM", "Machine Unlearning", "Privacy Protection", "Behavior Guidance", "Generative Models"]
institution: ["Seoul National University"]
description: "本文提出 PUBG 方法，通过引导遗忘后输出分布，确保大型视觉-语言模型在保护隐私的同时生成信息性、视觉相关的响应，有效缓解了现有遗忘方法的‘遗忘后遗症’。"
---

> **Summary:** 本文提出 PUBG 方法，通过引导遗忘后输出分布，确保大型视觉-语言模型在保护隐私的同时生成信息性、视觉相关的响应，有效缓解了现有遗忘方法的‘遗忘后遗症’。 

> **Keywords:** LVLM, Machine Unlearning, Privacy Protection, Behavior Guidance, Generative Models

**Authors:** Minsung Kim, Nakyeong Yang, Kyomin Jung

**Institution(s):** Seoul National University


## Problem Background

大型视觉-语言模型（LVLMs）因训练数据中包含敏感个人图像和隐私信息而存在隐私风险。
现有机器遗忘方法虽能移除特定知识以保护隐私，但忽视了遗忘后模型输出的质量，导致‘遗忘后遗症’（Unlearning Aftermaths），如退化输出、幻觉内容或过度拒绝回答，影响用户体验并可能引发误信息传播。
论文旨在解决这一问题，提出一个新任务，要求模型在遗忘隐私信息的同时，生成基于视觉特征的信息性替代响应。

## Method

*   **核心思想:** 提出 PUBG（Post-Unlearning Behavior Guidance）方法，不仅抑制遗忘目标的隐私信息输出，还通过引导模型输出分布，确保遗忘后生成有信息量且基于视觉特征的响应。
*   **具体实现:** 
    *   **行为引导损失（Behavior Guidance Loss）**：利用 KL 散度衡量模型输出分布与理想参考分布之间的差异，并通过优化缩小差距。参考分布通过原始模型结合上下文提示（in-context prompting）生成，提示模型忽略隐私信息，仅描述视觉特征（如发型、服装）。
    *   **梯度上升损失（Gradient Ascent Loss）**：针对遗忘数据集，增加损失以抑制隐私信息的生成，确保遗忘效果。
    *   **联合优化**：结合上述两种损失，通过小批量优化更新模型参数，在遗忘隐私信息的同时，引导输出行为接近理想分布。
*   **创新点:** 区别于传统遗忘方法仅关注知识移除，PUBG 强调‘替换’输出行为，关注遗忘后输出的质量和用户体验。

## Experiment

*   **有效性:** PUBG 在 LLaVA-1.6-Mistral 和 LLaVA-1.6-Vicuna 模型上测试，与基线方法（GA、NPO、RANDOM、REJECT）一样实现了完美的隐私保护（USR=1.0，JUDGE_privacy=1.0），同时显著提升了输出信息量（例如 CLIPScore 从 0.183-0.215 提升至 0.233，JUDGE_inform 从 1.0-1.6 提升至 3.4）。
*   **优越性:** 基线方法普遍存在‘遗忘后遗症’，如退化输出、幻觉（JUDGE_hall=5.0）或过度拒绝，而 PUBG 有效缓解了这些问题，输出更贴合视觉特征，幻觉程度低。
*   **实验设置合理性:** 实验覆盖不同遗忘实体数量（5、10、20），在已见和未见图像上测试，验证了泛化能力；消融研究表明联合优化两种损失是关键。
*   **开销:** 主要增加训练时的参考分布生成和损失计算成本，但推理时无额外负担。

## Further Thoughts

PUBG 通过引导输出分布控制模型行为的思路具有启发性，未来可扩展至其他生成式模型（如 LLMs）或更复杂遗忘场景；此外，利用上下文提示生成参考分布的方式，启发我们思考如何借助模型指令跟随能力实现可控生成，或通过多模态数据组合设计动态参考分布，应对多样化的隐私保护需求。