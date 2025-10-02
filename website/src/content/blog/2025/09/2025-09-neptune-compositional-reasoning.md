---
title: "NePTune: A Neuro-Pythonic Framework for Tunable Compositional Reasoning on Vision-Language"
pubDatetime: 2025-09-30T04:22:42+00:00
slug: "2025-09-neptune-compositional-reasoning"
type: "arxiv"
id: "2509.25757"
score: 0.43326698156548943
author: "grok-3-latest"
authors: ["Danial Kamali", "Parisa Kordjamshidi"]
tags: ["VLM", "Neuro-Symbolic", "Compositional Reasoning", "Visual Grounding", "Soft Logic"]
institution: ["Michigan State University"]
description: "NePTune 提出了一种神经符号框架，通过混合执行模型结合软组合推理和命令式逻辑，显著提升了视觉-语言组合推理的性能和泛化能力。"
---

> **Summary:** NePTune 提出了一种神经符号框架，通过混合执行模型结合软组合推理和命令式逻辑，显著提升了视觉-语言组合推理的性能和泛化能力。 

> **Keywords:** VLM, Neuro-Symbolic, Compositional Reasoning, Visual Grounding, Soft Logic

**Authors:** Danial Kamali, Parisa Kordjamshidi

**Institution(s):** Michigan State University


## Problem Background

现代视觉-语言模型（VLMs）在组合推理（Compositional Reasoning）方面存在显著不足，难以分解和重组概念来解决新颖问题。
现有神经符号（Neuro-Symbolic）方法虽然有潜力，但往往受限于严格的逻辑执行（对感知错误敏感）或预定义谓词（缺乏灵活性），限制了其在视觉-语言任务中的泛化能力和适应性。

## Method

*   **核心思想:** 提出 NePTune 框架，一个神经符号视觉推理框架，通过混合执行模型整合基础视觉模型的感知能力和符号推理的组合表达能力，实现鲁棒的组合推理。
*   **组件1 - LLM-based Program Generator:** 利用大型语言模型（LLM）将自然语言查询转化为可执行的 Python 程序，包含命令式控制流（如循环、条件语句），并提取相关对象名称用于后续处理。选择 Python 是因为其图灵完备性和 LLM 生成代码的熟练度。
*   **组件2 - Perceptual Grounding:** 将符号程序与视觉内容连接，包含两个子模块：
    *   **Object Proposal Generation:** 使用 Grounding DINO 模型，基于 LLM 提取的对象名称，在图像中生成候选对象的边界框。
    *   **Concept Grounding Interface:** 通过视觉-语言模型（VLM）提供两个接口：`score`（返回概念概率分数，支持单对象、多对象和全局图像查询）和 `query`（返回开放式答案），以 grounding 图像中的原子概念（如属性、类别、关系）。
*   **组件3 - Symbolic Executor:** 采用混合执行模型，结合两种推理模式：
    *   **Soft Compositional Reasoning:** 基于模糊逻辑（fuzzy logic）原则，使用软逻辑操作（如 AND、OR）直接处理 VLM 提供的不确定性分数（uncertainty scores），避免二元决策的脆弱性。
    *   **Imperative Reasoning:** 利用标准 Python 解释器处理程序的整体结构和控制流，支持复杂的程序逻辑（如条件、循环、变量赋值），充分发挥通用编程语言的表达能力。
*   **关键特性:** 框架以零样本（training-free）方式运行，解耦感知与推理，同时通过可微分操作支持微调以适应新领域。

## Experiment

*   **有效性:** NePTune 在多个视觉推理基准上表现出色：在 CLEVR（合成数据）上，零样本准确率达 92.65%，显著优于其他零样本方法（如 ViperGPT 的 36.05%），并接近训练模型（如 NeSyCoCo 的 99.68%）；与基础 VLM（InternVL2.5）相比，准确率从 90.25% 提升至 92.65%，尤其在计数（Count）任务上提升 12.5%。在 CLEVR-Humans（复杂人类查询）上，准确率达 87.67%，优于基础 VLM（85.95%）。在 RefCOCO-Adversarial（真实世界图像）上，准确率达 78.08%（带验证），优于基础 VLM（76.13%）。
*   **领域适应性:** 在 Ref-GTA（游戏模拟图像，领域迁移）上，NePTune 准确率达 69.69%，远超基础 VLM（6.95%），显示出强大的泛化能力；通过微调，性能进一步提升至 69.90%。
*   **实验设置合理性:** 实验覆盖了合成数据（CLEVR）、人类生成问题（CLEVR-Humans）、真实世界图像（RefCOCO-Adversarial）和领域迁移（Ref-GTA）等多种场景，任务包括视觉问答（VQA）和指代表达式 grounding（REG），设置全面且合理。
*   **局限性:** 论文指出对类比概念（如 'same color'）的处理仍受限于 VLM 的感知能力，部分任务（如 Qwen2VL 作为 backbone 时）性能有所下降，反映出对底层 VLM 质量的依赖。

## Further Thoughts

NePTune 的混合推理模型（软逻辑与命令式逻辑结合）为多模态任务提供了新思路，未来可探索其在视频理解等领域的应用；此外，针对 VLM 在类比和复杂关系推理上的局限，是否可以通过优化视觉提示策略或引入空间推理模块（如深度估计）来增强概念 grounding 能力？另外，如何通过少量标注数据高效进行神经符号微调，可能是一个值得深入研究的方向。