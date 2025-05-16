---
title: "Reproducibility Study of "Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents""
pubDatetime: 2025-05-14T11:15:14+00:00
slug: "2025-05-llm-cooperation-simulation"
type: "arxiv"
id: "2505.09289"
score: 0.6703839444459075
author: "grok-3-latest"
authors: ["Pedro M. P. Curvo", "Mara Dragomir", "Salvador Torpes", "Mohammadmahdi Rahimi"]
tags: ["LLM", "Cooperation", "Multi-Agent Systems", "Resource Sharing", "Behavioral Influence"]
institution: ["University of Amsterdam"]
description: "本文通过复现和扩展 `GovSim` 框架，验证了大型语言模型在资源共享场景中的合作能力，并揭示了模型规模、任务框架和智能体交互对合作行为的深远影响。"
---

> **Summary:** 本文通过复现和扩展 `GovSim` 框架，验证了大型语言模型在资源共享场景中的合作能力，并揭示了模型规模、任务框架和智能体交互对合作行为的深远影响。 

> **Keywords:** LLM, Cooperation, Multi-Agent Systems, Resource Sharing, Behavioral Influence

**Authors:** Pedro M. P. Curvo, Mara Dragomir, Salvador Torpes, Mohammadmahdi Rahimi

**Institution(s):** University of Amsterdam


## Problem Background

本文旨在复现并扩展 Piatti 等人（2024）提出的 `GovSim` 模拟框架，评估大型语言模型（LLMs）在资源共享场景中的合作决策能力，解决的核心问题是 LLMs 是否能在‘公地悲剧’等社会困境中实现可持续合作。
原研究表明大型模型（如 GPT-4-turbo）能实现合作，而小型模型往往失败；此外，研究还探索了普遍化原则（universalization principle）对合作行为的促进作用，以及框架在新模型、语言和场景下的适用性。

## Method

* **复现实验**：基于 `GovSim` 平台，针对 Fishery 场景（渔业资源共享），使用原研究模型（如 GPT-4-turbo、Llama-3-8B）进行测试，分为默认设置（default）和普遍化设置（universalization），后者通过提示引导模型考虑行动的广泛影响。
* **扩展实验**：
  1. **新模型测试**：引入 DeepSeek-V3、GPT-4o-mini 和 Qwen 系列模型，评估不同规模和架构模型的合作能力，保持 Fishery 场景和标准参数设置。
  2. **语言影响实验**：将 Fishery 场景指令翻译为日语，测试语言对模型行为的影响，假设日语文化中的集体主义可能促进合作，实验对象包括 DeepSeek-V3 和 GPT-4o 等支持日语的模型。
  3. **异构多智能体环境（MultiGov）**：修改代码支持不同模型组合（如 4 个 DeepSeek-V3 与 1 个 GPT-4o-mini），观察高性能模型是否能通过沟通影响低性能模型的行为，实验在默认 Fishery 场景下进行。
  4. **逆向环境（Inverse Environment）**：设计‘垃圾’场景，模型需合作消除有害资源（public bad），评估损失厌恶（loss aversion）对合作行为的影响，调整评估指标（如将 Total Gain 改为 Total Loss）。
* **评估方式**：通过生存率、生存时间、总收益/损失、效率、平等性和资源过度使用等指标量化模型性能，实验参数（如智能体数量、资源增长率）与原研究一致。

## Experiment

* **复现效果**：成功验证原研究结论，大型模型（如 GPT-4-turbo、GPT-4o）在默认场景下生存时间达 12 个月，实现可持续合作；小型模型（如 Llama-2-7B）通常在 1-2 个月内失败；普遍化原则显著提升部分小型模型（如 GPT-3.5）生存时间，从 1-2 个月增至 12 个月，结果与原论文一致。
* **扩展效果**：
  1. **新模型**：DeepSeek-V3 表现与 GPT-4-turbo 相当，生存时间 12 个月；GPT-4o-mini 在默认场景失败，但在普遍化场景提升至 12 个月；Qwen 系列表现较差。
  2. **日语指令**：语言影响不显著，DeepSeek-V3 和 GPT-4o 表现接近英文场景，GPT-4o-mini 仍失败。
  3. **逆向环境**：大多数模型（包括 GPT-4o-mini）在‘垃圾’场景实现 12 个月生存，但行为更不稳定，显示损失厌恶效应。
  4. **异构环境**：高性能模型（如 DeepSeek-V3）能引导低性能模型（如 GPT-4o-mini）减少资源消耗，部分组合实现可持续合作。
* **实验设置评价**：实验设计全面，覆盖模型规模、语言、场景和智能体组合，但受计算资源限制（总运行时间约 70 小时），部分大型模型未全面测试，运行次数较少（复现 3 次，扩展 5 次），可能影响统计显著性。

## Further Thoughts

异构多智能体系统中高性能模型对低性能模型的正向影响启发了我，未来可通过少量大型模型引导小型模型以降低计算成本，适用于资源受限场景；此外，任务框架（正向/负向）对 LLMs 行为的影响表明，调整任务表述可能优化合作效果；最后，文化嵌入式叙事（如特定地域背景）或将成为探索 LLMs 社会行为的重要方向。