---
title: "The Influence of Human-inspired Agentic Sophistication in LLM-driven Strategic Reasoners"
pubDatetime: 2025-05-14T13:51:24+00:00
slug: "2025-05-agentic-sophistication-llm"
type: "arxiv"
id: "2505.09396"
score: 0.45702185460154926
author: "grok-3-latest"
authors: ["Vince Trencsenyi", "Agnieszka Mensfelt", "Kostas Stathis"]
tags: ["LLM", "Strategic Reasoning", "Agentic Design", "Game Theory", "Human Likeness"]
institution: ["Royal Holloway University of London"]
description: "本文通过多代理模拟框架，系统分析了代理复杂性对 LLM 驱动战略推理中人类相似性的非线性影响，为设计目标导向的代理系统提供了重要指导。"
---

> **Summary:** 本文通过多代理模拟框架，系统分析了代理复杂性对 LLM 驱动战略推理中人类相似性的非线性影响，为设计目标导向的代理系统提供了重要指导。 

> **Keywords:** LLM, Strategic Reasoning, Agentic Design, Game Theory, Human Likeness

**Authors:** Vince Trencsenyi, Agnieszka Mensfelt, Kostas Stathis

**Institution(s):** Royal Holloway University of London


## Problem Background

随着大型语言模型（LLMs）向代理型 AI（Agentic AI）的转变，研究重点转向如何在博弈论场景中模拟人类战略推理（Strategic Reasoning）。
核心问题是：LLM 驱动的代理是否能有效复制人类在战略决策中的行为模式，尤其是在总体模式和个体差异上的表现？此外，‘代理复杂性’（Agentic Sophistication）如何影响这种模拟效果？
论文试图解决现有研究缺乏对代理复杂性与人类行为相似性之间关系的系统性分析，以及 LLM 在博弈论任务中的泛化能力和可解释性问题。

## Method

*   **核心框架：** 设计了一个多代理模拟框架，基于传统代理概念（如 Observe-Orient-Decide-Act 循环），用于在两玩家一次性猜测游戏中测试 LLM 驱动代理的战略推理能力。
*   **代理模型：** 提出了三种不同复杂度的代理设计：
    *   **基准模型（EWA）：** 采用自调经验加权吸引模型（Self-tuning Experience Weighted Attraction），一个基于经济学的博弈论学习算法，作为对比基准，不依赖自然语言处理。
    *   **简单代理（Simple Agent, S）：** 基于较松散的代理定义，直接将 LLM 作为代理，通过一次性推理过程生成决策，内部可能包含简单的信念推断。
    *   **推理代理（Reasoner Agent, R）：** 基于更强的代理定义，将 LLM 嵌入传统代理框架，采用分步推理（Reasoning）和决策（Decision）机制，模拟更复杂的人类认知过程。
*   **配置变量：** 引入不同层次的上下文（Context, C）和指令（Instruction, M）配置：
    *   上下文包括无背景（No Profile）、简单角色（Simple Profile）、详细背景（Biography），以测试角色信息对推理的影响。
    *   指令包括是否采用基于人类决策心理的‘适当性模型’（Model of Appropriateness, MoA），通过三个问题（情境认知、自我认知、行为选择）引导 LLM 推理，增强人类相似性。
*   **LLM 选择：** 测试了两种 LLM——Claude Haiku 3.5（轻量模型）和 Sonnet 3.7（旗舰模型），以对比模型能力对结果的影响。
*   **实验场景：** 使用两玩家一次性猜测游戏作为测试平台，通过与人类数据（学生和专家群体）对比，评估代理在总体和子群体层面的表现。

## Experiment

*   **有效性：** 人类启发的认知结构（如 Reasoner 代理）能显著提升 LLM 代理与人类战略行为的对齐程度，尤其是在 Haiku 模型上，Reasoner 代理在复杂配置下与人类推理水平（k-level）的误差仅为 0.02，表现出色。
*   **复杂性与表现：** 代理复杂性与人类相似性之间呈非线性关系。Haiku 在高复杂性配置（如 Reasoner + Biography + MoA）下表现更优，而 Sonnet 在简单配置下更接近人类行为，表明模型能力与架构复杂性的匹配对结果影响较大。
*   **泛化能力：** 在调整猜测范围（100-200）的泛化测试中，Haiku 比 Sonnet 表现更好，简单配置的代理也更容易适应新参数，表明更复杂的模型可能因训练数据过拟合而泛化能力较差。
*   **子群体模拟：** Reasoner 代理在模拟学生与专家行为差异时表现更平衡，而 Simple 代理倾向于过拟合专家行为，反映出复杂架构在捕捉个体差异上的优势。
*   **成本与权衡：** 复杂配置（如 Reasoner 代理和 MoA）显著增加计算成本（以 token 计），例如 Reasoner 代理的输入输出 token 数远高于 Simple 代理，需权衡性能与资源消耗。
*   **合理性与局限：** 实验覆盖了 25 种代理配置、超过 2000 个推理样本，并从多个维度（k-level 误差、分布相似性、零猜测率）对比人类数据，设计较为全面合理。但 Sonnet 在复杂配置下的不一致表现可能与指令遵循能力或过拟合有关，提示结果受底层 LLM 特性的影响较大。

## Further Thoughts

论文揭示了代理复杂性与人类相似性之间的非线性关系，这一发现启发我们在设计 LLM 代理时，应根据任务目标动态调整模型选择和架构复杂性，而非一味追求最复杂或最强大的配置。此外，人类启发认知结构（如 MoA）对 LLM 推理的增强效果，提示我们可以在其他领域（如社会模拟、决策支持）探索更多人类认知模型与 LLM 的结合方式，以提升代理在复杂环境中的表现。