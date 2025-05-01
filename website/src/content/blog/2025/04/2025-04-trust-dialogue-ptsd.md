---
title: "TRUST: An LLM-Based Dialogue System for Trauma Understanding and Structured Assessments"
pubDatetime: 2025-05-01T15:50:43Z
slug: "2025-04-trust-dialogue-ptsd"
type: "arxiv"
id: "2504.21851"
score: 0.8697280710230166
author: "grok-3-latest"
authors: ["Sichang Tu", "Abigail Powers", "Stephen Doogan", "Jinho D. Choi"]
tags: ["LLM", "Dialogue System", "Mental Health", "Clinical Interview", "Patient Simulation"]
institution: ["Emory University", "DooGood Foundation"]
description: "本文提出 TRUST 系统，利用大型语言模型模拟临床医生进行 PTSD 正式诊断访谈和评估，通过对话行为模式和患者模拟方法填补心理健康对话系统空白，为服务可及性提供新思路。"
---

> **Summary:** 本文提出 TRUST 系统，利用大型语言模型模拟临床医生进行 PTSD 正式诊断访谈和评估，通过对话行为模式和患者模拟方法填补心理健康对话系统空白，为服务可及性提供新思路。 

> **Keywords:** LLM, Dialogue System, Mental Health, Clinical Interview, Patient Simulation
> **Recommendation Score:** 0.8697280710230166

**Authors:** Sichang Tu, Abigail Powers, Stephen Doogan, Jinho D. Choi
**Institution(s):** Emory University, DooGood Foundation

## Problem Background

在美国，超过2800万患有精神疾病的成年人未接受治疗，主要由于心理健康服务提供者短缺和高昂的医疗成本；特别是在创伤后应激障碍（PTSD）诊断中，结构化临床访谈需要大量专业知识和时间，造成服务瓶颈；尽管大型语言模型（LLMs）在对话系统中显示出潜力，但目前缺乏专注于正式诊断访谈和评估的系统。

## Method

* **系统架构**：TRUST 是一个基于 LLM 的对话系统，旨在模拟临床医生进行 PTSD 的正式诊断访谈和评估，包含数据库和框架两大模块；数据库存储变量元数据（如 CAPS 标准中的诊断变量）、访谈历史和评估结果；框架模块包括对话管理和评估两个子模块，负责生成临床医生响应和诊断评估。
* **对话行为（DA）模式**：提出了一种专门为临床访谈设计的 DA 模式，包含 8 个标签（如问候、指导、共情、信息寻求等），用于分解复杂决策过程，提高系统对对话流程的控制；DA 标签通过 Claude 模型预测，结合访谈历史生成合适的临床响应。
* **访谈流程管理**：系统基于 DSM-5 的 Clinician-Administered PTSD Scale (CAPS) 标准，采用分层问题树结构，区分核心问题和可选问题；通过动态决策点判断是否需要进一步询问或评估，确保访谈的结构化和自然性。
* **患者模拟**：为解决临床测试的高成本和时间问题，提出了一种基于真实访谈记录的患者模拟方法，利用 LLM 生成符合患者沟通模式和情感状态的响应，避免手动测试的局限性，同时尽量减少幻觉风险。
* **技术实现**：系统使用 Claude (claude-3-5-sonnet-20241022) 模型进行 DA 标签预测和响应生成，支持在 Amazon Web Services (AWS) 上部署，理论上可替换为其他 LLM。

## Experiment

* **评估设置**：通过人类专家（对话专家和 PTSD 临床心理学家）和 LLM 自动评估两种方式对 TRUST 系统进行测试；评估分为代理生成（系统模拟临床医生）和患者模拟两部分，采用 5 点 Likert 量表，涵盖全面性、适当性和沟通风格等指标；选取 5 个 CAPS 访谈记录进行症状集群级别的细粒度评估。
* **结果分析**：人类专家评估显示代理生成性能与真实临床访谈相当（平均分 -0.14 到 0.03），但在沟通风格上存在分歧，临床专家认为系统有时偏离诊断重点；患者模拟评估为‘可接受’（平均分 0.31 和 0.32），但忠实度较低（-0.39 和 -0.33），存在幻觉问题；LLM 评估高估了系统性能且结果不一致，显示出自动评估的局限性。
* **实验合理性**：实验设置结合多角度指标和症状集群评估较为全面，样本量虽有限，但足以初步验证系统潜力；结果表明系统在模拟临床医生行为方面有显著潜力，但在沟通细微差别和模拟忠实度上需进一步改进。

## Further Thoughts

论文提出的对话行为（DA）模式为临床访谈提供了一种结构化的中间层，是否可以进一步抽象为通用的医疗对话框架，适用于更多心理健康条件？此外，患者模拟中的幻觉问题是否可以通过引入领域特定规则或知识图谱来缓解？LLM 在临床领域的评估局限性提示我们，未来是否可以通过结合情感计算或专门微调来提升其在医疗对话中的适用性？