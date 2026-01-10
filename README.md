# Context Engineering for Multi-Agent Systems 🤖
### 从脚本小子到系统架构师：构建可控、高可用 AI Agent 系统的实战指南

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](./LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)]()
[![Course](https://img.shields.io/badge/Course-配套视频课程-orange?style=for-the-badge)](你的课程链接)

> **"Agent 开发不仅是 Prompt Engineering，更是 Context Engineering。"**

---

## 📖 为什么在这个仓库学习？

如果你正在经历以下阶段：
- 只会调用 OpenAI API，但不知道如何让多个 AI 协同工作。
- 写的 Agent 只能处理简单对话，一旦任务复杂（如读取长文档、多步骤推理）就“幻觉”频发。
- 想从传统开发转行 AI Agent 开发，但缺乏**系统级**的设计思维。

**这个仓库就是为你准备的。**

这里没有黑盒魔法，只有**一行行手写的底层逻辑**。本项目从零开始，不依赖沉重的第三方框架（如 LangChain 的高层封装），带你手动构建一个**轻量级、基于上下文驱动的多智能体系统**。

通过解析 NASA 任务文档（Juno & Perseverance）的实战案例，你将掌握 Agent 开发中最核心的**上下文工程（Context Engineering）**技术。

---

## 🚀 你将学到什么 (Roadmap)

本项目代码对应我的**《AI Agent 系统架构实战》**课程章节，我们将一步步重构代码，见证一个系统的诞生：

| 章节 | 核心技能点 | 代码路径 |
| :--- | :--- | :--- |
| **Phase 1** | **单体智能与基础架构** <br> 摒弃脚本思维，建立 Agent 注册机制与基础类设计。 | `src/chapter2` - `src/chapter3` |
| **Phase 2** | **引擎与调度系统** <br> 手写 `Engine` 类，实现 Agent 之间的消息路由与任务分发。 | `src/chapter4` - `src/chapter5` |
| **Phase 3** | **上下文工程与 RAG** <br> 引入 NASA 真实文档，实现向量检索与动态上下文注入。 | `src/chapter7/text_vector.py` |
| **Phase 4** | **生产级多智能体协作** <br> 让 "Researcher" 与 "Writer" Agent 在复杂上下文中自主协作。 | `src/chapter6` - `src/chapter7` |

![架构流程图](./src/chapter7/流程图.png)
*(系统架构演进图 - 详见 Chapter 7)*

---

## 🎓 深度配套课程（适合想要转行的你）

代码只是骨架，思想才是灵魂。

如果你想：
1. **听懂每一行代码背后的设计权衡**（为什么这里用注册模式？为什么那里要切分 Context？）。
2. **获得 1V1 的职业转型建议**（如何把这个项目写进简历？面试官会问什么？）。
3. **加入高净值 AI 开发者社群**，与同行者一起进化。

👉 **[点击这里订阅《AI Agent 系统架构实战》视频课程](你的课程链接)**
*(早鸟优惠进行中，备注 "GitHub" 领取专属资料包)*

---

## ⚡ 快速开始 (Quick Start)

### 1. 克隆仓库
```bash
git clone [https://github.com/helloggx/Context-Engineering-For-Multi-Agent-Systems.git](https://github.com/helloggx/Context-Engineering-For-Multi-Agent-Systems.git)
cd Context-Engineering-For-Multi-Agent-Systems

```

### 2. 环境配置

推荐使用 Conda 或 venv 管理环境：

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

```

### 3. 配置密钥

在项目根目录创建 `.env` 文件，填入你的 API Key：

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

```

### 4. 运行最终章实战 (NASA RAG Agent)

```bash
# 进入第 7 章目录，体验完全体的 Multi-Agent 系统
python src/chapter7/execute.py

```

*你将看到 Agent 自动读取 `nasa_documents` 中的火星探测器数据，并进行各种复杂的查询与推理。*

---

## 📂 核心文件导读

* **`src/chapter7/engine.py`**: 系统的核心大脑，负责协调所有 Agent。
* **`src/chapter7/registry.py`**: Agent 的注册中心，解耦系统的关键。
* **`src/chapter7/nasa_source.py`**: 真实数据源处理，展示如何处理非结构化数据。
* **`src/chapter7/text_vector.py`**: 向量化处理模块，RAG 的核心实现。

---

## 🤝 贡献与社区

欢迎提交 PR 或 Issue！如果你在使用代码过程中遇到任何问题，或者对 Agent 开发有独特的见解：

* 提交 Issue 讨论
* **[加入我们的 Discord/微信群](https://www.google.com/search?q=%E4%BD%A0%E7%9A%84%E7%A4%BE%E7%BE%A4%E9%93%BE%E6%8E%A5)** (仅限课程学员)

---

## 📜 License

本项目采用 [MIT License](https://www.google.com/search?q=LICENSE) 开源。这意味着你可以自由地将其用于学习、研究甚至商业项目，但请保留原作者版权声明。

---

*Made with ❤️ by [Gavin] | 致力于让 AI 开发更简单、更系统。*

