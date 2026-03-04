# 🐟 AutoFishMan

> 基于 Playwright + LangChain + RAG 的闲鱼多模态智能客服机器人
>
> *A multimodal intelligent customer service bot for Xianyu, built with Playwright, LangChain & RAG*

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green)
![License](https://img.shields.io/badge/License-GPL--3.0-orange)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)

---

## 📖 项目简介 | Overview

**AutoFishMan** 是一个部署在闲鱼平台的多模态智能客服系统，能够自动处理买家的文字、图片、语音消息，结合 RAG 知识库与 ReAct Agent 实现专业的售前售后应答，并在无法处理时自动通过飞书通知人工客服介入。

**AutoFishMan** is a multimodal intelligent customer service system deployed on Xianyu (China's second-hand marketplace). It automatically handles text, image, and voice messages from buyers, leveraging RAG knowledge base and ReAct Agent for professional responses, with automatic escalation to human agents via Feishu notification.

---

## ✨ 核心功能 | Features

- 🔤 **多轮对话** — 基于 LangChain ReAct Agent，支持上下文感知的多轮问答
- 🖼️ **图片理解** — 集成 Qwen-VL 视觉大模型，识别买家发送的商品图片并作答
- 🎙️ **语音识别** — 接入 SenseVoice ASR，支持中英文混合语音转文字
- 🔊 **语音合成** — 集成 CosyVoice TTS，自动生成语音回复
- 📚 **RAG 知识库** — 基于 ChromaDB + DashScope Embedding，支持本地知识库检索增强
- 📊 **使用报告** — 自动生成用户机器人使用情况月度报告
- 🌤️ **天气感知** — 结合用户地理位置与实时天气，提供场景化使用建议
- 👨‍💼 **自动转人工** — 知识库无法覆盖时，自动飞书通知人工客服，实现无缝交接
- ⏱️ **消息聚合** — 5 秒防抖缓冲，将用户连续多条消息合并后统一处理

---

## 🏗️ 系统架构 | Architecture

```
闲鱼消息 (Playwright)
        ↓
  消息防抖聚合 (5s debounce)
        ↓
  ┌─────────────────────────┐
  │     ReAct Agent         │
  │  (LangChain + Qwen3)    │
  └────────┬────────────────┘
           │ Tools
    ┌──────┴──────────────────────────┐
    │  RAG检索  │  天气  │  用户定位  │
    │  报告生成 │  转人工│  外部数据  │
    └──────┬──────────────────────────┘
           ↓
  多模态处理 (Vision / ASR / TTS)
           ↓
     飞书通知 / 自动回复
```

---

## 🛠️ 技术栈 | Tech Stack

| 模块 | 技术 |
|------|------|
| 平台自动化 | Playwright (async) |
| Agent 框架 | LangChain ReAct Agent |
| 大语言模型 | Qwen3-Max (阿里云 DashScope) |
| 视觉模型 | Qwen-VL-Max |
| 语音识别 | SenseVoice-v1 / Paraformer-realtime-v2 |
| 语音合成 | CosyVoice-v1 |
| Embedding | text-embedding-v4 |
| 向量数据库 | ChromaDB |
| 消息通知 | 飞书开放平台 API |
| 配置管理 | YAML |

---

## 🚀 快速开始 | Getting Started

### 环境要求 | Prerequisites

- Python 3.10+
- 阿里云 DashScope API Key
- 飞书开放平台应用凭证

### 安装依赖 | Installation

```bash
git clone https://github.com/zyfyz666/AutoFishMan.git
cd AutoFishMan
pip install -r requirements.txt
playwright install chromium
```

### 配置 | Configuration

复制配置模板并填入你的密钥：

```bash
cp config/agent.yml.example config/agent.yml
```

编辑 `config/agent.yml`：

```yaml
feishu_app_id: "your_feishu_app_id"
feishu_app_secret: "your_feishu_app_secret"
feishu_human_agent_open_id: "your_open_id"
my_user_id: "your_xianyu_user_id"
external_data_path: data/external/records.csv
```

在环境变量中设置 DashScope API Key：

```bash
export DASHSCOPE_API_KEY="your_api_key"
```

### 运行 | Run

```bash
python main.py
```

---

## 📁 项目结构 | Project Structure

```
AutoFishMan/
├── agent/                  # Agent 核心模块
│   ├── react_agent.py      # ReAct Agent 主逻辑
│   ├── vision_agent.py     # 视觉理解 Agent
│   └── audio_agent.py      # 语音处理 Agent
├── model/                  # 模型工厂
│   ├── factory.py          # LLM & Embedding 工厂
│   └── multimodal_factory.py  # 多模态模型工厂 (VLM/ASR/TTS)
├── rag/                    # RAG 知识库模块
├── xianyu/                 # 闲鱼平台交互
│   ├── xianyu_live.py      # 消息监听与防抖聚合
│   └── xianyu_client.py    # 平台 API 封装
├── prompts/                # Prompt 模板
├── config/                 # 配置文件
│   ├── agent.yml.example   # 配置模板
│   └── rag.yml             # 模型配置
├── data/                   # 知识库文档
└── utils/                  # 工具函数
```

---

## 🔑 环境变量 | Environment Variables

| 变量名 | 说明 |
|--------|------|
| `DASHSCOPE_API_KEY` | 阿里云 DashScope API 密钥 |



---

## 📄 开源协议 | License

本项目基于 [GPL-3.0](LICENSE) 协议开源，禁止闭源商用。

---

## 🙋 作者 | Author

**zyfyz666** — 在校学生，专注大模型应用与 AI Agent 方向

> 如有问题欢迎提 Issue 或联系作者。
