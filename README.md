# AutoFishMan

基于 LangChain 和 RAG 技术的闲鱼智能客服系统，支持自动回复和人工客服接管。

## 功能特性

- 🤖 **AI 自动回复**：基于 RAG 技术，自动回答用户常见问题
- 👨‍💼 **人工客服接管**：当 AI 无法回答时，自动转人工处理
- 🔔 **飞书通知**：转人工时自动发送飞书通知
- 🖥️ **管理后台**：Streamlit 构建的客服管理界面
- 💾 **数据持久化**：SQLite 数据库存储会话和消息
- 🔄 **多进程支持**：管理后台和消息处理独立运行，通过数据库共享数据
- 🖼️ **图片支持**：支持图片消息的理解和回复
- 🎙️ **语音支持**：支持语音消息的识别和回复

## 系统架构

```
用户消息 → xianyu_client → xianyu_live → Agent处理
                                      ↓
                              需要转人工?
                              ↓         ↓
                             是         否
                              ↓         ↓
                    set_pending_human   自动回复
                              ↓
                         数据库
                              ↓
                    管理后台(独立进程)
                              ↓
                         人工回复
                              ↓
                    pending_outbox表
                              ↓
                    xianyu_live轮询器
                              ↓
                         发送给用户
```

## 项目结构

```
AutoFishMan/
├── agent/                  # Agent 相关代码
│   ├── react_agent.py     # ReAct Agent 实现
│   ├── audio_agent.py     # 语音 Agent
│   ├── vision_agent.py    # 视觉 Agent
│   └── tools/             # Agent 工具
│       ├── agent_tools.py # 工具函数
│       ├── middleware.py  # 中间件（转人工处理）
│       └── multimodal_tools.py
├── config/                # 配置示例（复制 .yml.example 为 .yml 后填写）
├── data/                  # 数据文件
│   ├── *.txt              # 知识库文档
│   └── sessions.db        # SQLite 数据库（自动创建）
├── model/                 # 模型相关
│   ├── factory.py         # 模型工厂
│   └── multimodal_factory.py
├── prompts/               # 提示词模板
├── rag/                   # RAG 相关
│   ├── rag_service.py     # RAG 服务
│   └── vector_store.py    # 向量存储
├── utils/                 # 工具类
│   ├── db_manager.py      # 数据库管理
│   ├── session_manager.py # 会话管理
│   ├── config_handler.py  # 配置处理
│   ├── logger_handler.py  # 日志处理
│   ├── feishu_client.py   # 飞书客户端
│   └── path_tool.py       # 路径工具
├── xianyu/                # 闲鱼相关
│   ├── xianyu_client.py   # 闲鱼客户端（Playwright）
│   └── xianyu_live.py     # 消息调度处理
├── admin.py               # 管理后台（Streamlit）
├── main.py                # 主入口
├── requirements.txt       # 依赖列表
└── README.md              # 项目说明
```

## 快速开始

### 1. 获取代码并安装依赖

使用 **Python 3.10–3.12，推荐 3.12**。目前的音频依赖使用 `audioop`，暂不支持 Python 3.13+。
以下命令均在仓库根目录执行。

```bash
git clone https://github.com/zyfyz666/AutoFishMan.git
cd AutoFishMan
python -m venv .venv
```

激活虚拟环境：Windows PowerShell 使用 `.\.venv\Scripts\Activate.ps1`；
macOS / Linux 使用 `source .venv/bin/activate`。

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check

# 安装 Playwright Chromium 浏览器
python -m playwright install chromium
```

Linux 若提示缺少浏览器系统库，可执行 `python -m playwright install --with-deps chromium`。
客户端会打开浏览器供扫码登录，需要图形桌面环境。

`sqlite3` 是 Python 标准库，无需另外安装。处理 MP3、M4A 等非 WAV/PCM 音频时，
还需安装 [FFmpeg](https://ffmpeg.org/download.html) 并将其加入 PATH，可用 `ffmpeg -version` 检查。

### 2. 复制配置示例并设置 API Key

仓库提供四份不含密钥的 `config/*.yml.example`。复制后再编辑 `.yml` 文件：

**Windows PowerShell：**

```powershell
Copy-Item config/agent.yml.example config/agent.yml
Copy-Item config/rag.yml.example config/rag.yml
Copy-Item config/chroma.yml.example config/chroma.yml
Copy-Item config/prompts.yml.example config/prompts.yml
$env:DASHSCOPE_API_KEY = "你的百炼 API Key"
```

**macOS / Linux：**

```bash
cp config/agent.yml.example config/agent.yml
cp config/rag.yml.example config/rag.yml
cp config/chroma.yml.example config/chroma.yml
cp config/prompts.yml.example config/prompts.yml
export DASHSCOPE_API_KEY="你的百炼 API Key"
```

上述环境变量仅对当前终端及其启动的进程生效；分别启动服务时，每个终端都需要设置。
模型实际使用阿里云百炼 DashScope，需为账号开通所选模型；程序不会自动读取 `.env`。

| 文件 | 需要确认的内容 |
| --- | --- |
| `config/agent.yml` | 将 `my_user_id` 填为自己的闲鱼数字 ID，用于过滤自己发送的消息 |
| `config/rag.yml` | `chat_model_name`、`embedding_model_name` 以及视觉、语音模型名称 |
| `config/chroma.yml` | 知识库路径、向量库目录和分段参数；可先使用示例默认值 |
| `config/prompts.yml` | 提示词文件路径；可直接使用示例默认值 |

飞书通知通过**应用机器人**发送。使用通知功能前，在 `agent.yml` 中填写
`feishu_app_id`、`feishu_app_secret` 和 `feishu_human_agent_open_id`，并为应用开通发送消息权限。
当前实现不读取 `feishu.yml` 或 webhook，也不需要 `xianyu.yml`；闲鱼登录态由扫码后生成的 `auth.json` 保存。

`external_data_path` 是报表工具使用的可选 CSV 路径，普通知识库问答无需该文件。
使用报表工具前需自行提供 CSV，首行为表头，后续列依次为用户 ID、特征、效率、耗材、对比、月份。

真实 `.yml` 配置、CSV 和登录状态均被 Git 忽略；只提交 `.yml.example` 示例。

### 3. 准备知识库数据

将知识库文档放入 `data/` 目录，支持 `.txt` 和 `.pdf` 格式。

### 4. 构建向量数据库

```bash
python -m rag.vector_store
```

这会读取 `data/` 目录下的文档，生成向量并存储到 ChromaDB。

### 5. 启动系统

```bash
python main.py
```

首次运行需要：
1. 扫码登录闲鱼
2. 登录状态会保存到 `auth.json`

系统会同时启动：
- 闲鱼消息监听（自动回复 + 人工回复轮询）
- 客服管理后台 http://localhost:8501

## 使用方式

### 方式一：完整系统（生产环境）

```bash
python main.py
```

### 方式二：分别启动（测试/开发）

**终端 1 - 启动管理后台：**
```bash
streamlit run admin.py --server.port 8501
```

**终端 2 - 启动闲鱼监听：**
```bash
python -c "import asyncio; from main import start_xianyu_listener; asyncio.run(start_xianyu_listener()())"
```

## 工作流程

### 1. 自动回复流程

1. 用户发送消息到闲鱼
2. `xianyu_client` 接收消息（WebSocket）
3. `xianyu_live` 调度处理（防抖 + 消息合并）
4. Agent 检索 RAG 知识库
5. 生成回复并通过闲鱼 API 发送给用户

### 2. 转人工流程

1. Agent 判断无法回答用户问题
2. 调用 `transfer_to_human` 工具
3. Middleware 监控到转人工操作
4. 标记会话为 `pending_human` 状态
5. 发送飞书通知（需配置应用机器人凭证和人工客服 open_id）
6. 管理后台显示待处理会话
7. 人工客服在管理后台回复消息
8. 回复存入 `pending_outbox` 表
9. `xianyu_live` 轮询器发送消息给闲鱼用户

## 管理后台操作

### 查看待处理会话

1. 打开 http://localhost:8501
2. 在"待处理会话"区域查看需要人工处理的会话
3. 点击会话查看详情

### 回复用户

1. 在会话详情页查看消息历史
2. 在"回复用户"输入框中输入回复内容
3. 点击"发送回复"按钮
4. 系统会自动将消息发送给闲鱼用户

### 标记已解决

1. 查看会话详情
2. 点击"标记已解决"按钮
3. 会话状态变为 `resolved`

## 功能清单

- [x] 基础消息收发
- [x] RAG 知识库问答
- [x] 转人工机制
- [x] 管理后台
- [x] 多进程数据共享
- [x] 飞书通知
- [x] 支持图片消息
- [x] 支持语音消息

## 开发计划

- [ ] 会话统计分析
- [ ] 多客服分配

## 常见问题

### Q: 管理后台看不到会话？

A: 检查以下几点：
1. 数据库是否初始化：`data/sessions.db` 是否存在
2. 两个进程是否使用同一个数据库文件
3. 检查日志是否有错误

### Q: 回复没有发送给用户？

A: 检查以下几点：
1. 检查 `pending_outbox` 表是否有待发送消息
2. 检查 xianyu_live 轮询器是否运行
3. 检查闲鱼客户端是否登录成功

### Q: Agent 没有转人工？

A: 检查以下几点：
1. 检查 RAG 检索结果
2. 检查 Agent 的 prompt 是否包含转人工指令
3. 检查 middleware 是否正常工作

### Q: RAG 无法检索到内容？

A: 检查以下几点：
1. 是否已运行 `python -m rag.vector_store` 构建向量库
2. `data/` 目录下是否有知识库文件
3. 检查 `chroma_db/` 目录是否存在

### Q: 如何清空测试数据？

```bash
python -c "from utils.db_manager import db_manager; db_manager.clear_all_sessions()"
```

## 日志查看

```bash
# 查看转人工相关日志
tail -f logs/agent.log | grep "转人工"

# 查看数据库操作
tail -f logs/agent.log | grep "DatabaseManager"

# 查看消息发送
tail -f logs/agent.log | grep "发送"
```

## 技术栈

- **Python 3.10–3.12**（推荐 3.12）
- **LangChain** - Agent 框架
- **LangGraph** - 工作流编排
- **Playwright** - 浏览器自动化（闲鱼 WebSocket）
- **Streamlit** - 管理后台
- **SQLite** - 数据存储
- **ChromaDB** - 向量数据库（RAG）
- **阿里云百炼 DashScope** - 通义千问文本、视觉、嵌入及语音模型

## 注意事项

1. **隐私保护**：`config/` 仅提交 `.yml.example` 示例；真实 `.yml`、CSV 和密钥请勿上传到 GitHub
2. **登录状态**：首次运行需要扫码登录，登录状态会保存到 `auth.json`
3. **知识库**：需要定期更新 `data/` 目录下的知识库文件，并重新构建向量库
4. **多进程**：管理后台和消息处理是独立进程，通过 SQLite 共享数据

## 贡献指南

欢迎提交 Issue 和 PR！安装依赖后可运行离线回归测试：

```bash
python -m unittest discover -s tests -v
```

测试在临时目录使用示例配置和占位 API Key，不登录闲鱼、不调用模型、不发送消息。
覆盖配置与依赖导入、PDF 读取、Agent 初始化以及启动资源释放。


## 许可证

MIT License

## 联系方式

如有问题，请提交 Issue 或联系开发者。
