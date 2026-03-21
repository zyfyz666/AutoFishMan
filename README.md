# AutoFishMan

基于 LangChain 和 RAG 技术的闲鱼智能客服系统，支持自动回复和人工客服接管。

## 功能特性

- 🤖 **AI 自动回复**：基于 RAG 技术，自动回答用户常见问题
- 👨‍💼 **人工客服接管**：当 AI 无法回答时，自动转人工处理
- 🔔 **飞书通知**：转人工时自动发送飞书通知
- 🖥️ **管理后台**：Streamlit 构建的客服管理界面
- 💾 **数据持久化**：SQLite 数据库存储会话和消息
- 🔄 **多进程支持**：管理后台和消息处理独立运行，通过数据库共享数据

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
AutoFishMan-Clean/
├── agent/                  # Agent 相关代码
│   ├── react_agent.py     # ReAct Agent 实现
│   ├── tools/             # Agent 工具
│   │   ├── agent_tools.py # 工具函数
│   │   └── middleware.py  # 中间件（转人工处理）
│   └── prompts/           # 提示词模板
├── config/                # 配置文件
│   ├── agent.yml          # Agent 配置
│   ├── xianyu.yml         # 闲鱼配置
│   └── feishu.yml         # 飞书配置
├── data/                  # 数据文件
│   └── sessions.db        # SQLite 数据库
├── rag/                   # RAG 相关
│   └── rag_service.py     # RAG 服务
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
├── mock_xianyu_users.py   # 模拟用户（测试用）
├── run_all_tests.py       # 自动化测试
├── TEST_FLOW.md           # 测试流程文档
├── requirements.txt       # 依赖列表
└── README.md              # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt

# 安装 Playwright 浏览器
playwright install chromium
```

## 配置说明

### 1. Agent 配置 (config/agent.yml)

```yaml
# 你的闲鱼数字ID
my_user_id: '123456789'

# RAG 配置
rag:
  top_k: 5
  score_threshold: 0.7
  
# 外部数据路径
external_data_path: config/external_data.csv
```

### 2. 闲鱼配置 (config/xianyu.yml)

```yaml
cookies:
  # 你的闲鱼 cookies
  
max_workers: 4
```

### 3. 飞书配置 (config/feishu.yml) - 可选

```yaml
webhook_url: 'https://open.feishu.cn/open-apis/bot/v2/hook/xxx'
```

## 使用方式

### 方式一：完整系统（生产环境）

```bash
python main.py
```

这会同时启动：
- 闲鱼消息监听（自动回复 + 人工回复轮询）
- 客服管理后台 http://localhost:8501

首次运行需要：
1. 扫码登录闲鱼
2. 登录状态会保存到 auth.json

### 方式二：分别启动（测试/开发）

**终端 1 - 启动管理后台：**
```bash
streamlit run admin.py --server.port 8501
```

**终端 2 - 启动模拟测试（无需真实闲鱼）：**
```bash
python mock_xianyu_users.py
```

**终端 3 - 启动真实闲鱼监听：**
```bash
# 修改 main.py 中的启动逻辑，只启动 xianyu
```

### 方式三：自动化测试

```bash
python run_all_tests.py
```

一键测试所有功能场景。

## 工作流程

### 1. 自动回复流程

1. 用户发送消息
2. xianyu_client 接收消息
3. xianyu_live 调度处理
4. Agent 检索 RAG 知识库
5. 生成回复并发送给用户

### 2. 转人工流程

1. Agent 判断无法回答用户问题
2. 调用 `transfer_to_human` 工具
3. middleware 监控到转人工操作
4. 标记会话为 `pending_human`
5. 发送飞书通知（如果配置了）
6. 管理后台显示待处理会话
7. 人工客服回复消息
8. 回复存入 `pending_outbox`
9. xianyu_live 轮询器发送消息给用户

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

## 测试

### 手动测试

使用模拟脚本测试各项功能：

```bash
python mock_xianyu_users.py
```

选项说明：
- 1. 发送单条消息 - 测试普通对话
- 2. 发送触发转人工的消息 - 测试转人工流程
- 3. 连续发送多条消息 - 测试消息防抖
- 4. 并发发送多条消息 - 测试并发处理
- 5. 同一用户连续发送 - 测试消息合并
- 6. 查看已发送消息 - 查看发送历史
- 7. 查看当前会话详情 - 查看会话状态
- 8. 查看收到的回复消息 - 查看人工回复
- 9. 退出

### 自动化测试

```bash
python run_all_tests.py
```

测试场景：
1. Agent 自动回复（普通问题）
2. Agent 转人工（RAG 无法回答）
3. 人工回复并发送
4. 多用户并发
5. 数据库完整性检查

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

- **Python 3.10+**
- **LangChain** - Agent 框架
- **LangGraph** - 工作流编排
- **Playwright** - 浏览器自动化（闲鱼 WebSocket）
- **Streamlit** - 管理后台
- **SQLite** - 数据存储
- **ChromaDB** - 向量数据库（RAG）
- **OpenAI API** - LLM 模型

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

## 贡献指南

欢迎提交 Issue 和 PR！

## 许可证

MIT License

## 联系方式

如有问题，请提交 Issue 或联系开发者。
