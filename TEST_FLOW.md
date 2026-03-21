# 智能客服系统测试流程

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

## 测试步骤

### 第一步：启动系统

```bash
# 方式1：同时启动闲鱼监听和管理后台
python main.py

# 方式2：分别启动（用于测试）
# 终端1：启动管理后台
streamlit run admin.py --server.port 8501

# 终端2：启动模拟脚本（代替真实闲鱼）
python mock_xianyu_users.py
```

### 第二步：测试场景

#### 场景1：Agent自动回复（无需转人工）

1. 在模拟脚本中选择 **1. 发送单条消息**
2. 选择用户 0（张三）
3. 输入问题："扫地机器人怎么充电？"
4. 观察：
   - Agent应该自动回复
   - 管理后台不会出现待处理会话

#### 场景2：Agent转人工（RAG无法回答）

1. 在模拟脚本中选择 **1. 发送单条消息**
2. 选择用户 1（李四）
3. 输入问题："这个机器人能飞吗？"（奇怪的问题，RAG无法回答）
4. 观察：
   - Agent判断无法回答，调用 `transfer_to_human`
   - 会话显示在管理后台的"待处理会话"
   - 飞书收到通知（如果配置了）

#### 场景3：人工回复并发送给用户

1. 在管理后台看到待处理会话
2. 点击会话查看详情
3. 在"回复用户"框中输入回复内容
4. 点击"发送回复"
5. 观察：
   - 管理后台显示"回复已提交"
   - 模拟脚本的选项8（查看收到的回复）能看到回复
   - 数据库中消息状态变为"sent"

#### 场景4：转人工后用户继续发送消息

1. 先触发转人工（场景2）
2. 在管理后台未回复前，用户继续发送消息
3. 观察：
   - 新消息应该被缓存，不交给Agent处理
   - 管理后台能看到所有消息历史

#### 场景5：并发多用户测试

1. 在模拟脚本中选择 **4. 并发发送多条消息**
2. 输入数量：5
3. 观察：
   - 系统能同时处理多个会话
   - 需要转人工的会话都显示在管理后台

### 第三步：验证数据

#### 检查数据库状态

```bash
python check_admin_interface.py
```

应该看到：
- 所有会话列表
- 待处理会话
- 消息历史
- 待发送消息（outbox）

#### 检查多进程共享

1. 在模拟脚本中发送消息触发转人工
2. 在管理后台刷新，应该立即看到新会话
3. 在管理后台回复
4. 在模拟脚本中查看收到的回复

## 预期结果

### ✅ 成功标志

1. **Agent自动回复**：普通问题Agent能自动回答
2. **转人工触发**：RAG无法回答时自动转人工
3. **管理后台显示**：待处理会话实时显示
4. **人工回复发送**：回复能发送给用户
5. **多进程共享**：两个进程数据一致
6. **消息缓存**：转人工后新消息被缓存

### ❌ 常见问题

1. **管理后台看不到会话**
   - 检查数据库是否初始化
   - 检查两个进程是否使用同一个数据库文件

2. **回复没有发送给用户**
   - 检查 outbox 表是否有待发送消息
   - 检查 xianyu_live 轮询器是否运行

3. **Agent没有转人工**
   - 检查 RAG 检索结果
   - 检查 Agent 的 prompt 是否包含转人工指令

## 配置文件

### agent.yml 关键配置

```yaml
# 转人工相关配置
my_user_id: '你的闲鱼数字ID'

# RAG配置
rag:
  top_k: 5
  score_threshold: 0.7
```

### feishu.yml 配置（可选）

```yaml
# 飞书通知
webhook_url: '你的飞书机器人webhook'
```

## 日志查看

### 关键日志位置

1. **xianyu_client**: 接收和发送闲鱼消息
2. **xianyu_live**: 消息调度和Agent执行
3. **middleware**: 转人工工具调用
4. **session_manager**: 会话状态变更
5. **db_manager**: 数据库操作

### 日志过滤

```bash
# 查看转人工相关日志
tail -f logs/agent.log | grep "转人工"

# 查看数据库操作
tail -f logs/agent.log | grep "DatabaseManager"
```

## 清理测试数据

```bash
# 清空所有会话和消息
python -c "from utils.db_manager import db_manager; db_manager.clear_all_sessions()"
```

## 下一步（生产环境）

测试完成后，将 `mock_xianyu_users.py` 替换为真实的 `xianyu_client`：

```python
# main.py 中
from xianyu.xianyu_client import XianyuClient

client = XianyuClient()
# 不再需要 mock
```

系统会自动通过 WebSocket 连接真实闲鱼，接收和发送真实消息。
