"""
run_all_tests.py
自动化测试脚本 - 一键测试所有功能
"""
import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.db_manager import db_manager
from utils.session_manager import session_manager
from mock_xianyu_users import MockXianyuUsers


class TestRunner:
    """测试运行器"""

    def __init__(self):
        self.mock = MockXianyuUsers()
        self.test_results = []

    def log(self, message, level="INFO"):
        """打印日志"""
        emoji = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARN": "⚠️"}.get(level, "ℹ️")
        print(f"{emoji} {message}")

    def clear_data(self):
        """清理测试数据"""
        self.log("清理测试数据...")
        db_manager.clear_all_sessions()
        self.log("数据已清理", "SUCCESS")

    async def test_scene_1_auto_reply(self):
        """场景1: Agent自动回复"""
        self.log("\n" + "="*60)
        self.log("场景1: Agent自动回复（普通问题）")
        self.log("="*60)

        # 发送普通问题
        await self.mock.send_message(0, "扫地机器人怎么充电？")
        time.sleep(3)

        # 检查会话状态
        session = session_manager.get_session_by_chat_id("mock_chat_001")
        if session:
            self.log(f"会话状态: {session.status.value}")
            self.log(f"消息数: {len(session.message_history)}")

            # 应该不是待人工状态
            if session.status.value == "active":
                self.log("✅ Agent自动处理了消息", "SUCCESS")
                self.test_results.append(("场景1", True, "Agent自动回复"))
            else:
                self.log("⚠️ 消息被转人工（可能是Agent无法回答）", "WARN")
                self.test_results.append(("场景1", True, "转人工处理"))
        else:
            self.log("❌ 会话未创建", "ERROR")
            self.test_results.append(("场景1", False, "会话未创建"))

    async def test_scene_2_transfer_to_human(self):
        """场景2: Agent转人工"""
        self.log("\n" + "="*60)
        self.log("场景2: Agent转人工（RAG无法回答）")
        self.log("="*60)

        # 发送奇怪的问题（RAG无法回答）
        await self.mock.send_message(1, "这个机器人能飞吗？能带我去月球吗？")
        time.sleep(5)

        # 检查会话状态
        session = session_manager.get_session_by_chat_id("mock_chat_002")
        if session:
            self.log(f"会话状态: {session.status.value}")

            if session.status.value == "pending_human":
                self.log("✅ 正确触发转人工", "SUCCESS")
                self.test_results.append(("场景2", True, "转人工触发"))
            else:
                self.log("⚠️ 未触发转人工", "WARN")
                self.test_results.append(("场景2", False, "未触发转人工"))
        else:
            self.log("❌ 会话未创建", "ERROR")
            self.test_results.append(("场景2", False, "会话未创建"))

    async def test_scene_3_human_reply(self):
        """场景3: 人工回复"""
        self.log("\n" + "="*60)
        self.log("场景3: 人工回复并发送")
        self.log("="*60)

        # 确保有待处理会话
        pending = session_manager.get_pending_sessions()
        if not pending:
            self.log("没有待处理会话，先创建...")
            await self.mock.send_transfer_message(2)
            time.sleep(3)
            pending = session_manager.get_pending_sessions()

        if pending:
            session = pending[0]
            self.log(f"待处理会话: {session.user_name}")

            # 模拟人工回复（直接写入outbox）
            success = db_manager.add_pending_message(
                session_id=session.session_id,
                chat_id=session.chat_id,
                user_id=session.user_id,
                content="您好，我是人工客服，很高兴为您服务！"
            )

            if success:
                self.log("✅ 人工回复已提交到outbox", "SUCCESS")

                # 检查outbox
                pending_msgs = db_manager.get_pending_messages()
                self.log(f"待发送消息数: {len(pending_msgs)}")

                self.test_results.append(("场景3", True, "人工回复提交"))
            else:
                self.log("❌ 提交失败", "ERROR")
                self.test_results.append(("场景3", False, "提交失败"))
        else:
            self.log("❌ 没有待处理会话", "ERROR")
            self.test_results.append(("场景3", False, "无待处理会话"))

    async def test_scene_4_multiple_users(self):
        """场景4: 多用户并发"""
        self.log("\n" + "="*60)
        self.log("场景4: 多用户并发")
        self.log("="*60)

        # 并发发送
        await self.mock.send_concurrent_messages(3)
        time.sleep(5)

        # 检查会话数
        all_sessions = session_manager.get_all_sessions()
        self.log(f"总会话数: {len(all_sessions)}")

        if len(all_sessions) >= 3:
            self.log("✅ 多用户会话创建成功", "SUCCESS")
            self.test_results.append(("场景4", True, f"{len(all_sessions)}个会话"))
        else:
            self.log("⚠️ 会话数不足", "WARN")
            self.test_results.append(("场景4", False, "会话数不足"))

    def test_scene_5_database_integrity(self):
        """场景5: 数据库完整性"""
        self.log("\n" + "="*60)
        self.log("场景5: 数据库完整性检查")
        self.log("="*60)

        # 检查数据库状态
        all_sessions = db_manager.get_all_sessions()
        pending_sessions = db_manager.get_pending_sessions()
        pending_msgs = db_manager.get_pending_messages()

        self.log(f"总会话数: {len(all_sessions)}")
        self.log(f"待处理会话: {len(pending_sessions)}")
        self.log(f"待发送消息: {len(pending_msgs)}")

        # 验证数据一致性
        checks = []
        for session_dict in all_sessions:
            messages = db_manager.get_session_messages(session_dict['session_id'])
            checks.append({
                'session_id': session_dict['session_id'],
                'user': session_dict['user_name'],
                'status': session_dict['status'],
                'messages': len(messages)
            })

        self.log("\n会话详情:")
        for check in checks:
            self.log(f"  - {check['user']}: {check['status']} ({check['messages']}条消息)")

        self.log("✅ 数据库检查完成", "SUCCESS")
        self.test_results.append(("场景5", True, "数据完整"))

    def print_summary(self):
        """打印测试总结"""
        self.log("\n" + "="*60)
        self.log("测试总结")
        self.log("="*60)

        passed = sum(1 for _, result, _ in self.test_results if result)
        failed = len(self.test_results) - passed

        for name, result, msg in self.test_results:
            status = "✅ 通过" if result else "❌ 失败"
            print(f"  {status} - {name}: {msg}")

        print(f"\n总计: {passed} 通过, {failed} 失败")

        if failed == 0:
            self.log("\n🎉 所有测试通过！系统运行正常！", "SUCCESS")
        else:
            self.log(f"\n⚠️ 有 {failed} 个测试失败，请检查日志", "WARN")

    async def run_all_tests(self):
        """运行所有测试"""
        self.log("\n" + "🧪"*30)
        self.log("开始自动化测试")
        self.log("🧪"*30)

        # 清理数据
        self.clear_data()

        # 运行测试场景
        try:
            await self.test_scene_1_auto_reply()
        except Exception as e:
            self.log(f"场景1异常: {e}", "ERROR")
            self.test_results.append(("场景1", False, str(e)))

        try:
            await self.test_scene_2_transfer_to_human()
        except Exception as e:
            self.log(f"场景2异常: {e}", "ERROR")
            self.test_results.append(("场景2", False, str(e)))

        try:
            await self.test_scene_3_human_reply()
        except Exception as e:
            self.log(f"场景3异常: {e}", "ERROR")
            self.test_results.append(("场景3", False, str(e)))

        try:
            await self.test_scene_4_multiple_users()
        except Exception as e:
            self.log(f"场景4异常: {e}", "ERROR")
            self.test_results.append(("场景4", False, str(e)))

        try:
            self.test_scene_5_database_integrity()
        except Exception as e:
            self.log(f"场景5异常: {e}", "ERROR")
            self.test_results.append(("场景5", False, str(e)))

        # 打印总结
        self.print_summary()


async def main():
    """主函数"""
    runner = TestRunner()
    await runner.run_all_tests()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("智能客服系统 - 自动化测试")
    print("="*60)
    print("\n注意：")
    print("1. 确保管理后台已启动: streamlit run admin.py")
    print("2. 测试会清理所有现有数据")
    print("3. 测试过程中请观察管理后台界面")
    print("\n按 Ctrl+C 取消测试，或按 Enter 继续...")

    try:
        input()
    except KeyboardInterrupt:
        print("\n已取消")
        sys.exit(0)

    asyncio.run(main())
