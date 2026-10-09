"""Offline regression tests for the supported startup APIs."""
import asyncio
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, create_autospec, patch


ROOT = Path(__file__).resolve().parents[1]


class ClientAPI:
    """Public interface used by the browser-backed XianyuClient."""

    on_message = None

    def __init__(self):
        pass

    async def run(self):
        pass

    async def close(self):
        pass


class LiveAPI:
    def __init__(self, xianyu_client):
        pass

    async def on_message(self, msg):
        pass

    async def start_outbox_poller(self, interval=2.0):
        pass


class StartupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        logger_module = types.ModuleType("utils.logger_handler")
        logger_module.logger = Mock()
        # main must not import the nonexistent xianyu_conf or read private config.
        config_module = types.ModuleType("utils.config_handler")
        spec = importlib.util.spec_from_file_location("startup_under_test", ROOT / "main.py")
        self.main = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {
            "utils.logger_handler": logger_module,
            "utils.config_handler": config_module,
        }), patch.object(sys, "path", sys.path.copy()):
            spec.loader.exec_module(self.main)

        self.client_factory = create_autospec(ClientAPI, spec_set=True)
        self.client = self.client_factory.return_value
        self.live_factory = create_autospec(LiveAPI, spec_set=True)
        self.live = self.live_factory.return_value
        client_module = types.ModuleType("xianyu.xianyu_client")
        client_module.XianyuClient = self.client_factory
        live_module = types.ModuleType("xianyu.xianyu_live")
        live_module.XianyuLive = self.live_factory
        self.listener_modules = {
            "xianyu.xianyu_client": client_module,
            "xianyu.xianyu_live": live_module,
        }
        self.poller_started = asyncio.Event()
        self.poller_stopped = asyncio.Event()

        async def poller(interval=2.0):
            self.poller_started.set()
            try:
                await asyncio.Future()
            finally:
                self.poller_stopped.set()

        self.live.start_outbox_poller.side_effect = poller

    async def test_listener_registers_callback_and_stops_when_client_returns(self):
        async def run():
            self.assertIs(self.client.on_message, self.live.on_message)
            await self.poller_started.wait()

        self.client.run.side_effect = run
        with patch.dict(sys.modules, self.listener_modules):
            await asyncio.wait_for(self.main.start_xianyu_listener()(), timeout=1)

        self.client_factory.assert_called_once_with()
        self.live_factory.assert_called_once_with(self.client)
        self.client.run.assert_awaited_once_with()
        self.live.start_outbox_poller.assert_awaited_once_with(interval=2.0)
        self.assertTrue(self.poller_stopped.is_set())
        self.client.close.assert_awaited_once_with()

    async def test_listener_releases_resources_on_startup_failure(self):
        async def run():
            await self.poller_started.wait()
            raise RuntimeError("browser startup failed")

        self.client.run.side_effect = run
        with patch.dict(sys.modules, self.listener_modules):
            with self.assertRaisesRegex(RuntimeError, "browser startup failed"):
                await asyncio.wait_for(self.main.start_xianyu_listener()(), timeout=1)

        self.assertTrue(self.poller_stopped.is_set())
        self.client.close.assert_awaited_once_with()

    async def test_listener_releases_resources_when_cancelled(self):
        async def run():
            await asyncio.Future()

        self.client.run.side_effect = run
        with patch.dict(sys.modules, self.listener_modules):
            task = asyncio.create_task(self.main.start_xianyu_listener()())
            await asyncio.wait_for(self.poller_started.wait(), timeout=1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        self.assertTrue(self.poller_stopped.is_set())
        self.client.close.assert_awaited_once_with()

    def test_admin_starts_streamlit_with_current_python(self):
        with patch.object(self.main.subprocess, "Popen") as popen:
            proc = self.main.start_streamlit_admin()

        self.assertIs(proc, popen.return_value)
        self.assertIn(proc, self.main.processes)
        command = popen.call_args.args[0]
        self.assertEqual(command[:4], [sys.executable, "-m", "streamlit", "run"])
        self.assertEqual(Path(command[4]), ROOT / "admin.py")
        self.assertEqual(command[5:], [
            "--server.port", "8501", "--server.headless", "true",
        ])


if __name__ == "__main__":
    unittest.main()
