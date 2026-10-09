"""Smoke-test a fresh install with example config and no external services."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]

SMOKE_SCRIPT = textwrap.dedent("""\
    import asyncio
    import inspect
    from pathlib import Path
    import sys

    # The isolated interpreter may import only the temporary application copy.
    sys.path.insert(0, str(Path.cwd()))
    # Lark obtains an event loop at import time. On Windows its wakeup pipe uses
    # a local socket pair, so create that before blocking all network operations.
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)
    network_attempts = []

    def block_network(event, args):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
            network_attempts.append(event)
            raise RuntimeError("Network access is forbidden in installation tests")

    sys.addaudithook(block_network)

    import streamlit
    from PIL import Image
    from pydub import AudioSegment
    from chromadb.config import Settings
    from agent.react_agent import ReactAgent
    from agent.tools import multimodal_tools
    from utils.feishu_client import feishu_client
    from utils.file_handler import pdf_loader
    from utils.prompt_loader import (
        load_system_prompts, load_rag_prompts, load_report_prompts,
    )
    from xianyu.xianyu_client import XianyuClient
    from xianyu.xianyu_live import XianyuLive
    import main

    assert Settings().anonymized_telemetry is False
    assert "height" in inspect.signature(streamlit.container).parameters
    assert Image.new("RGB", (1, 1)).size == (1, 1)
    assert len(AudioSegment.silent(duration=10)) == 10
    for load_prompt in (load_system_prompts, load_rag_prompts, load_report_prompts):
        assert load_prompt().strip(), "An example prompt path is empty or invalid"

    # Construction validates the real LangChain APIs and example configuration;
    # invoking the agent or loading embeddings would require a paid API call.
    assert ReactAgent().agent is not None
    documents = pdf_loader(str(next(Path("data").glob("*.pdf"))))
    assert any(document.page_content.strip() for document in documents)
    assert not network_attempts, f"Unexpected network attempts: {network_attempts}"
    assert not Path("auth.json").exists()
    event_loop.close()
    print("Offline installation smoke test passed")
""")


class InstallationTests(unittest.TestCase):
    def test_example_config_and_real_dependencies_work_offline(self):
        with tempfile.TemporaryDirectory(prefix="autofishman-install-") as directory:
            sandbox = Path(directory)
            # Copy source only: never copy a user's config, auth, databases or logs.
            for package in ("agent", "model", "rag", "utils", "xianyu"):
                for source in (ROOT / package).rglob("*.py"):
                    destination = sandbox / source.relative_to(ROOT)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
            shutil.copy2(ROOT / "main.py", sandbox / "main.py")
            (sandbox / "config").mkdir()
            for example in (ROOT / "config").glob("*.yml.example"):
                shutil.copy2(example, sandbox / "config" / example.stem)
            (sandbox / "prompts").mkdir()
            for prompt in (ROOT / "prompts").glob("*.txt"):
                shutil.copy2(prompt, sandbox / "prompts" / prompt.name)
            (sandbox / "data").mkdir()
            pdf = next((ROOT / "data").glob("*.pdf"))
            shutil.copy2(pdf, sandbox / "data" / pdf.name)

            # Do not inherit API credentials, tracing settings, or proxy settings.
            environment = {
                key: value for key, value in os.environ.items()
                if key.upper() in {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT"}
            }
            environment.update({
                "DASHSCOPE_API_KEY": "offline-installation-test-key",
                "ANONYMIZED_TELEMETRY": "False",
                "LANGCHAIN_TRACING_V2": "false",
                "LANGSMITH_TRACING": "false",
                "OTEL_SDK_DISABLED": "true",
                "USERPROFILE": directory,
                "TEMP": directory,
                "TMP": directory,
                "TMPDIR": directory,
            })
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-X", "utf8", "-c", SMOKE_SCRIPT],
                cwd=sandbox,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=90,
            )
            self.assertEqual(
                result.returncode, 0,
                f"Installation smoke test failed:\n{result.stdout}\n{result.stderr}",
            )


if __name__ == "__main__":
    unittest.main()
