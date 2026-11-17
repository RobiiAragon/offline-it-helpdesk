from __future__ import annotations

import json
import socketserver
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from conftest import KB, FakeLLM

from offline_helpdesk import download, llm
from offline_helpdesk.evaluate import Question, facts_found, run
from offline_helpdesk.rag import Assistant


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence
        pass

    def do_GET(self):
        self._send({"object": "list", "data": [{"id": "qwen-test"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["temperature"] == 0.0 and body["messages"][0]["role"] == "system"
        self._send(
            {
                "model": "qwen-test",
                "usage": {"completion_tokens": 7},
                "choices": [{"message": {"content": " Reset at the portal [1]. "}}],
            }
        )

    def _send(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _LocalServer(HTTPServer):
    def server_bind(self) -> None:
        # Skip HTTPServer's getfqdn() reverse-DNS lookup, which can take ~30 s on macOS.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


@pytest.fixture()
def fake_server():
    srv = _LocalServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def test_openai_compatible_client(fake_server) -> None:
    client = llm.make("llamacpp", url=fake_server)
    assert client.health() == "qwen-test"
    gen = client.generate("sys", "user", 50)
    assert gen.text == "Reset at the portal [1]." and gen.completion_tokens == 7


def test_unreachable_llm_raises_clear_error() -> None:
    client = llm.make("ollama", url="http://127.0.0.1:9/v1")
    with pytest.raises(llm.LLMUnavailable, match="not reachable"):
        client.generate("s", "u", 5)


def test_download_verifies_checksum(tmp_path) -> None:
    src = tmp_path / "blob.bin"
    src.write_bytes(b"hello")
    good = download.Asset(src.as_uri(), "ok.bin", download.sha256_file(src), 0.0, "MIT")
    assert download.fetch(good, tmp_path / "cache").read_bytes() == b"hello"
    bad = download.Asset(src.as_uri(), "bad.bin", "0" * 64, 0.0, "MIT")
    with pytest.raises(RuntimeError, match="checksum"):
        download.fetch(bad, tmp_path / "cache")
    assert not (tmp_path / "cache" / "bad.bin").exists()


def test_api_endpoints(kb_index) -> None:
    from fastapi.testclient import TestClient

    from offline_helpdesk.api import create_app

    client = TestClient(create_app(Assistant(kb_index, FakeLLM(), min_score=0.0), KB))
    assert "Offline IT Helpdesk" in client.get("/").text
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["chunks"] == len(kb_index.chunks)
    ans = client.post("/ask", json={"question": "my account is locked"}).json()
    assert ans["citations"][0]["n"] == 1 and not ans["refused"]
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert "vpn.md" in client.get("/kb").json()
    assert "GlobalProtect" in client.get("/kb/vpn.md").text
    assert client.get("/kb/..%2Fpyproject.toml").status_code == 404
    assert client.get("/kb/missing.md").status_code == 404


def test_facts_found_supports_alternatives() -> None:
    assert facts_found(
        "Run ipconfig /flushdns then wait 15 minutes", ["flushdns|flushcache", "15 minutes"]
    ) == (2, [])
    assert facts_found("nothing", ["a|b"]) == (0, ["a|b"])


def test_evaluation_metrics(kb_index) -> None:
    qs = [
        Question(
            "q1",
            "How long does an account lockout last?",
            "password-reset.md",
            ["30 minutes"],
            True,
        ),
        Question("o1", "capital of france", None, [], False),
    ]
    assistant = Assistant(kb_index, llm.ExtractiveLLM(), min_score=0.1)
    s = run(assistant, qs)["summary"]
    assert s["n_answerable"] == 1 and s["n_out_of_scope"] == 1
    assert s["retrieval_hit@k"] == 1.0 and s["answer_fact_recall"] == 1.0
    assert s["oos_refusal_rate"] == 1.0 and s["false_refusal_rate"] == 0.0


def test_cli_index_ask_eval(tmp_path, capsys) -> None:
    from offline_helpdesk.cli import main

    common = ["--kb-dir", str(KB), "--index-dir", str(tmp_path / "idx"), "--embedder", "hash"]
    assert main(["index", *common]) == 0
    assert (
        main(
            [
                "ask",
                *common,
                "--llm",
                "extractive",
                "--min-score",
                "0.1",
                "printer",
                "badge",
                "release",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "printers.md" in out
    qfile = tmp_path / "q.jsonl"
    qfile.write_text(
        json.dumps(
            {
                "id": "a",
                "question": "print job deleted after",
                "expected_doc": "printers.md",
                "must_include": ["24 hours"],
                "answerable": True,
            }
        )
        + "\n"
    )
    assert (
        main(
            [
                "eval",
                *common,
                "--llm",
                "extractive",
                "--min-score",
                "0.1",
                "--questions",
                str(qfile),
                "--out",
                str(tmp_path / "r.json"),
            ]
        )
        == 0
    )
    saved = json.loads((tmp_path / "r.json").read_text())
    assert saved["meta"]["embedder"] == "hash" and saved["summary"]["n_answerable"] == 1
