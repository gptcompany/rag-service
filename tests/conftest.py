import json
import os
import subprocess

import pytest


@pytest.fixture(autouse=True)
def figlio_supervisionato(monkeypatch):
    """I test chiamano cmd_run/cmd_retrieve come farebbe il figlio del supervisore (padre = questo processo di test).
    I test del rifiuto fuori dal supervisore tolgono la variabile."""
    monkeypatch.setenv("F775_SUPERVISOR_PID", str(os.getppid()))


FAKE_DOCKER = '''#!/usr/bin/env python3
import json, os, signal, sys, time

state, args = os.environ["F775_FAKE_DOCKER"], sys.argv[1:]
open(state + "/calls.jsonl", "a").write(json.dumps({"t": time.monotonic(), "argv": args}) + "\\n")
fail = os.environ.get("F775_FAKE_DOCKER_FAIL", "").split(",")
cmd = args[3] if args[:2] == ["compose", "-f"] and len(args) > 3 else None


def pid():
    try:
        p = int(open(state + "/rag.pid").read())
        return p if open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[0] != "Z" else None
    except (OSError, ValueError):
        return None


stopped_before = any(json.loads(line)["argv"][3:4] == ["stop"] for line in open(state + "/calls.jsonl").read().splitlines()[:-1])
if cmd in fail or (cmd == "ps" and stopped_before and "ps_after_stop" in fail):
    sys.exit("fake docker: " + cmd + " fallito")
if cmd == "stop":
    if pid():
        os.kill(pid(), signal.SIGTERM)
        while pid():
            time.sleep(0.01)
elif cmd == "ps":
    print("abc123" if pid() else "", end="")
elif cmd != "start":
    sys.exit(2)
'''


class FakeDockerControl:
    def __init__(self, state):
        self.state = state

    def calls(self) -> list[dict]:
        path = self.state / "calls.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def attach(self, pid: int) -> None:
        """Il processo che il `docker compose stop rag` finto deve fermare."""
        (self.state / "rag.pid").write_text(str(pid))

    def detach(self) -> None:
        """Nessun container del rag-service in esecuzione."""
        (self.state / "rag.pid").unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def fake_docker(tmp_path_factory, monkeypatch):
    """Nessun test tocca il docker vero (e il rag-service dell'host): `docker` e' un finto su PATH che registra le chiamate."""
    state = tmp_path_factory.mktemp("fake_docker")
    shim = state / "bin" / "docker"
    shim.parent.mkdir()
    shim.write_text(FAKE_DOCKER)
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{shim.parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("F775_FAKE_DOCKER", str(state))
    monkeypatch.delenv("F775_FAKE_DOCKER_FAIL", raising=False)
    control = FakeDockerControl(state)
    live = subprocess.Popen(["sleep", "300"])  # il rag-service finto di default: in esecuzione finche' il `stop` finto non lo ferma
    control.attach(live.pid)
    yield control
    live.terminate()
    live.wait()
