"""Compose versionato: tetti di risorse, riavvio e bind locale (F-775, decisione limiti-risorse-calcolo-agenti)."""
from __future__ import annotations

from pathlib import Path

import yaml

COMPOSE = yaml.safe_load((Path(__file__).resolve().parent.parent / "docker-compose.yml").read_text())
SERVICES = COMPOSE["services"]


def test_ogni_servizio_ha_mem_limit_e_nessuno_swap():
    for name, svc in SERVICES.items():
        assert svc.get("mem_limit"), name
        assert svc.get("memswap_limit") == svc["mem_limit"], name


def test_limite_di_isolamento_provvisorio_4g():
    assert SERVICES["rag"]["mem_limit"] == "4g"


def test_ogni_servizio_riparte_se_non_fermato_a_mano():
    for name, svc in SERVICES.items():
        assert svc.get("restart") == "unless-stopped", name


def test_rete_host_con_bind_esplicito_su_loopback():
    """Decisione dell'operatore: network_mode host, niente ports/extra_hosts, bind e URL su 127.0.0.1."""
    for name, svc in SERVICES.items():
        assert svc.get("network_mode") == "host", name
        assert svc.get("pull_policy") == "build", name
        assert "ports" not in svc and "extra_hosts" not in svc, name
    env = dict(e.split("=", 1) for e in SERVICES["rag"]["environment"])
    assert env["RAG_HOST"] == "127.0.0.1"
    assert "127.0.0.1:11434" in env["OLLAMA_HOST"] and "host.docker.internal" not in env["OLLAMA_HOST"]


def test_modello_ollama_versionato_nel_compose():
    env = dict(e.split("=", 1) for e in SERVICES["rag"]["environment"])
    assert env["RAG_OLLAMA_MODEL"] == "qwen3:8b"


def test_i_template_del_wizard_portano_le_stesse_salvaguardie_del_compose():
    from scripts.setup._config import COMPOSE_EXTERNAL_TEMPLATE, COMPOSE_SIDECAR_TEMPLATE

    external = yaml.safe_load(COMPOSE_EXTERNAL_TEMPLATE)["services"]["rag"]
    for key in ("mem_limit", "memswap_limit", "restart", "network_mode"):
        assert external[key] == SERVICES["rag"][key], key
    assert "ports" not in external
    # Con l'ollama sidecar il rag resta in rete bridge (la rete host userebbe la 11434 dell'host), ma sempre su loopback.
    sidecar = yaml.safe_load(COMPOSE_SIDECAR_TEMPLATE)["services"]["rag"]
    for key in ("mem_limit", "memswap_limit", "restart"):
        assert sidecar[key] == SERVICES["rag"][key], key
    assert all(p.startswith("127.0.0.1:") for p in sidecar["ports"])
