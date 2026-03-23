from pathlib import Path

from scripts.setup._config import COMPOSE_EXTERNAL_TEMPLATE, DOCKERFILE_TEMPLATE


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_committed_dockerfile_matches_template():
    assert (REPO_ROOT / "Dockerfile").read_text() == DOCKERFILE_TEMPLATE


def test_committed_compose_matches_external_template():
    assert (REPO_ROOT / "docker-compose.yml").read_text() == COMPOSE_EXTERNAL_TEMPLATE


def test_docker_constraints_pin_known_problematic_runtime_packages():
    text = (REPO_ROOT / "docker-constraints.txt").read_text()
    assert "configparser==7.2.0" in text
    assert "mineru==2.7.1" in text
    assert "lightrag-hku==1.4.9.10" in text
