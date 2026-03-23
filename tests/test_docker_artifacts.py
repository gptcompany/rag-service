from pathlib import Path

from scripts.setup._config import COMPOSE_EXTERNAL_TEMPLATE, DOCKERFILE_TEMPLATE


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_committed_dockerfile_matches_template():
    assert (REPO_ROOT / "Dockerfile").read_text() == DOCKERFILE_TEMPLATE


def test_committed_compose_matches_external_template():
    assert (REPO_ROOT / "docker-compose.yml").read_text() == COMPOSE_EXTERNAL_TEMPLATE
