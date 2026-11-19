import os

import pytest


@pytest.fixture(autouse=True)
def figlio_supervisionato(monkeypatch):
    """I test chiamano cmd_run/cmd_retrieve come farebbe il figlio del supervisore (padre = questo processo di test).
    I test del rifiuto fuori dal supervisore tolgono la variabile."""
    monkeypatch.setenv("F775_SUPERVISOR_PID", str(os.getppid()))
