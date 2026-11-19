"""Sensori termici (F-775): soglie lette dai sensori, fail-closed su sensore assente, illeggibile o fuori dominio."""
from __future__ import annotations

import pytest

from scripts import trial_supervisor as ts


def test_soglie_lette_dai_sensori_e_fuori_dominio_bloccano(tmp_path, monkeypatch):
    hw = tmp_path / "hwmon3"
    hw.mkdir()
    (hw / "name").write_text("coretemp\n")
    (hw / "temp1_max").write_text("100000\n")
    monkeypatch.setattr(ts, "HWMON", tmp_path)
    assert ts.cpu_limit() == (hw / "temp1_input", 100.0)
    for bad in ("0\n", "-5\n", "n/a\n"):
        (hw / "temp1_max").write_text(bad)
        with pytest.raises(ts.SensorError):
            ts.cpu_limit()
    (hw / "temp1_max").unlink()
    with pytest.raises(ts.SensorError):
        ts.cpu_limit()


@pytest.mark.parametrize("out, ok", [("  GPU Slowdown Temp : 92 C\n", 92.0), ("GPU Slowdown Temp : N/A\n", None), ("", None),
                                      ("GPU Slowdown Temp : 0 C\n", None), ("GPU Slowdown Temp : 92junk\n", None),
                                      ("GPU Slowdown Temp : 92 F\n", None)])
def test_limite_gpu_da_nvidia_smi_o_il_sensore_blocca(monkeypatch, out, ok):
    monkeypatch.setattr(ts.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": out, "returncode": 0})())
    if ok is None:
        with pytest.raises(ts.SensorError):
            ts.gpu_slowdown_limit()
    else:
        assert ts.gpu_slowdown_limit() == ok


def test_senza_nvidia_smi_il_sensore_blocca(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(ts.subprocess, "run", missing)
    with pytest.raises(ts.SensorError):
        ts.gpu_slowdown_limit()


def test_lettura_corrente_con_picchi_e_fuori_dominio_che_blocca(tmp_path, monkeypatch):
    (tmp_path / "temp1_input").write_text("61500\n")
    sensors = object.__new__(ts.Sensors)
    sensors.cpu_input, sensors.peaks = tmp_path / "temp1_input", {"vram_mib": 0, "gpu_c": 0.0, "cpu_c": 0.0}
    smi = {"out": "812, 70\n", "rc": 0}
    monkeypatch.setattr(ts.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": smi["out"], "returncode": smi["rc"]})())
    assert sensors.read() == {"gpu": 70.0, "cpu": 61.5, "vram_mib": 812} and sensors.peaks["gpu_c"] == 70.0
    for bad in ("N/A, N/A\n", "812, -3\n", ""):
        smi["out"] = bad
        with pytest.raises(ts.SensorError):
            sensors.read()
    smi["out"], smi["rc"] = "812, 70\n", 9  # nvidia-smi fallisce pur avendo scritto qualcosa di parsabile
    with pytest.raises(ts.SensorError):
        sensors.read()
    smi["rc"] = 0
    (tmp_path / "temp1_input").unlink()  # il sensore sparisce
    smi["out"] = "812, 70\n"
    with pytest.raises(ts.SensorError):
        sensors.read()


def test_nvidia_smi_che_esce_con_errore_blocca_anche_la_soglia(monkeypatch):
    monkeypatch.setattr(ts.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": "GPU Slowdown Temp : 92 C\n", "returncode": 9})())
    with pytest.raises(ts.SensorError, match="exit 9"):
        ts.gpu_slowdown_limit()


@pytest.mark.parametrize("out", ["812.9, 70\n", "812, 7e-1\n", "812, 70, 3\n"])
def test_telemetria_gpu_non_intera_blocca(monkeypatch, tmp_path, out):
    (tmp_path / "temp1_input").write_text("50000\n")
    sensors = object.__new__(ts.Sensors)
    sensors.cpu_input, sensors.peaks = tmp_path / "temp1_input", {"vram_mib": 0, "gpu_c": 0.0, "cpu_c": 0.0}
    monkeypatch.setattr(ts.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": out, "returncode": 0})())
    with pytest.raises(ts.SensorError):
        sensors.read()


def test_output_di_nvidia_smi_non_decodificabile_blocca_soglia_e_lettura(monkeypatch):
    def undecodable(*a, **k):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    monkeypatch.setattr(ts.subprocess, "run", undecodable)
    with pytest.raises(ts.SensorError):
        ts.gpu_slowdown_limit()
    sensors = object.__new__(ts.Sensors)
    with pytest.raises(ts.SensorError):
        sensors.read()
