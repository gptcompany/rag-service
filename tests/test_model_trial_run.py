"""Prova modelli (F-775): punteggio Beta-Binomiale, P(migliore), sensori, arresto, scelta."""
from __future__ import annotations

import hashlib
import importlib
import json

import pytest

mt = importlib.import_module("scripts.model_trial")


def test_sensori_senza_soglie_dichiarate_non_partono(monkeypatch):
    monkeypatch.setattr(mt, "gpu_slowdown_limit", lambda: None)
    monkeypatch.setattr(mt, "cpu_limits", lambda: None)
    with pytest.raises(SystemExit):
        mt.Sensors()


def test_guard_ok_segue_exit_code():
    assert mt.guard_ok(["true"]) and not mt.guard_ok(["false"])


def test_limite_gpu_letto_da_nvidia_smi(monkeypatch):
    class R:
        stdout = "    GPU Slowdown Temp                 : 92 C\n"
    monkeypatch.setattr(mt.subprocess, "run", lambda *a, **k: R())
    assert mt.gpu_slowdown_limit() == 92.0


def test_cpu_limite_da_coretemp_temp1_max(tmp_path, monkeypatch):
    hw = tmp_path / "hwmon3"
    hw.mkdir()
    (hw / "name").write_text("coretemp\n")
    (hw / "temp1_max").write_text("100000\n")
    (hw / "temp1_input").write_text("45000\n")
    monkeypatch.setattr(mt, "HWMON", tmp_path)
    path, limit = mt.cpu_limits()
    assert path == hw / "temp1_input" and limit == 100.0


def _args(tmp_path, **kw):
    import argparse

    return argparse.Namespace(scratch=str(tmp_path), models=["a", "b"], guard_cmd="true", ollama_host="http://127.0.0.1:11434", **kw)


class FakeSensors:
    def __init__(self):
        self.hot = __import__("threading").Event()
        self.peak_vram_mib = self.peak_gpu_c = self.peak_cpu_c = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


def _setup_run(tmp_path, monkeypatch, ask):
    monkeypatch.setattr(mt, "SCRATCH_ROOT", tmp_path)
    qs = [{"id": f"q{i}", "paper": f"p{i}", "answer": "x", "question": "?"} for i in range(3)]
    (tmp_path / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in qs))
    (tmp_path / "retrieval.json").write_text(json.dumps({"questions_sha1": hashlib.sha1((tmp_path / "questions.jsonl").read_bytes()).hexdigest()}))
    (tmp_path / "contexts.jsonl").write_text("".join(json.dumps({"id": q["id"], "context": "c"}) + "\n" for q in qs))
    monkeypatch.setattr(mt, "Sensors", FakeSensors)
    monkeypatch.setattr(mt, "ollama_ask", ask)
    mt.cmd_run(_args(tmp_path))
    return json.loads((tmp_path / "result.json").read_text())


def test_round_interrotto_dalla_soglia_termica_non_entra_nei_punteggi(tmp_path, monkeypatch):
    calls = []

    def ask(host, model, ctx, question, hot):
        calls.append(model)
        if len(calls) > 2:
            raise mt.ThermalStop(model)
        return {}, 1.0, {"size": 10, "size_vram": 10}

    res = _setup_run(tmp_path, monkeypatch, ask)
    assert res["rounds_completed"] == 1 and res["stop_reason"].startswith("soglia termica")
    assert all(n == 1 for _, n in map(tuple, res["scores"].values()))


def test_ollama_ask_chiude_lo_stream_a_soglia_termica(monkeypatch):
    import threading

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def raise_for_status(self):
            pass

        def iter_lines(self):
            yield b'{"message": {"content": "{"}}'
            yield b'{"message": {"content": "}"}}'

    import requests

    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())
    hot = threading.Event()
    hot.set()
    with pytest.raises(mt.ThermalStop):
        mt.ollama_ask("h", "m", "c", "q", hot)


def test_sensore_illeggibile_alza_hot(monkeypatch):
    sensors = object.__new__(mt.Sensors)
    sensors.hot, sensors.stop = __import__("threading").Event(), __import__("threading").Event()

    def boom():
        raise ValueError("N/A")

    sensors._sample = boom
    sensors._loop()
    assert sensors.hot.is_set()


def test_lettura_sensore_in_timeout_ferma_la_prova(monkeypatch):
    def hang(*a, **k):
        raise mt.subprocess.TimeoutExpired("nvidia-smi", 10)

    monkeypatch.setattr(mt.subprocess, "run", hang)
    sensors = object.__new__(mt.Sensors)
    sensors.hot, sensors.stop = __import__("threading").Event(), __import__("threading").Event()
    sensors._loop()
    assert sensors.hot.is_set()


def test_run_rifiuta_contesti_mancanti(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "SCRATCH_ROOT", tmp_path)
    (tmp_path / "questions.jsonl").write_text(json.dumps({"id": "q1", "paper": "p", "answer": "x", "question": "?"}) + "\n")
    (tmp_path / "contexts.jsonl").write_text("")
    with pytest.raises(SystemExit):
        mt.cmd_run(_args(tmp_path))


def test_p_migliore_somma_a_uno_e_segue_i_dati():
    p = mt.p_best({"a": (95, 100), "b": (40, 100), "c": (60, 100)})
    assert sum(p.values()) == pytest.approx(1.0)
    assert p["a"] > 0.99 and p["b"] < 0.01


def test_p_migliore_e_incerta_con_pochi_dati_e_deterministica():
    scores = {"a": (5, 10), "b": (4, 10)}
    p = mt.p_best(scores)
    assert 0.5 < p["a"] < 0.9 and p == mt.p_best(scores)


def test_p_migliore_usa_il_prior_uniforme_beta_1_1(monkeypatch):
    seen = []
    real = mt.np.random.default_rng

    class Spy:
        def __init__(self):
            self.rng = real(0)

        def beta(self, a, b, size):
            seen.append((a, b))
            return self.rng.beta(a, b, size)

    monkeypatch.setattr(mt.np.random, "default_rng", lambda seed: Spy())
    mt.p_best({"a": (3, 10)})
    assert seen == [(4, 8)]  # 1+corrette, 1+errate


def test_vince_l_argmax_di_p_migliore_senza_soglie():
    vram = {"a": 6000, "b": 4000}
    out = mt.pick_winner({"a": (51, 100), "b": (50, 100)}, vram, {"a", "b"})
    assert out["winner"] == "a" and out["reason"] == "argmax P(migliore)"  # nessuna separazione richiesta
    assert set(out["p_best"]) == {"a", "b"} and set(out["p_best_all"]) == {"a", "b"}


def test_solo_a_parita_esatta_vince_il_minore_size_vram():
    vram = {"a": 6000, "b": 4000}
    out = mt.pick_winner({"a": (5, 10), "b": (5, 10)}, vram, {"a", "b"})
    assert out["winner"] == "b" and out["tied"] == ["a", "b"]
    near = mt.pick_winner({"a": (6, 10), "b": (5, 10)}, vram, {"a", "b"})
    assert near["winner"] == "a"  # niente pareggio per intervalli sovrapposti


def test_candidato_non_ammesso_non_vince_ma_ha_la_sua_p_nel_report():
    vram = {"a": 6000, "b": 4000}
    out = mt.pick_winner({"a": (9, 10), "b": (1, 10)}, vram, {"b"})
    assert out["winner"] == "b" and set(out["p_best_all"]) == {"a", "b"}
    assert mt.pick_winner({"a": (9, 10)}, vram, set())["winner"] is None


def test_run_usa_tutte_le_domande_e_ignora_i_non_ammessi(tmp_path, monkeypatch):
    def ask(host, model, ctx, question, hot):
        right = {"answer": "x", "paper": ["p0", "p1", "p2"]}
        # "b" e' perfetto ma sfora la VRAM: non deve vincere.
        return right, 1.0, {"size": 10, "size_vram": 10 if model == "a" else 5}

    res = _setup_run(tmp_path, monkeypatch, ask)
    assert res["rounds_completed"] == 3 and res["scores"]["a"] == [3, 3]  # nessun arresto anticipato
    assert res["admitted"] == ["a"] and res["decision"]["winner"] == "a"
    assert res["method"]["prior"] == "Beta(1, 1)" and "p_best" in res["decision"]


def test_ollama_ask_non_parte_se_la_soglia_e_gia_scattata(monkeypatch):
    import threading

    called = []
    import requests

    monkeypatch.setattr(requests, "post", lambda *a, **k: called.append(1))
    hot = threading.Event()
    hot.set()
    with pytest.raises(mt.ThermalStop):
        mt.ollama_ask("h", "m", "c", "q", hot)
    assert not called


def test_errore_ollama_scarta_il_round_e_lo_registra(tmp_path, monkeypatch):
    def ask(host, model, ctx, question, hot):
        raise mt.OllamaError("boom")

    res = _setup_run(tmp_path, monkeypatch, ask)
    assert res["rounds_completed"] == 0 and res["stop_reason"].startswith("errore ollama")


def test_run_rifiuta_ollama_su_un_altro_host(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "SCRATCH_ROOT", tmp_path)
    args = _args(tmp_path)
    args.ollama_host = "http://10.0.0.5:11434"
    with pytest.raises(SystemExit, match="questo host"):
        mt.cmd_run(args)


def test_run_rifiuta_candidati_duplicati(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "SCRATCH_ROOT", tmp_path)
    args = _args(tmp_path)
    args.models, args.ollama_host = ["a", "a"], "http://127.0.0.1:11434"
    with pytest.raises(SystemExit, match="duplicati"):
        mt.cmd_run(args)


def test_esito_vecchio_rimosso_all_avvio(tmp_path, monkeypatch):
    (tmp_path / "result.json").write_text('{"decision": {"winner": "vecchio"}}')

    def ask(host, model, ctx, question, hot):
        raise RuntimeError("fallisce prima di scrivere l'esito")

    with pytest.raises(RuntimeError):
        _setup_run(tmp_path, monkeypatch, ask)
    assert not (tmp_path / "result.json").exists()


def test_run_rifiuta_contesti_di_altre_domande(tmp_path, monkeypatch):
    def ask(host, model, ctx, question, hot):
        raise AssertionError("non deve partire")

    monkeypatch.setattr(mt, "SCRATCH_ROOT", tmp_path)
    (tmp_path / "questions.jsonl").write_text(json.dumps({"id": "q1", "paper": "p", "answer": "x", "question": "?"}) + "\n")
    (tmp_path / "contexts.jsonl").write_text(json.dumps({"id": "q1", "context": "c"}) + "\n")
    (tmp_path / "retrieval.json").write_text(json.dumps({"questions_sha1": "altra-impronta"}))
    monkeypatch.setattr(mt, "ollama_ask", ask)
    with pytest.raises(SystemExit, match="rilanciare retrieve"):
        mt.cmd_run(_args(tmp_path))
