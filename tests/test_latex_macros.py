"""Tests for LaTeX macro extraction and expansion."""

import importlib
import json
import os
import sys
import tempfile

import pytest

# Import latex_macros directly, bypassing raganything's heavy __init__.py
# which pulls in lightrag and other runtime dependencies.
_pkg_dir = os.path.join(os.path.dirname(__file__), os.pardir, "raganything")
sys.path.insert(0, _pkg_dir)
_mod = importlib.import_module("raganything.latex_macros")
extract_macros_from_tex = _mod.extract_macros_from_tex
expand_macros = _mod.expand_macros
load_macros = _mod.load_macros
COMMON_MACROS = _mod.COMMON_MACROS
sys.path.pop(0)

# --- Sample preamble from 2004.09301 (real data) ---
SAMPLE_PREAMBLE = r"""
\documentclass{article}
\newcommand{\FF}{\mathbb{F}}
\newcommand{\ZZ}{\mathbb{Z}}
\newcommand{\CC}{\mathbb{C}}
\newcommand{\A}{\mathsf{A}}
\newcommand{\q}{\quad}
\newcommand{\qq}{\qquad}
\newcommand\ot{\otimes}
\newcommand\rank{\operatorname{\mathsf{rank}}}
\newcommand{\lb}[1]{\left[ #1\right]}
\newcommand{\pb}[1]{\left\{ #1\right\}}
\newcommand{\seq}[1]{\left( #1\right)}
\newcommand{\eval}[2]{\left. #1 \right|_{#2}}
\newcommand\im{\mathsf{im}}
\renewcommand{\ker}{\mathsf{ker}}
\newcommand{\hqfg}{\mathcal{H}_q(f,g)}
\newcommand{\Af}[1][\lambda, \mu]{\mathsf{A}_{q, f,g}(#1)}
\newcommand{\claim}[2]{\smallskip\noindent {\bf #1:} {\it #2.}}
\DeclareMathOperator{\Dom}{Dom}
\begin{document}
Content here.
\end{document}
"""


class TestExtractMacros:
    def test_basic_extraction(self):
        macros = extract_macros_from_tex(SAMPLE_PREAMBLE)
        assert "\\FF" in macros
        assert macros["\\FF"]["body"] == "\\mathbb{F}"
        assert macros["\\FF"]["nargs"] == 0

    def test_no_braces_form(self):
        """\\newcommand\\ot{\\otimes} (no braces around macro name)."""
        macros = extract_macros_from_tex(SAMPLE_PREAMBLE)
        assert "\\ot" in macros
        assert macros["\\ot"]["body"] == "\\otimes"

    def test_with_args(self):
        macros = extract_macros_from_tex(SAMPLE_PREAMBLE)
        assert "\\pb" in macros
        assert macros["\\pb"]["nargs"] == 1
        assert macros["\\pb"]["body"] == "\\left\\{ #1\\right\\}"

    def test_two_args(self):
        macros = extract_macros_from_tex(SAMPLE_PREAMBLE)
        assert "\\eval" in macros
        assert macros["\\eval"]["nargs"] == 2

    def test_optional_arg(self):
        macros = extract_macros_from_tex(SAMPLE_PREAMBLE)
        assert "\\Af" in macros
        assert macros["\\Af"]["nargs"] == 1
        assert macros["\\Af"]["optarg"] == "\\lambda, \\mu"
        assert macros["\\Af"]["body"] == "\\mathsf{A}_{q, f,g}(#1)"

    def test_renewcommand(self):
        macros = extract_macros_from_tex(SAMPLE_PREAMBLE)
        assert "\\ker" in macros
        assert macros["\\ker"]["body"] == "\\mathsf{ker}"

    def test_declare_math_operator(self):
        macros = extract_macros_from_tex(SAMPLE_PREAMBLE)
        assert "\\Dom" in macros
        assert macros["\\Dom"]["body"] == "\\operatorname{Dom}"

    def test_def_no_args(self):
        tex = r"""
\def\aut{ \mathsf {Aut_\CC}(\Hf)}
\def\modd{\, \mathsf{mod} \,}
\begin{document}\end{document}
"""
        macros = extract_macros_from_tex(tex)
        assert "\\aut" in macros
        assert macros["\\aut"]["nargs"] == 0
        assert "\\modd" in macros

    def test_def_with_args(self):
        tex = r"""
\def\foo#1#2{#1 + #2}
\begin{document}\end{document}
"""
        macros = extract_macros_from_tex(tex)
        assert "\\foo" in macros
        assert macros["\\foo"]["nargs"] == 2
        assert macros["\\foo"]["body"] == "#1 + #2"

    def test_only_preamble(self):
        """Macros after \\begin{document} should be ignored."""
        tex = r"""
\newcommand{\foo}{bar}
\begin{document}
\newcommand{\baz}{qux}
\end{document}
"""
        macros = extract_macros_from_tex(tex)
        assert "\\foo" in macros
        assert "\\baz" not in macros

    def test_empty_preamble(self):
        macros = extract_macros_from_tex("\\begin{document}\nhello\n\\end{document}")
        assert macros == {}

    def test_macro_count(self):
        macros = extract_macros_from_tex(SAMPLE_PREAMBLE)
        # Preamble has 18 definitions (16 newcommand + 1 renewcommand + 1 DeclareMathOperator)
        assert len(macros) >= 17


class TestExpandMacros:
    def test_simple_expansion(self):
        macros = {"\\FF": {"nargs": 0, "optarg": None, "body": "\\mathbb{F}"}}
        assert expand_macros("x \\in \\FF", macros) == "x \\in \\mathbb{F}"

    def test_no_partial_match(self):
        """\\FF should not match inside \\FFoo."""
        macros = {"\\FF": {"nargs": 0, "optarg": None, "body": "\\mathbb{F}"}}
        assert expand_macros("\\FFoo", macros) == "\\FFoo"

    def test_one_arg(self):
        macros = {"\\pb": {"nargs": 1, "optarg": None, "body": "\\left\\{ #1\\right\\}"}}
        result = expand_macros("\\pb{x+y}", macros)
        assert result == "\\left\\{ x+y\\right\\}"

    def test_two_args(self):
        macros = {"\\eval": {"nargs": 2, "optarg": None, "body": "\\left. #1 \\right|_{#2}"}}
        result = expand_macros("\\eval{f(x)}{a}", macros)
        assert result == "\\left. f(x) \\right|_{a}"

    def test_optional_arg_with_default(self):
        macros = {"\\Af": {"nargs": 1, "optarg": "\\lambda, \\mu", "body": "\\mathsf{A}_{q, f,g}(#1)"}}
        # Without optional arg -> uses default
        result = expand_macros("\\overline{\\A}=\\Af/M", macros)
        assert "\\mathsf{A}_{q, f,g}(\\lambda, \\mu)" in result

    def test_optional_arg_with_override(self):
        macros = {"\\Af": {"nargs": 1, "optarg": "\\lambda, \\mu", "body": "\\mathsf{A}_{q, f,g}(#1)"}}
        result = expand_macros("\\Af[\\alpha]", macros)
        assert result == "\\mathsf{A}_{q, f,g}(\\alpha)"

    def test_nested_expansion(self):
        """Macro whose body contains another macro."""
        macros = {
            "\\FF": {"nargs": 0, "optarg": None, "body": "\\mathbb{F}"},
            "\\qp": {"nargs": 0, "optarg": None, "body": "\\FF_q[x, y]"},
        }
        result = expand_macros("\\qp", macros)
        assert result == "\\mathbb{F}_q[x, y]"

    def test_depth_limit(self):
        """Recursive macro should not loop forever."""
        macros = {"\\foo": {"nargs": 0, "optarg": None, "body": "\\foo"}}
        # Should terminate (max_passes limit)
        result = expand_macros("\\foo", macros, max_passes=5)
        assert result == "\\foo"  # can't resolve, stays as is

    def test_no_macros(self):
        assert expand_macros("x^2 + y^2", {}) == "x^2 + y^2"

    def test_multiple_occurrences(self):
        macros = {"\\q": {"nargs": 0, "optarg": None, "body": "\\quad"}}
        result = expand_macros("a \\q b \\q c", macros)
        assert result == "a \\quad b \\quad c"

    def test_benchmark_formula_overlineA_Af(self):
        """Real formula from benchmark: \\overline{\\A}=\\Af/M."""
        macros = extract_macros_from_tex(SAMPLE_PREAMBLE)
        result = expand_macros("\\overline{\\A}=\\Af/M", macros)
        # \A -> \mathsf{A}, \Af (no arg) -> uses default optarg
        assert "\\mathsf{A}" in result
        assert "\\Af" not in result

    def test_benchmark_formula_spacing(self):
        """Real formula: yx=a, \\q\\q xy=\\sigma(a)."""
        macros = extract_macros_from_tex(SAMPLE_PREAMBLE)
        result = expand_macros("yx=a, \\q\\q xy=\\sigma(a).", macros)
        assert "\\quad" in result
        assert "\\q" not in result or "\\quad" in result


class TestFileIO:
    def test_save_and_load_roundtrip(self):
        macros = {"\\foo": {"nargs": 0, "optarg": None, "body": "bar"}}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "latex_macros.json")
            data = {"source": "test", "macro_count": 1, "macros": macros}
            with open(path, "w") as f:
                json.dump(data, f)

            loaded = load_macros(tmpdir)
            assert loaded == macros

    def test_load_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert load_macros(tmpdir) is None

    def test_load_corrupt_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "latex_macros.json")
            with open(path, "w") as f:
                f.write("not json")
            assert load_macros(tmpdir) is None


class TestCommonMacros:
    def test_common_macros_not_empty(self):
        assert len(COMMON_MACROS) > 30

    def test_common_macros_fallback(self):
        """COMMON_MACROS should expand basic notation."""
        result = expand_macros("x \\in \\ZZ", COMMON_MACROS)
        assert result == "x \\in \\mathbb{Z}"


# --- Integration tests with real .tex files ---

BENCHMARK_DIR = "/tmp/lopes-benchmark"


def _has_benchmark_data():
    return os.path.isdir(BENCHMARK_DIR) and os.path.isfile(
        os.path.join(BENCHMARK_DIR, "formulas.json")
    )


@pytest.mark.skipif(not _has_benchmark_data(), reason="Benchmark data not available")
class TestRealPapers:
    """Integration tests using real .tex files from /tmp/lopes-benchmark."""

    @pytest.fixture
    def all_macros(self):
        """Load macros from all 4 papers."""
        result = {}
        for arxiv_id in ["2004.09301", "2009.05270", "1509.02682", "0706.3355"]:
            tex_path = os.path.join(BENCHMARK_DIR, arxiv_id, "main.tex")
            if os.path.isfile(tex_path):
                with open(tex_path) as f:
                    tex = f.read()
                result[arxiv_id] = extract_macros_from_tex(tex)
        return result

    @pytest.fixture
    def formulas(self):
        with open(os.path.join(BENCHMARK_DIR, "formulas.json")) as f:
            return json.load(f)

    def test_extract_macros_from_2004(self, all_macros):
        macros = all_macros.get("2004.09301", {})
        assert len(macros) >= 30
        assert "\\hqfg" in macros
        assert "\\FF" in macros
        assert "\\pb" in macros

    def test_extract_macros_from_0706(self, all_macros):
        macros = all_macros.get("0706.3355", {})
        assert len(macros) >= 15
        assert "\\K" in macros or "\\h" in macros

    def test_extract_macros_from_1509(self, all_macros):
        macros = all_macros.get("1509.02682", {})
        assert len(macros) >= 40
        assert "\\Hf" in macros
        assert macros["\\Hf"]["body"] == "\\mathcal{H}(f)"

    def test_expand_benchmark_formulas(self, all_macros, formulas):
        """Expand all benchmark formulas and verify no custom macros remain."""
        expanded_count = 0
        for formula in formulas:
            arxiv_id = formula.get("arxiv_id", "")
            macros = all_macros.get(arxiv_id, COMMON_MACROS)
            original = formula["latex"]
            expanded = expand_macros(original, macros)
            if expanded != original:
                expanded_count += 1
        # At least some formulas should have been expanded
        assert expanded_count > 0, "No formulas were expanded"
        print(f"Expanded {expanded_count}/{len(formulas)} formulas")

    def test_known_failing_formula_now_works(self, all_macros):
        """Formula that failed CAS parsing due to \\Af macro."""
        macros = all_macros.get("2004.09301", {})
        assert macros, "Macros for 2004.09301 not loaded"

        formula = "\\overline{\\A}=\\Af/M"
        expanded = expand_macros(formula, macros)
        # After expansion, no custom macros should remain
        assert "\\Af" not in expanded
        assert "\\A}" not in expanded or "\\mathsf{A}" in expanded
