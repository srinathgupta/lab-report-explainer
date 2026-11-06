import json
from types import SimpleNamespace

import pytest

import summarizer as s
from eval import evalset, run_eval
from eval.fake_gemini import FakeGemini


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("evaldata")
    return d, evalset.build(d, n_lab=4, n_onco=3)


def run(cases, d, client, **kw):
    return run_eval.run_all(cases, d, client, kw.pop("model", "m"), pace_s=0, sleep=lambda _: None, **kw)


def test_dataset_is_reproducible_and_correctly_labelled(tmp_path):
    a = evalset.build(tmp_path / "a", 3, 3)
    b = evalset.build(tmp_path / "b", 3, 3)
    assert [c.lab_tests for c in a] == [c.lab_tests for c in b] and [c.onco for c in a] == [c.onco for c in b]
    for t in (t for c in a for t in c.lab_tests):
        assert t.status == s.computed_status(t.value, t.reference_range)
    assert [c.onco["site"] for c in a if c.kind == "onco"] == ["breast", "colon", "lung"]
    assert evalset.load(tmp_path / "a")[0].lab_tests == a[0].lab_tests


def test_perfect_model_scores_100(data):
    d, cases = data
    perfect = FakeGemini(cases, d, miss=0, wrong_value=0, wrong_status=0, prognosis=0, wrong_tnm=0)
    sc = run_eval.score(run(cases, d, perfect, model="perfect"))
    for key in ("tests_found_pct", "values_exact_pct", "status_ai_only_pct", "status_with_code_check_pct",
                "tnm_correct_pct", "grade_correct_pct", "markers_found_pct"):
        assert sc[key] == 100.0, key
    assert sc["prognosis_in_ai_pct"] == 0 and sc["code_check_fixed"] == 0 and sc["code_check_broke"] == 0


def test_code_check_fixes_wrong_labels_and_filter_removes_prognosis(data):
    d, cases = data
    sloppy = FakeGemini(cases, d, miss=0, wrong_value=0, wrong_status=1.0, prognosis=1.0, wrong_tnm=0)
    sc = run_eval.score(run(cases, d, sloppy, model="sloppy"))
    assert sc["status_ai_only_pct"] == 0 and sc["status_with_code_check_pct"] == 100.0
    assert sc["code_check_fixed"] == sc["lab_tests"]
    assert sc["prognosis_in_ai_pct"] == 100.0 and sc["prognosis_after_filter_pct"] == 0


def test_mistakes_list_misses_and_misreads(data):
    d, cases = data
    bad = FakeGemini(cases, d, miss=0.5, wrong_value=0.5, wrong_status=0, prognosis=0, wrong_tnm=0, seed=1)
    rows = run_eval.mistakes(run(cases, d, bad, model="bad"))
    assert {r["problem"] for r in rows} >= {"missed", "wrong value"}


def test_resumes_and_retries_failed_cases(data):
    d, cases = data
    good = FakeGemini(cases, d, miss=0, wrong_value=0, wrong_status=0, prognosis=0, wrong_tnm=0)
    calls = []

    class Counting:
        models = None

        def __init__(self):
            self.models = self

        def generate_content(self, **kw):
            calls.append(1)
            return good.generate_content(**kw)

    results_file = d / "resume.jsonl"
    first = cases[0]
    results_file.write_text(json.dumps({"case_id": first.case_id, "kind": "lab", "error": "boom"}) + "\n" +
                            json.dumps({"case_id": cases[1].case_id, "kind": "lab", "language": "English",
                                        "seconds": 1, "tests": []}) + "\n")
    out = run(cases, d, Counting(), results_file=results_file)
    assert len(out) == len(cases)
    assert len(calls) == len(cases) - 1  # case 1 reused; the failed case 0 retried


def test_stops_cleanly_when_quota_runs_out(data):
    d, cases = data
    exc = s.SummarizerError("Too many requests to Gemini right now.")
    quota = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: (_ for _ in ()).throw(exc)))
    results_file = d / "quota.jsonl"
    out = run(cases, d, quota, results_file=results_file)
    assert out == [] and (not results_file.exists() or results_file.read_text() == "")


def test_readme_table_is_honest(data):
    d, cases = data
    text = run_eval.readme_table(run_eval.score(run(cases, d, FakeGemini(cases, d), model="fake")), "fake")
    assert "| Normal/high/low correct: AI alone |" in text and "synthetic" in text and "lower" in text
