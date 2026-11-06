"""Run the synthetic evaluation set through the real app code and score the answers.

The key comparison is "AI alone" versus "AI + the app's code checks": the same Gemini answer
is scored before and after the plain-code safety layer, so the value of that layer is measured,
not assumed.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path
from typing import Callable, Optional

from medexplain.core import SummarizerError, _generate, make_part
from medexplain.lab import SYSTEM_INSTRUCTION, ReportSummary, single_prompt, verify_statuses
from medexplain.oncology import (
    GRADE_MEANINGS,
    ONCOLOGY_SYSTEM_INSTRUCTION,
    PROGNOSIS_RE,
    OncologySummary,
    apply_oncology_safety,
    explain_grade,
    explain_tnm,
    oncology_prompt,
)

from .evalset import Case


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def same_number(a: str, b: str) -> bool:
    try:
        return abs(float(a.replace(",", "")) - float(b.replace(",", ""))) < 1e-9
    except ValueError:
        return norm(a) == norm(b)


def _find(name: str, predicted: list):
    key = norm(name)
    for p in predicted:
        if norm(p.name) == key:
            return p
    for p in predicted:  # e.g. "Haemoglobin (Hb)" vs "Haemoglobin"
        if key in norm(p.name) or norm(p.name) in key:
            return p
    return None


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------


def run_lab_case(case: Case, data_dir: Path, client, model: str, language: str, mode: str) -> dict:
    part = make_part((data_dir / case.pdf).read_bytes())
    start = time.time()
    summary = _generate(client, model, [part, single_prompt(language)], ReportSummary, mode, SYSTEM_INSTRUCTION)
    seconds = time.time() - start
    raw = summary.model_copy(deep=True)
    verify_statuses(summary)
    tests = []
    for truth in case.lab_tests:
        ai, final = _find(truth.name, raw.tests), _find(truth.name, summary.tests)
        tests.append({
            "name": truth.name, "truth_value": truth.value, "truth_status": truth.status,
            "range": truth.reference_range, "found": ai is not None,
            "ai_value": ai.value if ai else None, "value_ok": bool(ai and same_number(ai.value, truth.value)),
            "ai_status": ai.status if ai else None, "final_status": final.status if final else None,
        })
    return {"case_id": case.case_id, "kind": "lab", "language": language, "seconds": seconds, "tests": tests}


def _level(meaning: Optional[tuple[str, str]]) -> Optional[int]:
    if not meaning:
        return None
    for level, text in GRADE_MEANINGS.items():
        if text == meaning[1] and level.isdigit():
            return min(int(level), 3)  # grades 3 and 4 share "high grade"
    return None


def run_onco_case(case: Case, data_dir: Path, client, model: str, language: str, mode: str) -> dict:
    part = make_part((data_dir / case.pdf).read_bytes())
    start = time.time()
    onco = _generate(client, model, [part, oncology_prompt(language)], OncologySummary, mode,
                     ONCOLOGY_SYSTEM_INSTRUCTION)
    seconds = time.time() - start
    raw_text = json.dumps(onco.model_dump(exclude={"questions_for_doctor"}), ensure_ascii=False)
    cleaned, removed, _ = apply_oncology_safety(onco.model_copy(deep=True))
    final_text = json.dumps(cleaned.model_dump(exclude={"questions_for_doctor"}), ensure_ascii=False)

    truth = case.onco
    ai_codes = {c for c, _ in explain_tnm(onco.staging.tnm_as_written)}
    truth_codes = {c for c, _ in explain_tnm(truth["tnm"])}
    names = [b.name for b in onco.biomarkers]
    found_markers = [m for m in truth["markers"] if any(norm(m) in norm(n) for n in names)]
    return {
        "case_id": case.case_id, "kind": "onco", "language": language, "seconds": seconds,
        "site": truth["site"], "truth_tnm": truth["tnm"], "ai_tnm": onco.staging.tnm_as_written,
        "tnm_ok": ai_codes == truth_codes,
        "truth_grade": truth["grade"], "ai_grade": onco.staging.grade_as_written,
        "grade_ok": _level(explain_grade(onco.staging.grade_as_written)) == truth["grade"],
        "markers_total": len(truth["markers"]), "markers_found": len(found_markers),
        "prognosis_in_ai": bool(PROGNOSIS_RE.search(raw_text)), "prognosis_removed": removed,
        "prognosis_after_filter": bool(PROGNOSIS_RE.search(final_text)),
    }


# ---------------------------------------------------------------------------
# Whole run (paced, resumable)
# ---------------------------------------------------------------------------


def run_all(
    cases: list[Case], data_dir: Path, client, model: str, language: str = "English", mode: str = "fast",
    pace_s: float = 4.0, results_file: Optional[Path] = None, progress: Callable = lambda x, **k: x,
    sleep=time.sleep,
) -> list[dict]:
    """Run every case once. Results are appended to `results_file` as they arrive, so if Colab
    disconnects or the free quota runs out you can re-run and it continues where it stopped."""
    safe_model = re.sub(r"[^A-Za-z0-9.-]", "_", model)
    results_file = results_file or data_dir / f"results_{safe_model}_{language.lower()}.jsonl"
    done = {}
    if results_file.exists():
        for line in results_file.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r:  # failed cases are retried on the next run
                done[r["case_id"]] = r
    todo = [c for c in cases if c.case_id not in done]
    for i, case in enumerate(progress(todo, desc=f"{language} ({len(done)} already done)")):
        runner = run_lab_case if case.kind == "lab" else run_onco_case
        try:
            r = runner(case, data_dir, client, model, language, mode)
        except SummarizerError as e:
            if "Too many requests" in str(e) or "busy" in str(e):
                print(f"Stopped at {case.case_id}: {e} Re-run this cell later to continue where it stopped.")
                break
            r = {"case_id": case.case_id, "kind": case.kind, "language": language, "error": str(e)}
        with results_file.open("a") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        done[case.case_id] = r
        if i < len(todo) - 1:
            sleep(pace_s)
    return [done[c.case_id] for c in cases if c.case_id in done]


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


def _pct(num: int, den: int) -> Optional[float]:
    return round(100 * num / den, 1) if den else None


def score(results: list[dict]) -> dict:
    ok = [r for r in results if "error" not in r]
    lab = [t for r in ok if r["kind"] == "lab" for t in r["tests"]]
    found = [t for t in lab if t["found"]]
    abnormal = [t for t in lab if t["truth_status"] in ("high", "low")]
    onco = [r for r in ok if r["kind"] == "onco"]
    secs = [r["seconds"] for r in ok]
    fixed = sum(1 for t in found if t["ai_status"] != t["truth_status"] and t["final_status"] == t["truth_status"])
    broke = sum(1 for t in found if t["ai_status"] == t["truth_status"] and t["final_status"] != t["truth_status"])
    return {
        "reports_run": len(results), "errors": len(results) - len(ok),
        "lab_tests": len(lab),
        "tests_found_pct": _pct(len(found), len(lab)),
        "values_exact_pct": _pct(sum(t["value_ok"] for t in found), len(found)),
        "status_ai_only_pct": _pct(sum(t["ai_status"] == t["truth_status"] for t in found), len(found)),
        "status_with_code_check_pct": _pct(sum(t["final_status"] == t["truth_status"] for t in found), len(found)),
        "abnormal_caught_ai_only_pct": _pct(sum(t["ai_status"] == t["truth_status"] for t in abnormal), len(abnormal)),
        "abnormal_caught_with_code_check_pct": _pct(sum(t["final_status"] == t["truth_status"] for t in abnormal), len(abnormal)),
        "code_check_fixed": fixed, "code_check_broke": broke,
        "onco_reports": len(onco),
        "tnm_correct_pct": _pct(sum(r["tnm_ok"] for r in onco), len(onco)),
        "grade_correct_pct": _pct(sum(r["grade_ok"] for r in onco), len(onco)),
        "markers_found_pct": _pct(sum(r["markers_found"] for r in onco), sum(r["markers_total"] for r in onco)),
        "prognosis_in_ai_pct": _pct(sum(r["prognosis_in_ai"] for r in onco), len(onco)),
        "prognosis_after_filter_pct": _pct(sum(r["prognosis_after_filter"] for r in onco), len(onco)),
        "median_seconds": round(statistics.median(secs), 1) if secs else None,
    }


LABELS = {
    "tests_found_pct": "Lab tests found on the report",
    "values_exact_pct": "Values read exactly right",
    "status_ai_only_pct": "Normal/high/low correct: AI alone",
    "status_with_code_check_pct": "Normal/high/low correct: AI + code check",
    "abnormal_caught_ai_only_pct": "Abnormal results caught: AI alone",
    "abnormal_caught_with_code_check_pct": "Abnormal results caught: AI + code check",
    "tnm_correct_pct": "Cancer TNM stage read correctly",
    "grade_correct_pct": "Cancer grade read correctly",
    "markers_found_pct": "Tumour markers (ER, HER2, PD-L1…) found",
    "prognosis_in_ai_pct": "Reports where AI mentioned survival/outlook",
    "prognosis_after_filter_pct": "…still present after the code filter",
}


def mistakes(results: list[dict]) -> list[dict]:
    """Every lab result the app still got wrong, for error analysis."""
    rows = []
    for r in results:
        if r.get("kind") != "lab" or "error" in r:
            continue
        for t in r["tests"]:
            if not t["found"] or not t["value_ok"] or t["final_status"] != t["truth_status"]:
                problem = "missed" if not t["found"] else "wrong value" if not t["value_ok"] else "wrong status"
                rows.append({"case": r["case_id"], "test": t["name"], "problem": problem,
                             "truth": f"{t['truth_value']} ({t['truth_status']})",
                             "app": f"{t['ai_value']} ({t['final_status']})" if t["found"] else "—",
                             "range": t["range"]})
    return rows


def readme_table(s: dict, model: str) -> str:
    lines = [f"**Evaluation** on {s['reports_run'] - s['errors']} synthetic reports ({s['lab_tests']} lab results, "
             f"{s['onco_reports']} cancer biopsy reports), model `{model}`:", "", "| Check | Score |", "|---|---|"]
    for key, label in LABELS.items():
        if s.get(key) is not None:
            lines.append(f"| {label} | {s[key]}% |")
    lines += ["", f"The code check corrected {s['code_check_fixed']} AI mistakes and introduced {s['code_check_broke']}. "
              "The correct normal/high/low answer is defined from the printed reference range (the same rule the "
              "code check applies), so the check can only be as good as the value it reads; see \"values read "
              "exactly right\". Reports are synthetic, generated with known answers by `eval/evalset.py`, and "
              "cleaner than real scans or photos, so real-world accuracy will be lower. Reproduce with "
              "`eval/Lab_Report_Explainer_Eval.ipynb`."]
    return "\n".join(lines)
