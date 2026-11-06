"""A stand-in for Gemini that answers from the ground truth but makes realistic mistakes
at fixed rates. Used to test the evaluation pipeline (and the notebook's DRY_RUN mode)
without an API key or quota. Its scores say nothing about the real model."""

from __future__ import annotations

import hashlib
import json
import random
from types import SimpleNamespace

from medexplain.lab import ReportSummary
from medexplain.oncology import OncologySummary

from .evalset import Case


class FakeGemini:
    def __init__(self, cases: list[Case], data_dir, seed: int = 7, miss=0.03, wrong_value=0.03,
                 wrong_status=0.12, prognosis=0.3, wrong_tnm=0.1):
        self.by_hash = {hashlib.sha1((data_dir / c.pdf).read_bytes()).hexdigest(): c for c in cases}
        self.rng = random.Random(seed)
        self.rates = dict(miss=miss, wrong_value=wrong_value, wrong_status=wrong_status,
                          prognosis=prognosis, wrong_tnm=wrong_tnm)
        self.models = self

    def _chance(self, name: str) -> bool:
        return self.rng.random() < self.rates[name]

    def generate_content(self, model, contents, config):
        case = self.by_hash[hashlib.sha1(contents[0].inline_data.data).hexdigest()]
        if config.response_schema is ReportSummary:
            return SimpleNamespace(text=json.dumps(self._lab(case)))
        if config.response_schema is OncologySummary:
            return SimpleNamespace(text=json.dumps(self._onco(case)))
        raise ValueError("unexpected schema")

    def _lab(self, case: Case) -> dict:
        tests = []
        for t in case.lab_tests:
            if self._chance("miss"):
                continue
            value = t.value
            if self._chance("wrong_value"):
                value = value[:-1] + str((int(value[-1]) + 1) % 10)  # a misread last digit
            status = t.status
            if self._chance("wrong_status"):
                status = self.rng.choice([s for s in ("normal", "high", "low") if s != t.status])
            tests.append({"name": t.name, "category": "Panel", "value": value, "unit": t.unit,
                          "reference_range": t.reference_range, "status": status, "explanation": "..."})
        return {"is_lab_report": True, "report_date": "", "lab_name": "", "overall_summary": "Summary.",
                "key_findings": [], "urgent_flags": [], "tests": tests, "questions_for_doctor": [], "notes": ""}

    def _onco(self, case: Case) -> dict:
        o = case.onco
        tnm = o["tnm"].replace("pN0", "pN1") if self._chance("wrong_tnm") else o["tnm"]
        summary = "The report describes the cancer that was removed."
        if self._chance("prognosis"):
            summary += " The 5-year survival rate for this stage is good."
        item = lambda: {"as_written": "", "plain_explanation": ""}  # noqa: E731
        return {
            "document_kind": "pathology_report", "cancer_type_as_written": o["site"], "organ": o["site"], "date": "",
            "overall_summary": summary, "key_points": [],
            "staging": {"tnm_as_written": tnm, "stage_group_as_written": "",
                        "grade_as_written": f"Grade {o['grade']}"},
            "tumour_size": item(), "margins": item(), "lymph_nodes": item(),
            "biomarkers": [{"name": m, "result_as_written": "x", "plain_explanation": "y"} for m in o["markers"]],
            "tumour_markers": [], "terms": [], "questions_for_doctor": [], "notes": "",
        }
