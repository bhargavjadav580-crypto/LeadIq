"""LeadIQ backend integration tests covering end-to-end and parser validation flows."""

import io
import json
import os
import time
from pathlib import Path

import pandas as pd
import pytest
import requests
import yaml
from dotenv import load_dotenv


ROOT = Path("/app")
load_dotenv(ROOT / "frontend/.env")
load_dotenv(ROOT / "backend/.env")
BASE_URL = os.environ.get("REACT_APP_BACKEND_URL")


def _require_base_url():
    if not BASE_URL:
        pytest.skip("REACT_APP_BACKEND_URL is missing; cannot run API integration tests")


def _url(path: str) -> str:
    return f"{BASE_URL.rstrip('/')}{path}"


def _headers(token: str) -> dict:
    return {"X-Project-Key": token}


def _create_project(session: requests.Session):
    r = session.post(_url("/api/projects"), timeout=30)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("id") and data.get("token")
    return data["id"], data["token"]


def _poll_job(session: requests.Session, pid: str, token: str, job_id: str, timeout_seconds: int = 900):
    deadline = time.time() + timeout_seconds
    last = None
    while time.time() < deadline:
        r = session.get(_url(f"/api/projects/{pid}/jobs/{job_id}"), headers=_headers(token), timeout=30)
        assert r.status_code == 200, r.text
        last = r.json()
        if last.get("status") in {"completed", "failed"}:
            return last
        time.sleep(2)
    raise AssertionError(f"Job timeout after {timeout_seconds}s. Last state: {last}")


@pytest.fixture(scope="module")
def session():
    _require_base_url()
    s = requests.Session()
    s.headers.update({"Accept": "application/json"})
    return s


def _csv_bytes(df: pd.DataFrame, sep=",", encoding="utf-8") -> bytes:
    return df.to_csv(index=False, sep=sep).encode(encoding)


def _xlsx_bytes(sheet_to_df: dict[str, pd.DataFrame]) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for name, df in sheet_to_df.items():
            df.to_excel(writer, sheet_name=name, index=False)
    return out.getvalue()


def test_health_and_cron_manifest(session):
    r = session.get(_url("/api/health"), timeout=30)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["service"] == "LeadIQ"

    cron_cfg = yaml.safe_load((ROOT / ".emergent/crons.yml").read_text())
    assert isinstance(cron_cfg.get("crons"), list) and cron_cfg["crons"]
    entry = cron_cfg["crons"][0]
    assert entry["endpoint"] == "{{BASE_URL}}/api/cron/cleanup"
    assert entry["method"] == "POST"
    assert entry["enabled"] is True


def test_project_key_enforcement_and_isolation(session):
    pid, token = _create_project(session)

    missing = session.get(_url(f"/api/projects/{pid}"), timeout=30)
    assert missing.status_code in {404, 422}

    valid = session.get(_url(f"/api/projects/{pid}"), headers=_headers(token), timeout=30)
    assert valid.status_code == 200
    pdata = valid.json()
    assert pdata["id"] == pid

    pid2, token2 = _create_project(session)
    wrong_token = session.get(_url(f"/api/projects/{pid2}"), headers=_headers(token), timeout=30)
    assert wrong_token.status_code in {404, 410}

    cleanup = session.delete(_url(f"/api/projects/{pid}"), headers=_headers(token), timeout=30)
    assert cleanup.status_code == 200
    cleanup2 = session.delete(_url(f"/api/projects/{pid2}"), headers=_headers(token2), timeout=30)
    assert cleanup2.status_code == 200


def test_sample_full_nine_step_flow_and_exports(session):
    """Sample data full flow: upload/configure/clean/features/train/evaluate/score/explain/export."""
    pid, token = _create_project(session)
    h = _headers(token)

    sample = session.post(_url(f"/api/projects/{pid}/sample"), headers=h, timeout=60)
    assert sample.status_code == 200, sample.text
    sdata = sample.json()
    assert sdata["rows"] >= 5000
    assert sdata["filename"].endswith(".csv")

    preview = session.get(_url(f"/api/projects/{pid}/preview"), headers=h, timeout=60)
    assert preview.status_code == 200
    pv = preview.json()
    assert pv["rows"] >= 5000
    assert any(c["name"] == "Converted" for c in pv["columns"])

    cfg = session.post(
        _url(f"/api/projects/{pid}/configure"),
        headers={**h, "Content-Type": "application/json"},
        json={"target": "Converted"},
        timeout=60,
    )
    assert cfg.status_code == 200, cfg.text
    cdata = cfg.json()
    assert cdata["target"] == "Converted"
    assert cdata["labeled"] >= 200
    assert cdata["positives"] >= 8

    clean_start = session.post(
        _url(f"/api/projects/{pid}/clean"),
        headers={**h, "Content-Type": "application/json"},
        json={"keep_columns": []},
        timeout=60,
    )
    assert clean_start.status_code == 200, clean_start.text
    clean_job = _poll_job(session, pid, token, clean_start.json()["job_id"], timeout_seconds=240)
    assert clean_job["status"] == "completed", clean_job

    quality = session.get(_url(f"/api/projects/{pid}/quality"), headers=h, timeout=60)
    assert quality.status_code == 200
    q = quality.json()
    assert q["rows_after"] >= 200
    assert q["rows_before"] >= q["rows_after"]
    assert q["exact_duplicates"] >= 100

    eda = session.get(_url(f"/api/projects/{pid}/eda"), headers=h, timeout=60)
    assert eda.status_code == 200
    ed = eda.json()
    assert "balance" in ed and len(ed["balance"]) == 2

    feat_start = session.post(
        _url(f"/api/projects/{pid}/features"),
        headers={**h, "Content-Type": "application/json"},
        json={"pretrained": True},
        timeout=60,
    )
    assert feat_start.status_code == 200, feat_start.text
    feat_job = _poll_job(session, pid, token, feat_start.json()["job_id"], timeout_seconds=300)
    assert feat_job["status"] == "completed", feat_job

    feat = session.get(_url(f"/api/projects/{pid}/features"), headers=h, timeout=60)
    assert feat.status_code == 200
    f = feat.json()
    assert f["split"]["train"] > 0 and f["split"]["test"] > 0
    assert f["count"] > 10
    assert isinstance(f["required_columns"], list) and f["required_columns"]

    train_start = session.post(
        _url(f"/api/projects/{pid}/train"),
        headers={**h, "Content-Type": "application/json"},
        json={"mode": "quick"},
        timeout=60,
    )
    assert train_start.status_code == 200, train_start.text
    train_job = _poll_job(session, pid, token, train_start.json()["job_id"], timeout_seconds=900)
    assert train_job["status"] == "completed", train_job

    metrics = session.get(_url(f"/api/projects/{pid}/metrics"), headers=h, timeout=60)
    assert metrics.status_code == 200
    m = metrics.json()
    assert len(m["models"]) == 5
    assert m["best_model"]
    assert m["best"]["pr_auc"] >= 0

    verify = session.get(_url(f"/api/projects/{pid}/verification"), headers=h, timeout=60)
    assert verify.status_code == 200
    v = verify.json()
    assert len(v["rows"]) == 20
    assert v["all_match"] is True

    threshold = session.post(
        _url(f"/api/projects/{pid}/threshold"),
        headers={**h, "Content-Type": "application/json"},
        json={"model": m["best_model"], "threshold": 0.42},
        timeout=60,
    )
    assert threshold.status_code == 200
    t = threshold.json()
    assert "confusion_matrix" in t and len(t["confusion_matrix"]) == 2

    importance = session.get(_url(f"/api/projects/{pid}/importance"), headers=h, timeout=60)
    assert importance.status_code == 200
    imp = importance.json()
    assert "status" in imp

    score_test = session.post(
        _url(f"/api/projects/{pid}/score"),
        headers={**h, "Content-Type": "application/json"},
        json={"scope": "test"},
        timeout=60,
    )
    assert score_test.status_code == 200
    score_job = _poll_job(session, pid, token, score_test.json()["job_id"], timeout_seconds=240)
    assert score_job["status"] == "completed", score_job

    summary = session.get(_url(f"/api/projects/{pid}/score-summary"), headers=h, timeout=60)
    assert summary.status_code == 200
    s = summary.json()
    assert s["source"] == "test"
    assert s["total"] > 0
    assert sum(s["tiers"].values()) == s["total"]

    leads = session.get(_url(f"/api/projects/{pid}/leads"), headers=h, params={"page_size": 20}, timeout=60)
    assert leads.status_code == 200
    l = leads.json()
    assert l["total"] > 0 and len(l["rows"]) > 0
    lead_id = l["rows"][0]["lead_id"]

    explain = session.get(_url(f"/api/projects/{pid}/explain/{lead_id}"), headers=h, timeout=120)
    assert explain.status_code == 200, explain.text
    ex = explain.json()
    assert ex["lead_id"] == lead_id
    assert isinstance(ex.get("contributions"), list)

    score_all = session.post(
        _url(f"/api/projects/{pid}/score"),
        headers={**h, "Content-Type": "application/json"},
        json={"scope": "all"},
        timeout=60,
    )
    assert score_all.status_code == 200
    score_all_job = _poll_job(session, pid, token, score_all.json()["job_id"], timeout_seconds=240)
    assert score_all_job["status"] == "completed", score_all_job

    summary_call = session.get(_url(f"/api/projects/{pid}/summary"), headers=h, timeout=60)
    assert summary_call.status_code == 200
    methodology = summary_call.json()
    assert methodology["best_model"] == m["best_model"]
    assert methodology["seed"] == 42

    for fmt in ["csv", "xlsx", "pdf", "metrics", "reproducibility"]:
        exp = session.get(_url(f"/api/projects/{pid}/export"), headers=h, params={"format": fmt}, timeout=120)
        assert exp.status_code == 200, f"{fmt}: {exp.text[:200]}"
        assert len(exp.content) > 50

    del_ok = session.delete(_url(f"/api/projects/{pid}"), headers=h, timeout=30)
    assert del_ok.status_code == 200


def test_score_new_sheet_selection_and_missing_feature_error(session):
    """New-file scoring checks: needs_sheet, missing required features, drift metadata."""
    pid, token = _create_project(session)
    h = _headers(token)

    session.post(_url(f"/api/projects/{pid}/sample"), headers=h, timeout=60)
    cfg = session.post(
        _url(f"/api/projects/{pid}/configure"),
        headers={**h, "Content-Type": "application/json"},
        json={"target": "Converted"},
        timeout=60,
    )
    assert cfg.status_code == 200

    c = session.post(_url(f"/api/projects/{pid}/clean"), headers={**h, "Content-Type": "application/json"}, json={"keep_columns": []}, timeout=60)
    assert _poll_job(session, pid, token, c.json()["job_id"], timeout_seconds=240)["status"] == "completed"
    f = session.post(_url(f"/api/projects/{pid}/features"), headers={**h, "Content-Type": "application/json"}, json={"pretrained": True}, timeout=60)
    assert _poll_job(session, pid, token, f.json()["job_id"], timeout_seconds=300)["status"] == "completed"
    t = session.post(_url(f"/api/projects/{pid}/train"), headers={**h, "Content-Type": "application/json"}, json={"mode": "quick"}, timeout=60)
    assert _poll_job(session, pid, token, t.json()["job_id"], timeout_seconds=900)["status"] == "completed"

    base_df = pd.DataFrame(
        {
            "Business Name": ["A", "B", "C"],
            "Category": ["retail", "retail", "services"],
            "City": ["ny", "sf", "la"],
            "Rating": [4.2, 3.9, 4.8],
            "Review Count": [12, 45, 9],
            "Website": ["https://a.com", "https://b.com", "https://c.com"],
            "Phone": ["+14155550111", "+14155550112", "+14155550113"],
            "Email": ["a@x.com", "b@x.com", "c@x.com"],
        }
    )
    multi_xlsx = _xlsx_bytes({"Sheet1": base_df, "Sheet2": base_df.copy()})
    r = session.post(
        _url(f"/api/projects/{pid}/score-new"),
        headers=h,
        files={"file": ("new_leads.xlsx", multi_xlsx, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        timeout=90,
    )
    assert r.status_code == 200
    assert r.json().get("needs_sheet") is True
    assert len(r.json().get("sheets", [])) == 2

    bad_df = base_df.drop(columns=["Category"]) if "Category" in base_df.columns else base_df.drop(columns=[base_df.columns[0]])
    bad_csv = _csv_bytes(bad_df)
    miss = session.post(
        _url(f"/api/projects/{pid}/score-new"),
        headers=h,
        files={"file": ("missing_feature.csv", bad_csv, "text/csv")},
        timeout=90,
    )
    assert miss.status_code == 400
    assert "Missing required columns" in miss.json().get("detail", "")

    good = session.post(
        _url(f"/api/projects/{pid}/score-new"),
        headers=h,
        files={"file": ("new_leads.xlsx", multi_xlsx, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        data={"sheet": "Sheet2"},
        timeout=90,
    )
    assert good.status_code == 200
    sj = _poll_job(session, pid, token, good.json()["job_id"], timeout_seconds=240)
    assert sj["status"] == "completed", sj

    summary = session.get(_url(f"/api/projects/{pid}/score-summary"), headers=h, timeout=60)
    assert summary.status_code == 200
    s = summary.json()
    assert s["source"] == "new"
    assert "new_file" in s

    session.delete(_url(f"/api/projects/{pid}"), headers=h, timeout=30)


def test_parser_adversarial_cases_and_target_guards(session):
    """Parser matrix: latin-1 semicolon, xls, pdf reject, one-class/tiny/continuous target blocks."""
    pid, token = _create_project(session)
    h = _headers(token)

    rows = 260
    converted = ["yes" if i % 7 == 0 else "no" for i in range(rows)]
    latin_df = pd.DataFrame(
        {
            "Business Name": [f"Café {i}" for i in range(rows)],
            "Category": ["resto" if i % 2 else "shop" for i in range(rows)],
            "City": ["München" if i % 3 else "São Paulo" for i in range(rows)],
            "Converted": converted,
        }
    )
    latin_csv = _csv_bytes(latin_df, sep=";", encoding="latin-1")
    upload_latin = session.post(
        _url(f"/api/projects/{pid}/upload"),
        headers=h,
        files={"file": ("latin_semicolon.csv", latin_csv, "text/csv")},
        timeout=90,
    )
    assert upload_latin.status_code == 200, upload_latin.text
    prev = session.get(_url(f"/api/projects/{pid}/preview"), headers=h, timeout=60)
    assert prev.status_code == 200
    notices = " ".join(prev.json().get("notices", []))
    assert "delimiter" in notices.lower()

    one_class = latin_df.copy()
    one_class["Converted"] = "yes"
    one_class_csv = _csv_bytes(one_class)
    session.post(_url(f"/api/projects/{pid}/upload"), headers=h, files={"file": ("one_class.csv", one_class_csv, "text/csv")}, timeout=90)
    one_cfg = session.post(
        _url(f"/api/projects/{pid}/configure"),
        headers={**h, "Content-Type": "application/json"},
        json={"target": "Converted"},
        timeout=60,
    )
    assert one_cfg.status_code == 400
    assert "only one class" in one_cfg.json().get("detail", "").lower()

    tiny_df = latin_df.head(180)
    tiny_csv = _csv_bytes(tiny_df)
    session.post(_url(f"/api/projects/{pid}/upload"), headers=h, files={"file": ("tiny.csv", tiny_csv, "text/csv")}, timeout=90)
    tiny_cfg = session.post(
        _url(f"/api/projects/{pid}/configure"),
        headers={**h, "Content-Type": "application/json"},
        json={"target": "Converted"},
        timeout=60,
    )
    assert tiny_cfg.status_code == 400
    assert "at least 200" in tiny_cfg.json().get("detail", "").lower()

    cont_df = latin_df.copy()
    cont_df["Converted"] = [float(i) / 10.0 for i in range(rows)]
    cont_csv = _csv_bytes(cont_df)
    session.post(_url(f"/api/projects/{pid}/upload"), headers=h, files={"file": ("continuous.csv", cont_csv, "text/csv")}, timeout=90)
    cont_cfg = session.post(
        _url(f"/api/projects/{pid}/configure"),
        headers={**h, "Content-Type": "application/json"},
        json={"target": "Converted"},
        timeout=60,
    )
    assert cont_cfg.status_code == 400
    assert "continuous" in cont_cfg.json().get("detail", "").lower()

    pdf_upload = session.post(
        _url(f"/api/projects/{pid}/upload"),
        headers=h,
        files={"file": ("bad.pdf", b"%PDF-1.4 fake", "application/pdf")},
        timeout=90,
    )
    assert pdf_upload.status_code == 400
    assert "Unsupported file" in pdf_upload.json().get("detail", "")

    xls_supported = True
    try:
        xls_bytes = io.BytesIO()
        latin_df.to_excel(xls_bytes, index=False, engine="xlwt")
        xls_payload = xls_bytes.getvalue()
    except Exception:
        xls_supported = False
        pytest.skip("xlwt not available for generating .xls test file in this environment")

    if xls_supported:
        xls_upload = session.post(
            _url(f"/api/projects/{pid}/upload"),
            headers=h,
            files={"file": ("legacy.xls", xls_payload, "application/vnd.ms-excel")},
            timeout=90,
        )
        assert xls_upload.status_code == 200, xls_upload.text
        p2 = session.get(_url(f"/api/projects/{pid}/preview"), headers=h, timeout=60)
        assert p2.status_code == 200
        assert p2.json()["filename"].endswith(".xls")

    session.delete(_url(f"/api/projects/{pid}"), headers=h, timeout=30)


def test_generic_dataset_train_score_and_wide_dataset_features(session):
    """Generic non-lead data full train+score and wide dataset configure/clean/features."""
    pid, token = _create_project(session)
    h = _headers(token)

    n = 320
    generic = pd.DataFrame(
        {
            "age": [20 + (i % 45) for i in range(n)],
            "job": ["admin" if i % 4 == 0 else "tech" if i % 4 == 1 else "sales" if i % 4 == 2 else "ops" for i in range(n)],
            "married": ["yes" if i % 3 else "no" for i in range(n)],
            "balance": [None if i % 13 == 0 else float((i * 37) % 5000) for i in range(n)],
            "contact_date": [f"2024-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}" for i in range(n)],
            "notes": ["interested in premium package" if i % 5 == 0 else "requested callback" for i in range(n)],
            "outcome": ["yes" if i % 8 == 0 else "no" for i in range(n)],
        }
    )
    up = session.post(
        _url(f"/api/projects/{pid}/upload"),
        headers=h,
        files={"file": ("bank_style.csv", _csv_bytes(generic), "text/csv")},
        timeout=90,
    )
    assert up.status_code == 200
    cfg = session.post(
        _url(f"/api/projects/{pid}/configure"),
        headers={**h, "Content-Type": "application/json"},
        json={"target": "outcome", "positive": "yes"},
        timeout=60,
    )
    assert cfg.status_code == 200, cfg.text
    c = cfg.json()
    assert c["positives"] >= 30

    cl = session.post(_url(f"/api/projects/{pid}/clean"), headers={**h, "Content-Type": "application/json"}, json={"keep_columns": []}, timeout=60)
    assert _poll_job(session, pid, token, cl.json()["job_id"], timeout_seconds=240)["status"] == "completed"
    fe = session.post(_url(f"/api/projects/{pid}/features"), headers={**h, "Content-Type": "application/json"}, json={"pretrained": True}, timeout=60)
    assert _poll_job(session, pid, token, fe.json()["job_id"], timeout_seconds=300)["status"] == "completed"
    tr = session.post(_url(f"/api/projects/{pid}/train"), headers={**h, "Content-Type": "application/json"}, json={"mode": "quick"}, timeout=60)
    assert _poll_job(session, pid, token, tr.json()["job_id"], timeout_seconds=900)["status"] == "completed"
    sc = session.post(_url(f"/api/projects/{pid}/score"), headers={**h, "Content-Type": "application/json"}, json={"scope": "test"}, timeout=60)
    assert _poll_job(session, pid, token, sc.json()["job_id"], timeout_seconds=240)["status"] == "completed"
    ss = session.get(_url(f"/api/projects/{pid}/score-summary"), headers=h, timeout=60)
    assert ss.status_code == 200
    assert ss.json()["total"] > 0

    wide = generic.copy()
    for i in range(1, 106):
        wide[f"feature_{i}"] = [(i * j) % 11 for j in range(n)]
    up_wide = session.post(
        _url(f"/api/projects/{pid}/upload"),
        headers=h,
        files={"file": ("wide.csv", _csv_bytes(wide), "text/csv")},
        timeout=90,
    )
    assert up_wide.status_code == 200
    cfg_wide = session.post(
        _url(f"/api/projects/{pid}/configure"),
        headers={**h, "Content-Type": "application/json"},
        json={"target": "outcome", "positive": "yes"},
        timeout=60,
    )
    assert cfg_wide.status_code == 200
    c2 = session.post(_url(f"/api/projects/{pid}/clean"), headers={**h, "Content-Type": "application/json"}, json={"keep_columns": []}, timeout=60)
    assert _poll_job(session, pid, token, c2.json()["job_id"], timeout_seconds=240)["status"] == "completed"
    f2 = session.post(_url(f"/api/projects/{pid}/features"), headers={**h, "Content-Type": "application/json"}, json={"pretrained": False}, timeout=60)
    assert _poll_job(session, pid, token, f2.json()["job_id"], timeout_seconds=300)["status"] == "completed"
    feat = session.get(_url(f"/api/projects/{pid}/features"), headers=h, timeout=60)
    assert feat.status_code == 200
    assert feat.json()["count"] > 50

    session.delete(_url(f"/api/projects/{pid}"), headers=h, timeout=30)


def test_cron_cleanup_auth_and_idempotency(session):
    secret = os.environ.get("WEBHOOK_CRON_SECRET")
    if not secret:
        pytest.skip("WEBHOOK_CRON_SECRET missing")

    bad = session.post(_url("/api/cron/cleanup"), headers={"Authorization": "Bearer wrong"}, json={"event": "schedule.triggered", "run_id": "r1"}, timeout=30)
    assert bad.status_code == 401

    malformed = session.post(
        _url("/api/cron/cleanup"),
        headers={"Authorization": f"Bearer {secret}", "X-Webhook-Id": "run-x"},
        json={"event": "wrong.event"},
        timeout=30,
    )
    assert malformed.status_code == 400

    run_id = f"test-run-{int(time.time())}"
    first = session.post(
        _url("/api/cron/cleanup"),
        headers={"Authorization": f"Bearer {secret}", "X-Webhook-Id": run_id},
        json={"event": "schedule.triggered", "run_id": run_id},
        timeout=30,
    )
    assert first.status_code == 200
    assert first.json().get("accepted") is True
    assert first.json().get("duplicate") is False

    second = session.post(
        _url("/api/cron/cleanup"),
        headers={"Authorization": f"Bearer {secret}", "X-Webhook-Id": run_id},
        json={"event": "schedule.triggered", "run_id": run_id},
        timeout=30,
    )
    assert second.status_code == 200
    assert second.json().get("accepted") is True
    assert second.json().get("duplicate") is True
