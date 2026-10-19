"""
Tests for the Attack / Beast / Warhammer phishing pipeline.
"""

import pytest

from phishing.config import PhishingConfig
from phishing.detector import PhishingDetector
from phishing.indicators import FEATURE_NAMES, IndicatorExtractor
from phishing.train import PhishingTrainer
from util.loaders.phishing import PhishingDatasetLoader

PHISH = "URGENT: your account is suspended! Verify your password at http://192.168.4.20/login now!"
HAM = "Hi team, the project sync moved to Thursday. Agenda attached, see you there."


@pytest.fixture(scope="module")
def splits(tmp_path_factory):
    out = PhishingDatasetLoader.create_sample_dataset(tmp_path_factory.mktemp("phish"), samples_per_class=40)
    loader = PhishingDatasetLoader()
    loader.load_enron_spam(out / "enron_spam")
    return loader.get_splits()


@pytest.fixture(scope="module")
def shallow_trainer(splits):
    """Attack + Beast only — no DistilBERT download/training."""
    trainer = PhishingTrainer(PhishingConfig(use_warhammer=False))
    trainer.fit(*splits["train"], *splits["val"])
    return trainer


def test_indicators_flag_phishing_structure():
    ext = IndicatorExtractor()
    phish, ham = ext.extract(PHISH), ext.extract(HAM)
    assert set(phish) == set(FEATURE_NAMES)
    assert phish["url_count"] == 1
    assert phish["ip_url_ratio"] == 1.0
    assert phish["urgency_density"] > ham["urgency_density"]
    assert phish["credential_density"] > 0
    assert ham["url_count"] == 0
    assert ext.extract_batch([PHISH, HAM]).shape == (2, len(FEATURE_NAMES))


def test_indicators_bare_url_sample():
    feats = IndicatorExtractor().extract("secure-paypa1-login.xyz/verify")
    assert feats["url_count"] == 1
    assert feats["suspicious_tld_ratio"] == 1.0
    assert feats["max_domain_hyphens"] == 2


def test_shallow_pipeline_predicts(shallow_trainer, splits):
    test_texts, test_labels = splits["test"]
    results = shallow_trainer.detector.predict(test_texts)
    assert len(results) == len(test_texts)
    for r in results:
        assert 0.0 <= r["p_phishing"] <= 1.0
        assert r["p_legitimate"] == pytest.approx(1.0 - r["p_phishing"])
        assert r["label"] in ("phishing", "legitimate")
        assert r["stage"] in ("attack", "beast")  # Warhammer disabled
    accuracy = sum(r["label"] == y for r, y in zip(results, test_labels)) / len(test_labels)
    assert accuracy >= 0.9


def test_escalation_bands_route_stages(shallow_trainer):
    detector = shallow_trainer.detector
    cfg = detector.config
    # Force everything past Attack: a (-1, 2) band contains every probability.
    cfg.attack_escalate_low, cfg.attack_escalate_high = -1.0, 2.0
    try:
        assert {r["stage"] for r in detector.predict([PHISH, HAM])} == {"beast"}
        assert {r["stage"] for r in detector.predict([PHISH, HAM], max_stage="attack")} == {"attack"}
    finally:
        cfg.attack_escalate_low, cfg.attack_escalate_high = 0.1, 0.9


def test_predict_rejects_unknown_stage(shallow_trainer):
    with pytest.raises(ValueError):
        shallow_trainer.detector.predict([HAM], max_stage="bazooka")
    assert shallow_trainer.detector.predict([]) == []


def test_shallow_save_load_roundtrip(shallow_trainer, tmp_path):
    shallow_trainer.detector.save(tmp_path)
    assert (tmp_path / "attack.joblib").exists()
    assert (tmp_path / "beast.json").exists()
    assert not (tmp_path / "warhammer.pt").exists()

    restored = PhishingDetector(PhishingConfig(use_warhammer=False))
    restored.load(tmp_path)
    before = shallow_trainer.detector.predict([PHISH, HAM])
    after = restored.predict([PHISH, HAM])
    for a, b in zip(before, after):
        assert a["p_phishing"] == pytest.approx(b["p_phishing"], abs=1e-6)


def test_warhammer_trains_and_takes_escalations(splits, tmp_path):
    config = PhishingConfig(
        max_seq_length=48, num_epochs_frozen=1, num_epochs_finetune=1,
        train_batch_size=16, warmup_steps=0,
        # Escalate everything so Warhammer makes every call.
        attack_escalate_low=-1.0, attack_escalate_high=2.0,
        beast_escalate_low=-1.0, beast_escalate_high=2.0,
    )
    trainer = PhishingTrainer(config)
    events = []
    trainer.callback = events.append
    trainer.fit(*splits["train"], *splits["val"])

    assert trainer.detector.warhammer_ready
    assert any(e["type"] == "epoch" and e["stage"] == "finetune" for e in events)
    assert {r["stage"] for r in trainer.detector.predict([PHISH, HAM])} == {"warhammer"}

    trainer.detector.save(tmp_path)
    restored = PhishingDetector(config)
    assert not restored.warhammer_ready
    restored.load(tmp_path)
    assert restored.warhammer_ready


def test_normalize_phishing_label_aliases():
    from util.loaders.phishing import normalize_phishing_label

    for raw in ("phishing", "Phishing Email", "spam", 1, "1", "malicious"):
        assert normalize_phishing_label(raw) == "phishing"
    for raw in ("legitimate", "Safe Email", "ham", 0, "benign"):
        assert normalize_phishing_label(raw) == "legitimate"
    assert normalize_phishing_label("unknown-thing") is None


def test_load_directory_reads_generic_csv_and_jsonl(tmp_path):
    (tmp_path / "emails.csv").write_text(
        'Email Text,Email Type\n"verify your account",Phishing Email\n"lunch?",Safe Email\n"x",weird\n',
        encoding="utf-8",
    )
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "urls.jsonl").write_text(
        '{"url": "http://bad.example", "label": 1}\n{"url": "https://good.example", "label": 0}\n',
        encoding="utf-8",
    )
    (tmp_path / "raw").mkdir()  # raw downloads are never read as generic files
    (tmp_path / "raw" / "ignored.json").write_text('[{"text": "nope", "label": 1}]', encoding="utf-8")

    loader = PhishingDatasetLoader()
    assert loader.load_directory(tmp_path) == 4
    pairs = dict(zip(loader.text_dataset.data, loader.text_dataset.labels))
    assert pairs == {
        "verify your account": "phishing", "lunch?": "legitimate",
        "http://bad.example": "phishing", "https://good.example": "legitimate",
    }


def test_download_source_normalizes_and_balances(tmp_path, monkeypatch):
    import json

    from util import download

    def fake_fetch(url, dest, progress=None):
        rows = [{"text": f"phish {i}", "label": 1} for i in range(30)]
        rows += [{"text": f"ham {i}", "label": 0} for i in range(10)]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(rows), encoding="utf-8")

    monkeypatch.setattr(download, "_fetch", fake_fetch)
    path, counts = download.download_source("hf_emails", tmp_path, max_samples=20)
    assert counts == {"phishing": 10, "legitimate": 10}
    assert path == tmp_path / "hf_emails.jsonl"

    loader = PhishingDatasetLoader()
    assert loader.load_directory(tmp_path) == 20

    # Cached output is reused without fetching again.
    monkeypatch.setattr(download, "_fetch", lambda *a, **k: pytest.fail("should not re-download"))
    assert download.download_source("hf_emails", tmp_path, max_samples=20)[1] == counts


def test_resolve_sources():
    from util.download import DEFAULT_SOURCES, SOURCES, resolve_sources

    assert resolve_sources(None) == DEFAULT_SOURCES
    assert resolve_sources([]) == DEFAULT_SOURCES
    assert resolve_sources(["all"]) == list(SOURCES)
    assert resolve_sources(["openphish,tranco", "openphish"]) == ["openphish", "tranco"]
    with pytest.raises(ValueError):
        resolve_sources(["nope"])
