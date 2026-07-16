from fakebook.config import Config


def test_default_config_sections():
    c = Config.default()
    for section in ("ingest", "separation", "mir", "harmony", "timbre", "interpret", "assemble"):
        assert isinstance(c.section(section), dict)


def test_dotted_get_and_default():
    c = Config.default()
    assert c.get("harmony.max_chord_interpretations") == 5
    assert c.get("nope.nope", "fallback") == "fallback"


def test_env_override(monkeypatch):
    monkeypatch.setenv("FAKEBOOK_HARMONY__SALIENCE_THRESHOLD", "0.42")
    monkeypatch.setenv("FAKEBOOK_TIMBRE__ENABLED", "true")
    c = Config.load()
    assert c.get("harmony.salience_threshold") == 0.42
    assert c.get("timbre.enabled") is True


def test_user_yaml_override(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("harmony:\n  max_scale_candidates: 9\n")
    c = Config.load(p)
    assert c.get("harmony.max_scale_candidates") == 9
    # untouched keys remain
    assert c.get("harmony.max_chord_interpretations") == 5
