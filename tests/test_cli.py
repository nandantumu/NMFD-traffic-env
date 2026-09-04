import json
from contextlib import ExitStack
from pathlib import Path

import pytest

from nmfd_traffic.cli import (
    DEFAULT_CONFIG_RESOURCE,
    _evaluation_parser,
    _resolve_asset,
    _training_parser,
    evaluate_main,
    train_main,
)

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/seven_region.toml"


def _smoke_config(tmp_path: Path, scenario_name: str = "nominal") -> Path:
    config_text = CONFIG.read_text(encoding="utf-8")
    replacements = {
        "horizon = 240": "horizon = 3",
        "hidden_dim = 128": "hidden_dim = 8",
        "num_hidden_layers = 3": "num_hidden_layers = 2",
        "[scenarios.nominal]": f"[scenarios.{scenario_name}]",
    }
    for original, replacement in replacements.items():
        config_text = config_text.replace(original, replacement, 1)
    path = tmp_path / "smoke.toml"
    path.write_text(config_text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "arguments",
    [
        ["--rollouts", "0"],
        ["--mppi-samples", "-1"],
        ["--mppi-iterations", "0"],
        ["--mppi-temperature", "0"],
        ["--mppi-noise-std", "nan"],
    ],
)
def test_evaluation_parser_rejects_invalid_positive_values(arguments):
    with pytest.raises(SystemExit):
        _evaluation_parser().parse_args(["--checkpoint", "unused.msgpack", *arguments])


def test_evaluation_parser_requires_an_explicit_checkpoint():
    with pytest.raises(SystemExit):
        _evaluation_parser().parse_args([])


def test_training_parser_accepts_custom_scenario_and_rejects_zero_epochs():
    parsed = _training_parser().parse_args(["--scenario", "custom"])
    assert parsed.scenario == "custom"

    with pytest.raises(SystemExit):
        _training_parser().parse_args(["--epochs", "0"])


def test_packaged_configuration_resolves_outside_the_repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with ExitStack() as stack:
        path, reference = _resolve_asset(stack, None, DEFAULT_CONFIG_RESOURCE)
        assert path.is_file()
        assert reference == "package:nmfd_traffic/data/seven_region.toml"


def test_training_then_evaluation_records_complete_provenance(tmp_path):
    config = _smoke_config(tmp_path)
    checkpoint = tmp_path / "trained.msgpack"
    output = tmp_path / "results"

    train_main(
        [
            "--config",
            str(config),
            "--output",
            str(checkpoint),
            "--epochs",
            "1",
            "--steps-per-epoch",
            "1",
            "--batch-size",
            "2",
            "--seed",
            "3",
        ]
    )
    evaluate_main(
        [
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--rollouts",
            "1",
            "--mppi-samples",
            "1",
            "--seed",
            "11",
            "--output-dir",
            str(output),
            "--no-plots",
        ]
    )

    checkpoint_metadata = json.loads(checkpoint.with_suffix(".json").read_text())
    assert checkpoint.stat().st_size > 0
    assert checkpoint_metadata["seed"] == 3
    assert checkpoint_metadata["training"]["epochs"] == 1
    assert checkpoint_metadata["training"]["completed_updates"] == 1
    assert checkpoint_metadata["policy"]["controller"] == "perimeter control only"
    assert checkpoint_metadata["objective"]["name"] == "L1 total vehicle time"
    assert checkpoint_metadata["environment"]["acyclic_plant_model_used"] is False
    assert len(checkpoint_metadata["artifact"]["sha256"]) == 64
    assert checkpoint_metadata["artifact"]["size_bytes"] == checkpoint.stat().st_size

    report = json.loads((output / "nominal_metrics.json").read_text())
    reproducibility = report["reproducibility"]
    assert reproducibility["seed"] == 11
    assert reproducibility["config"]["source"] == str(config.resolve())
    assert reproducibility["checkpoint"]["source"] == str(checkpoint.resolve())
    assert len(reproducibility["config"]["sha256"]) == 64
    assert len(reproducibility["checkpoint"]["sha256"]) == 64
    assert len(reproducibility["checkpoint"]["provenance"]["sha256"]) == 64
    assert reproducibility["software"]["jax"]
    assert reproducibility["generated_at_utc"]
    assert report["objective"]["shared_by"] == ["dpc", "mppi"]
    assert report["dpc_settings"]["policy"]["hidden_dim"] == 8
    assert report["mppi_settings"]["doi"] == "10.1109/TRO.2018.2865891"
    assert report["experiment"]["acyclic_plant_model_used"] is False
    assert "mppi" in report["controllers"]


def test_evaluation_cli_accepts_a_scenario_defined_only_by_custom_config(tmp_path):
    config = _smoke_config(tmp_path, scenario_name="educational_demo")
    checkpoint = tmp_path / "trained.msgpack"
    output = tmp_path / "custom-results"
    train_main(
        [
            "--config",
            str(config),
            "--output",
            str(checkpoint),
            "--scenario",
            "educational_demo",
            "--epochs",
            "1",
            "--steps-per-epoch",
            "1",
            "--batch-size",
            "1",
        ]
    )

    evaluate_main(
        [
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--scenario",
            "educational_demo",
            "--rollouts",
            "1",
            "--mppi-samples",
            "1",
            "--output-dir",
            str(output),
            "--no-plots",
        ]
    )

    report = json.loads((output / "educational_demo_metrics.json").read_text())
    assert report["scenario"] == "educational_demo"
