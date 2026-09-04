import json
from pathlib import Path

import pytest

from nmfd_traffic.cli import (
    _evaluation_parser,
    _training_parser,
    evaluate_main,
    train_main,
)

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/seven_region.toml"
CHECKPOINT = ROOT / "checkpoints/dpc_policy.msgpack"


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
        _evaluation_parser().parse_args(arguments)


def test_training_parser_accepts_custom_scenario_and_rejects_zero_epochs():
    parsed = _training_parser().parse_args(["--scenario", "custom"])
    assert parsed.scenario == "custom"

    with pytest.raises(SystemExit):
        _training_parser().parse_args(["--epochs", "0"])


def test_evaluation_cli_uses_packaged_defaults_outside_the_repository(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "results"

    evaluate_main(
        [
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

    report = json.loads((output / "in_distribution_metrics.json").read_text())
    metadata = report["reproducibility"]
    assert metadata["seed"] == 11
    assert metadata["config"]["source"] == (
        "package:nmfd_traffic/data/seven_region.toml"
    )
    assert metadata["checkpoint"]["source"] == (
        "package:nmfd_traffic/data/dpc_policy.msgpack"
    )
    assert len(metadata["config"]["sha256"]) == 64
    assert len(metadata["checkpoint"]["sha256"]) == 64
    assert metadata["software"]["nmfd_traffic_env"] == "0.1.0"
    assert metadata["software"]["jax"]
    assert metadata["generated_at_utc"]
    assert report["mppi_settings"]["doi"] == "10.1109/TRO.2018.2865891"
    assert report["mppi_settings"]["uses_nmfd_objective_extensions"] is False
    assert "mppi" in report["controllers"]


def test_evaluation_cli_accepts_a_scenario_defined_only_by_custom_config(tmp_path):
    config_text = CONFIG.read_text(encoding="utf-8").replace(
        "scenarios.in_distribution",
        "scenarios.educational_demo",
    )
    custom_config = tmp_path / "custom.toml"
    custom_config.write_text(config_text, encoding="utf-8")
    output = tmp_path / "custom-results"

    evaluate_main(
        [
            "--config",
            str(custom_config),
            "--checkpoint",
            str(CHECKPOINT),
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


def test_training_cli_smoke_run_writes_parameters_and_provenance(tmp_path):
    config_text = CONFIG.read_text(encoding="utf-8")
    replacements = {
        "horizon = 40": "horizon = 1",
        "hidden_dim = 512": "hidden_dim = 8",
        "num_hidden_layers = 5": "num_hidden_layers = 1",
    }
    for original, replacement in replacements.items():
        config_text = config_text.replace(original, replacement)
    smoke_config = tmp_path / "training-smoke.toml"
    smoke_config.write_text(config_text, encoding="utf-8")
    output = tmp_path / "trained.msgpack"

    train_main(
        [
            "--config",
            str(smoke_config),
            "--output",
            str(output),
            "--epochs",
            "1",
            "--steps-per-epoch",
            "1",
            "--sample-pool-size",
            "4",
            "--batch-size",
            "2",
            "--seed",
            "3",
        ]
    )

    metadata = json.loads(output.with_suffix(".json").read_text())
    assert output.stat().st_size > 0
    assert metadata["seed"] == 3
    assert metadata["epochs"] == 1
    assert metadata["steps_per_epoch"] == 1
    assert metadata["source_config_sha256"]
    assert metadata["nmfd_traffic_env_version"] == "0.1.0"
