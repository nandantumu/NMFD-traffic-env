from pathlib import Path

import numpy as np
from matplotlib import image as mpl_image

from nmfd_traffic import load_config
from nmfd_traffic.visualization import (
    plot_mean_controls,
    plot_network,
    plot_per_region_accumulation,
    plot_total_accumulation,
)

CONFIG = Path(__file__).parents[1] / "configs" / "seven_region.toml"


def test_visualization_outputs(tmp_path):
    params = load_config(CONFIG).environment
    rng = np.random.default_rng(0)
    trajectories = {
        "DPC": rng.uniform(0.0, 100.0, (3, 5, params.state_dim)),
        "MPPI": rng.uniform(0.0, 100.0, (3, 5, params.state_dim)),
        "Open gates": rng.uniform(0.0, 100.0, (3, 5, params.state_dim)),
    }
    controls = {
        "DPC": rng.uniform(params.u_low, params.u_high, (3, 4, params.control_dim)),
        "MPPI": rng.uniform(
            params.u_low,
            params.u_high,
            (3, 4, params.control_dim),
        ),
    }
    outputs = [
        tmp_path / "total.png",
        tmp_path / "regions.png",
        tmp_path / "controls.png",
        tmp_path / "network.png",
    ]
    plot_total_accumulation(trajectories, params.dt, outputs[0], title="Total")
    plot_per_region_accumulation(trajectories, params, outputs[1], title="Regions")
    plot_mean_controls(controls, params, outputs[2], title="Controls")
    plot_network(params, outputs[3])
    assert all(path.stat().st_size > 1_000 for path in outputs)

    for output in outputs:
        image = mpl_image.imread(output)
        assert image.ndim == 3
        assert min(image.shape[:2]) >= 500
        assert float(np.ptp(image)) > 0.5
        assert np.unique(image.reshape(-1, image.shape[-1]), axis=0).shape[0] > 20
