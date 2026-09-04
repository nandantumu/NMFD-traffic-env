from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import NMFDParameters

COLORS = {
    "DPC": "#0072B2",
    "MPPI": "#D55E00",
    "Open gates": "#777777",
}


def _prepare_output(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def plot_total_accumulation(
    trajectories: Mapping[str, np.ndarray],
    dt: float,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Plot mean total accumulation with one-standard-deviation bands."""

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for label, states in trajectories.items():
        total = np.asarray(states).sum(axis=-1)
        time_minutes = np.arange(total.shape[1]) * dt / 60.0
        mean = total.mean(axis=0)
        std = total.std(axis=0)
        color = COLORS.get(label)
        axis.plot(time_minutes, mean, label=label, color=color, linewidth=2.0)
        axis.fill_between(time_minutes, mean - std, mean + std, color=color, alpha=0.16)
    axis.set(xlabel="Time [min]", ylabel="Vehicles", title=title)
    axis.grid(alpha=0.22, linewidth=0.7)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(_prepare_output(output_path), dpi=200)
    plt.close(figure)


def plot_per_region_accumulation(
    trajectories: Mapping[str, np.ndarray],
    params: NMFDParameters,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Plot mean accumulation in each origin region for every controller."""

    regions = params.num_regions
    columns = min(4, regions)
    rows = int(np.ceil(regions / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.2 * columns, 2.7 * rows),
        sharex=True,
        squeeze=False,
    )
    for region, axis in enumerate(axes.flat[:regions]):
        for label, states in trajectories.items():
            reshaped = np.asarray(states).reshape(
                states.shape[0], states.shape[1], regions, regions
            )
            accumulation = reshaped[:, :, region, :].sum(axis=-1)
            time_minutes = np.arange(accumulation.shape[1]) * params.dt / 60.0
            mean = accumulation.mean(axis=0)
            std = accumulation.std(axis=0)
            color = COLORS.get(label)
            axis.plot(time_minutes, mean, label=label, color=color, linewidth=1.5)
            axis.fill_between(
                time_minutes, mean - std, mean + std, color=color, alpha=0.13
            )
        axis.set_title(f"Region {region}")
        axis.grid(alpha=0.18, linewidth=0.6)
    for axis in axes.flat[regions:]:
        axis.set_visible(False)
    for axis in axes[-1, :]:
        if axis.get_visible():
            axis.set_xlabel("Time [min]")
    for axis in axes[:, 0]:
        axis.set_ylabel("Vehicles")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0.07, 1, 0.95))
    figure.savefig(_prepare_output(output_path), dpi=200)
    plt.close(figure)


def plot_mean_controls(
    controls: Mapping[str, np.ndarray],
    params: NMFDParameters,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Compare time- and rollout-averaged perimeter controls."""

    count = len(controls)
    figure, axes = plt.subplots(
        1,
        count,
        figsize=(4.5 * count + 0.5, 4.0),
        squeeze=False,
        layout="constrained",
    )
    valid = np.asarray(params.adjacency, dtype=bool)
    np.fill_diagonal(valid, False)
    image = None
    for axis, (label, values) in zip(axes.flat, controls.items(), strict=True):
        mean = (
            np.asarray(values)
            .mean(axis=(0, 1))
            .reshape(
                params.num_regions,
                params.num_regions,
            )
        )
        shown = np.where(valid, mean, np.nan)
        image = axis.imshow(
            shown,
            cmap="viridis",
            vmin=float(params.u_low),
            vmax=float(params.u_high),
        )
        axis.set(title=label, xlabel="Receiving region", ylabel="Sending region")
        axis.set_xticks(range(params.num_regions))
        axis.set_yticks(range(params.num_regions))
    if image is not None:
        figure.colorbar(
            image,
            ax=axes.ravel().tolist(),
            label="Mean control",
            shrink=0.82,
            pad=0.03,
        )
    figure.suptitle(title)
    figure.savefig(_prepare_output(output_path), dpi=200)
    plt.close(figure)


def plot_network(
    params: NMFDParameters,
    output_path: str | Path,
    *,
    title: str = "Seven-region NMFD network",
) -> None:
    """Render the undirected region graph without a graph-library dependency."""

    regions = params.num_regions
    if regions == 7:
        positions = np.array(
            [
                [-0.5, 0.87],
                [-1.0, 0.0],
                [-0.5, -0.87],
                [0.5, 0.87],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, -0.87],
            ]
        )
    else:
        angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, regions, endpoint=False)
        positions = np.column_stack((np.cos(angles), np.sin(angles)))
    adjacency = np.asarray(params.adjacency, dtype=bool)

    figure, axis = plt.subplots(figsize=(5.2, 5.2))
    for source in range(regions):
        for destination in range(source + 1, regions):
            if adjacency[source, destination]:
                axis.plot(
                    positions[[source, destination], 0],
                    positions[[source, destination], 1],
                    color="#A9A9A9",
                    linewidth=2.0,
                    zorder=1,
                )
    axis.scatter(
        positions[:, 0],
        positions[:, 1],
        s=900,
        color="#E6F2F8",
        edgecolor="#0072B2",
        linewidth=2.0,
        zorder=2,
    )
    for region, (x, y) in enumerate(positions):
        axis.text(x, y, str(region), ha="center", va="center", fontsize=12, zorder=3)
    axis.set_title(title)
    axis.set_aspect("equal")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(_prepare_output(output_path), dpi=200, bbox_inches="tight")
    plt.close(figure)
