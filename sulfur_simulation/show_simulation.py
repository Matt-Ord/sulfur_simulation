from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib import animation

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.collections import PathCollection
    from matplotlib.figure import Figure

    from sulfur_simulation.scattering_calculation import SimulationParameters


def animate_particle_positions(
    all_particle_positions: np.ndarray,  # shape: (n_particles, n_timesteps, 2)
    grid_size: int,
    timesteps: np.ndarray,
    lattice_spacing: float = 2.5,
) -> animation.FuncAnimation:
    """Animate particle positions on a lattice with given spacing."""
    all_particle_positions.shape[0]

    # Set up the figure and axis
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-lattice_spacing, grid_size * lattice_spacing)
    ax.set_ylim(-lattice_spacing, grid_size * lattice_spacing)
    ax.set_aspect("equal")

    # Plot initial positions (will be updated in animation)
    particle_scatter: PathCollection = ax.scatter(
        [], [], color="red", s=40, label="Particle", edgecolors="black"
    )

    ax.legend(loc="upper right")
    ax.set_title("Particle Simulation")

    # Update function for animation
    def update(frame: int) -> tuple[PathCollection]:
        current_positions: np.ndarray = all_particle_positions[
            :, frame, :
        ]  # shape: (n_particles, 2)

        # Round positions to nearest grid to identify empties
        # Use tolerance to deal with float precision

        # Update scatter plots
        particle_scatter.set_offsets(current_positions)

        ax.set_title(f"Timestep: {frame}")

        return (particle_scatter,)

    # Create animation and keep reference to it
    return animation.FuncAnimation(
        fig,
        update,
        frames=timesteps,
        interval=5000 / len(timesteps),
        blit=False,
        repeat=True,
    )


def print_timeframe(
    positions: np.ndarray, timestep: int, params: SimulationParameters
) -> str:
    """Print a preview of positions at specified timestep. Works for small grids."""
    dimension = params.lattice_dimension
    max_dimension = 100
    if dimension > max_dimension:
        return "Lattice too large to print"
    grid = positions[timestep].reshape((dimension, dimension))

    green = "\033[92m"
    gray = "\033[0m"

    def colorize(val: bool) -> str:  # noqa: FBT001
        return f"{green}o{gray}" if val else "."

    lines: list[str] = []
    for row in grid:
        colored_row = [colorize(v) for v in row]
        lines.append(" ".join(colored_row))

    return "\n".join(lines)
