from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from hopping_calculator import HoppingCalculator
    from numpy.random import Generator


def _update_positions(
    particle_positions: np.ndarray,
    jump_probabilities: np.ndarray,
    rng: Generator,
    params: SimulationParameters,
) -> np.ndarray:
    n_particles = params.n_particles
    true_locations = np.flatnonzero(particle_positions)

    for particle_index in range(n_particles):
        cumulative_probs = np.cumsum(jump_probabilities[particle_index])
        rand_val = rng.random()

        for i, threshold in enumerate(cumulative_probs):
            if rand_val < threshold:
                particle_positions = _make_jump(
                    i=i,
                    particle_positions=particle_positions,
                    params=params,
                    current_particle_position=true_locations[particle_index],
                )
                break

    return particle_positions


def _make_jump(
    current_particle_position: int,
    params: SimulationParameters,
    i: int,
    particle_positions: np.ndarray,
) -> np.ndarray:
    dimension = params.lattice_dimension

    relative_destinations = [
        -1 * dimension - 1,
        -1 * dimension,
        -1 * dimension + 1,
        -1,
        0,
        1,
        dimension - 1,
        dimension,
        dimension + 1,
    ]
    up_right = 2
    down_right = 8
    down_left = 6

    if (
        current_particle_position == dimension - 1 and i == up_right
    ):  # if in top right corner + if going up + right
        new_destination = dimension * (dimension - 1)
    elif (
        current_particle_position == dimension**2 - 1 and i == down_right
    ):  # if in bottom right corner + if going down + right
        new_destination = 0
    elif current_particle_position == 0 and i == 0:  # if in top left corner
        new_destination = dimension**2 - 1
    elif (
        current_particle_position == dimension * (dimension - 1) and i == down_left
    ):  # if in bottom left corner
        new_destination = dimension - 1
    elif (current_particle_position + 1) % dimension == 0 and i in {
        2,
        5,
        8,
    }:  # if on right side
        new_destination = (
            current_particle_position + relative_destinations[i] - dimension
        )
    elif current_particle_position % dimension == 0 and i in {
        0,
        3,
        6,
    }:  # if on left side
        new_destination = (
            current_particle_position + relative_destinations[i] + dimension
        )
    else:
        new_destination = current_particle_position + relative_destinations[i]

    if new_destination < 0:  # now checking if top/bottom rows went out of bounds
        new_destination += dimension**2
    elif new_destination > dimension**2 - 1:
        new_destination -= dimension**2

    if particle_positions[new_destination]:  # if destination is full, don't do anything
        return particle_positions

    particle_positions[new_destination] = True
    # otherwise, update destination to True and update current location to false
    particle_positions[current_particle_position] = False
    return particle_positions


def run_simulation(
    params: SimulationParameters,
    hop_params: HoppingCalculator,
) -> np.ndarray:
    """Run the simulation."""
    rng = np.random.default_rng(seed=params.rng_seed)
    all_particle_positions = np.empty(
        (params.n_timesteps, params.lattice_dimension**2), dtype=bool
    )  # create empty array

    all_particle_positions[0] = params.initial_positions  # insert starting positions

    for i in range(1, params.n_timesteps):
        jump_probabilities = hop_params.get_hopping_probabilities(
            all_particle_positions[i - 1]
        )  # generate probabilities

        all_particle_positions[i] = _update_positions(
            particle_positions=all_particle_positions[i - 1],
            jump_probabilities=jump_probabilities,
            params=params,
            rng=rng,
        )  # jump forward one step

    return all_particle_positions


@dataclass(kw_only=True, frozen=True)
class SimulationParameters:
    """Parameters for simulating diffusion."""

    n_timesteps: int
    """Number of timesteps"""
    lattice_dimension: int
    "Dimension of lattice"
    lattice_type: Literal["square", "hex"]
    "Type of lattice"
    lattice_spacing: float = 2.5
    "Spacing of lattice in Angstroms"
    hopping_probability: float = 0.01
    """The probability of hopping to a new position at each step."""
    n_particles: int
    """The number of particles"""
    rng_seed: int
    """rng seed for reproducibility"""

    @property
    def times(self) -> np.ndarray:
        """Times for simulation."""
        return np.arange(0, self.n_timesteps)

    @property
    def initial_positions(self) -> np.ndarray:
        """
        Initial particle positions.

        Raises
        ------
        ValueError
            If the number of particles exceeds the number of lattice spaces.
        """
        if self.n_particles > self.lattice_dimension * self.lattice_dimension:
            msg = "More particles than lattice spaces"
            raise ValueError(msg)

        rng = np.random.default_rng(seed=self.rng_seed)
        initial_positions = np.zeros(self.lattice_dimension**2, dtype=bool)
        initial_positions[: self.n_particles] = True
        return rng.permutation(initial_positions)
