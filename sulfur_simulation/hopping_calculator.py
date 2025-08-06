from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, override

import numpy as np

if TYPE_CHECKING:
    from sulfur_simulation.scattering_calculation import SimulationParameters


class HoppingCalculator(ABC):
    """Abstract base class for calculating hopping probabilities."""

    @abstractmethod
    def get_hopping_probabilities(self, positions: np.ndarray) -> np.ndarray:
        """Get hopping probabilities."""


class SquareHoppingCalculator(HoppingCalculator):
    """Class for calculating hopping probabilities in a square lattice."""

    def __init__(self, baserate: float, params: SimulationParameters) -> None:
        self._baserate = baserate
        self._lattice_dimension = params.lattice_dimension
        self._params = params
        self._k_B_T = params.k_b_t

    @override
    def get_hopping_probabilities(self, positions: np.ndarray) -> np.ndarray:
        energies = self._energy_landscape_generator(positions=positions)
        true_locations = np.flatnonzero(positions)
        n = self._lattice_dimension

        rows = true_locations // n
        cols = true_locations % n

        delta = np.array(
            [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 0),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]
        )

        neighbor_rows = (rows[:, None] + delta[:, 0]) % n
        neighbor_cols = (cols[:, None] + delta[:, 1]) % n
        neighbor_indices = neighbor_rows * n + neighbor_cols

        neighbor_energies = energies[neighbor_indices]  # shape (n_particles, 9)
        current_energies = energies[true_locations][:, None]  # shape (n_particles, 1)

        # --- Symmetric rate based on energy difference ---
        beta = 1 / (2 * self._k_B_T)  # assumes k_B_T is stored in the object
        energy_diffs = neighbor_energies - current_energies
        rates = np.exp(-beta * energy_diffs) * self._baserate

        # --- Prevent self-jumps ---
        rates[:, 4] = 0

        return rates

    def _energy_landscape_generator(self, positions: np.ndarray) -> np.ndarray:
        """Generate the energy landscape for the lattice."""
        energies = np.full(
            (self._lattice_dimension**2), 3.2e-19
        )  # temporary, all energies equivalent except for a line of low energy spaces across the middle

        # true_locations = np.flatnonzero(positions)  <-- this causes all particles to get stuck under line, needs further investigation
        # energies[true_locations] += 1

        mid_row_start = ((self._lattice_dimension - 1) // 2) * self._lattice_dimension
        middle_row_indices = np.arange(
            mid_row_start, mid_row_start + self._lattice_dimension
        )
        energies[middle_row_indices] = 0
        return energies
