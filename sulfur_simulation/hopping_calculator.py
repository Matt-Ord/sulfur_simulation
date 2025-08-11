from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, override

import numpy as np
from scipy.constants import Boltzmann  # type: ignore[reportMissingTypeStubs]


class HoppingCalculator(ABC):
    """Abstract base class for calculating hopping probabilities."""

    @abstractmethod
    def get_hopping_probabilities(self, positions: np.ndarray) -> np.ndarray:
        """Get hopping probabilities."""


class SquareHoppingCalculator(HoppingCalculator):
    """Class for calculating hopping probabilities in a square lattice."""

    def __init__(self, baserate: float, temperature: float) -> None:
        self._baserate = baserate
        self._temperature = temperature

    @override
    def get_hopping_probabilities(
        self, positions: np.ndarray[tuple[int, int], np.dtype[np.bool_]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        energies = self._get_energy_landscape(positions=positions)
        rows, cols = np.nonzero(positions)

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

        neighbor_rows = (rows[:, None] + delta[:, 0]) % positions.shape[0]
        neighbor_cols = (cols[:, None] + delta[:, 1]) % positions.shape[1]
        np.ravel_multi_index((neighbor_rows, neighbor_cols), positions.shape)

        neighbor_energies = energies[neighbor_rows, neighbor_cols]
        current_energies = energies[rows, cols][:, None]

        # Calculate the rate based on the boltzmann factor
        # beta = delta_e / (k_B * T)  # noqa: ERA001
        beta = 1 / (2 * Boltzmann * self._temperature)
        energy_difference = neighbor_energies - current_energies
        rates = np.exp(-beta * energy_difference) * self._baserate

        # Prevent self-jumps
        rates[:, 4] = 0

        return rates

    @classmethod
    def _get_energy_landscape(
        cls, positions: np.ndarray[tuple[int, int], np.dtype[np.bool_]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Generate the energy landscape for the lattice."""
        return np.full(positions.shape, 3.2e-19)


class LineDefectHoppingCalculator(SquareHoppingCalculator):
    """Hopping Calculator for a line defect in a square lattice."""

    @classmethod
    @override
    def _get_energy_landscape(
        cls, positions: np.ndarray[tuple[int, int], np.dtype[np.bool_]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Generate the energy landscape for the lattice."""
        energies = super()._get_energy_landscape(positions=positions)
        energies[positions.shape[0] // 2, :] = 0
        return energies


class LennardJonesHoppingCalculator(SquareHoppingCalculator):
    """Hopping Calculator with a Lennard Jones potential between particles."""

    lj_table: ClassVar[dict[tuple[int, int], float]] = {}

    @classmethod
    def initialize_lj_table(
        cls,
        sigma: float,
        epsilon: float,
        cutoff: int,
        lattice_spacing: float,
    ) -> None:
        """Create lookup table for Lennard Jones values."""
        lookup = {}
        for dx in range(-cutoff, cutoff + 1):
            for dy in range(-cutoff, cutoff + 1):
                if dx == 0 and dy == 0:
                    continue
                r = lattice_spacing * np.sqrt(dx**2 + dy**2)
                if r <= cutoff * lattice_spacing:
                    sr6 = (sigma / r) ** 6
                    sr12 = sr6**2
                    lookup[dx, dy] = 4 * epsilon * (sr12 - sr6)
        cls.lj_table = lookup

    @classmethod
    @override
    def _get_energy_landscape(
        cls, positions: np.ndarray[tuple[int, int], np.dtype[np.bool_]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Generate the energy landscape for the lattice."""
        energies = super()._get_energy_landscape(positions)

        n_rows, n_columns = positions.shape
        occupied_coordinates = np.argwhere(positions)

        # Add Lennard-Jones contributions from each occupied site
        for particle_row, particle_col in occupied_coordinates:
            for (delta_row, delta_col), lj_potential in cls.lj_table.items():
                target_row = (particle_row + delta_row) % n_rows
                target_col = (particle_col + delta_col) % n_columns
                energies[target_row, target_col] += lj_potential

        return energies
