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

    @override
    def get_hopping_probabilities(self, positions: np.ndarray) -> np.ndarray:
        # generate energy landscape
        # calculate probabilities from energy landscape
        # returns probabilities

        # placeholder uniform probability
        probabilities = np.full((self._params.n_particles, 9), self._baserate / 8)
        probabilities[:, 4] = 0

        return probabilities

    def _energy_landscape_generator(self, current_positions: np.ndarray) -> np.ndarray:
        """Generate the energy landscape for the lattice."""
        energies = np.full(
            (self._lattice_dimension**2), 1
        )  # temporary, all energies are equivalent
        for i in range(len(current_positions)):  # stops particles overlapping
            energies[int(current_positions[i][0])][int(current_positions[i][1])] = 2
        return energies
