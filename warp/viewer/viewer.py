"""Newton-inspired Viewer API for warp 1.0.0-beta.2.

Provides a thin, backend-agnostic abstract base class. Concrete backends
(``ViewerGL``, ``ViewerUSD``) wrap the existing ``wp.sim.render.SimRenderer*``
classes to expose a clean ``begin_frame`` / ``log_state`` API similar to
newton's viewer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

import warp as wp
import warp.sim


class ViewerBase(ABC):
    """Abstract viewer interface."""

    def __init__(self):
        self.model: Optional[warp.sim.Model] = None
        self.time = 0.0
        self._terrain_logged = False

    @abstractmethod
    def begin_frame(self, time: float) -> None:
        ...

    @abstractmethod
    def end_frame(self) -> None:
        ...

    @abstractmethod
    def log_state(self, state: warp.sim.State) -> None:
        """Render the current state's body transforms (and particles etc.)."""

    @abstractmethod
    def log_terrain(self, vertices, indices, name: str = "terrain") -> None:
        """Render a static terrain/heightfield mesh once."""

    def log_contacts(
        self,
        positions,
        normals=None,
        count: Optional[int] = None,
        radius: float = 0.02,
        normal_length: float = 0.1,
    ) -> None:
        """Render contact points (and optional normal arrows).

        Args:
            positions: (N, 3) array-like of contact world positions.
            normals: optional (N, 3) array-like of unit contact normals.
            count: optional number of valid contacts (rest treated as NaN).
            radius: radius of the contact point spheres.
            normal_length: length of each normal arrow.
        """
        positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        if count is not None:
            positions = positions[:count]
        # filter NaN/sentinel rows (sim/render.py emits NAN for invalid contacts)
        valid = np.isfinite(positions).all(axis=1)
        positions = positions[valid]
        if len(positions) == 0:
            return
        self._render_contact_points(positions, radius=radius)
        if normals is not None:
            normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
            if count is not None:
                normals = normals[:count]
            normals = normals[valid]
            if len(normals) == len(positions):
                ends = positions + normals * normal_length
                self._render_contact_normals(positions, ends)

    @abstractmethod
    def _render_contact_points(self, positions: np.ndarray, radius: float) -> None:
        ...

    @abstractmethod
    def _render_contact_normals(self, starts: np.ndarray, ends: np.ndarray) -> None:
        ...

    def is_running(self) -> bool:
        return True

    def save(self) -> None:
        """Backend-specific finalization (commit USD stage, etc.). Non-blocking."""
        pass
