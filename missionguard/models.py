"""Core data structures for MissionGuard."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

MU_EARTH = 398600.4418  # km^3 / s^2
R_EARTH = 6378.137      # km


@dataclass
class Window:
    """A time window in hours relative to scenario epoch."""
    start: float
    end: float

    def contains(self, t: float) -> bool:
        return self.start <= t <= self.end

    def overlaps(self, a: float, b: float) -> bool:
        return self.start < b and a < self.end


@dataclass
class Activity:
    name: str
    window: Window
    priority: int  # 1 = low, 2 = medium, 3 = critical


@dataclass
class Health:
    propulsion: float  # 0 (failed) .. 1 (nominal)
    power: float
    adcs: float        # attitude determination & control
    comms: float


@dataclass
class Conjunction:
    """Conjunction geometry in the encounter (B-) plane."""
    object_id: str
    tca: float                     # hours from epoch
    miss_rtn: np.ndarray           # km, relative position at TCA in RTN frame
    cov_b: np.ndarray              # km^2, 2x2 combined covariance in B-plane
    hbr: float                     # combined hard-body radius, km
    rel_speed: float               # km/s
    theta: float                   # encounter angle in T-N plane (rad)


@dataclass
class Scenario:
    scenario_id: int
    sat_name: str
    altitude_km: float
    epoch_label: str
    primary: Conjunction
    secondaries: List[Conjunction]
    comm_windows: List[Window]
    activities: List[Activity]
    health: Health
    fuel_available: float          # m/s of delta-v remaining
    fuel_reserve: float            # m/s that must stay unused
    cdm_age_h: float               # age of the latest conjunction data message

    @property
    def mean_motion(self) -> float:
        a = R_EARTH + self.altitude_km
        return float(np.sqrt(MU_EARTH / a ** 3))  # rad/s

    @property
    def period_h(self) -> float:
        return 2 * np.pi / self.mean_motion / 3600.0


@dataclass
class Candidate:
    """A candidate collision-avoidance action."""
    kind: str                        # no-action | in-track | radial | cross-track
    dv: np.ndarray                   # m/s impulse in RTN
    burn_time: Optional[float]       # hours from epoch (None for no-action)
    label: str = ""
    # filled in by evaluation
    pc: float = 0.0
    miss_km: float = 0.0
    secondary_pcs: List[float] = field(default_factory=list)
    objectives: dict = field(default_factory=dict)
    feasible: bool = True
    violations: List[str] = field(default_factory=list)
    utility: float = float("nan")
    shap: dict = field(default_factory=dict)
    confidence: float = float("nan")
    confidence_parts: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def dv_mag(self) -> float:
        return float(np.linalg.norm(self.dv))
