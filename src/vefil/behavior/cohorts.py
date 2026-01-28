"""User cohort definitions for behavioral modeling.

Key Concepts:
- Each cohort has an addressable_fil representing total FIL that could potentially lock
- Cohort state tracks current locked amount to enforce bounded adoption
- Partial adjustment model: new_locks = adjustment_speed * gap_to_desired
"""

from dataclasses import dataclass, field
from typing import Literal, List


@dataclass
class CohortState:
    """Mutable state for a cohort's lock position."""
    locked_fil: float = 0.0  # Currently locked by this cohort
    eligible_fil: float = 0.0  # Remaining FIL that can be locked
    chosen_duration: float = 0.0  # Current chosen duration
    participation_rate: float = 0.0  # Current participation rate


@dataclass
class Cohort:
    """User cohort definition with addressable FIL bounds."""
    name: str
    size_fraction: float  # Fraction of total user base (for weighting)
    required_premium: float  # Required yield premium vs alternatives (as fraction)
    avg_lock_size: float  # Average lock size in FIL (for per-user sizing)
    avg_duration_years: float  # Average preferred lock duration
    risk_tolerance: Literal["low", "medium", "high"] = "medium"
    # Addressable FIL: explicit bound on how much this cohort can lock
    # If None, computed as fraction of circulating supply
    addressable_fil: float = None
    addressable_fraction: float = 0.0  # Alternative: fraction of circulating
    # Allowed duration set for discrete optimization
    allowed_durations: List[float] = field(default_factory=lambda: [0.25, 0.5, 1, 2, 3, 4, 5])
    # Cohort state (mutable)
    state: CohortState = field(default_factory=CohortState)

    def compute_required_yield(self, baseline_yield: float) -> float:
        """
        Compute required yield for this cohort.

        Args:
            baseline_yield: Baseline yield (e.g., from alternatives)

        Returns:
            Required yield (baseline + premium)
        """
        return baseline_yield + self.required_premium

    def compute_addressable(self, circulating: float) -> float:
        """
        Compute addressable FIL for this cohort.

        Args:
            circulating: Current circulating supply

        Returns:
            Addressable FIL (absolute amount)
        """
        if self.addressable_fil is not None:
            return self.addressable_fil
        return circulating * self.addressable_fraction

    def compute_eligible(self, circulating: float) -> float:
        """
        Compute eligible FIL (addressable minus already locked).

        Args:
            circulating: Current circulating supply

        Returns:
            Eligible FIL that can still be locked
        """
        addressable = self.compute_addressable(circulating)
        return max(0.0, addressable - self.state.locked_fil)

    def update_state(
        self,
        new_locks: float,
        unlocks: float,
        chosen_duration: float,
        participation: float,
        circulating: float
    ):
        """Update cohort state after a timestep."""
        self.state.locked_fil = max(0.0, self.state.locked_fil + new_locks - unlocks)
        self.state.chosen_duration = chosen_duration
        self.state.participation_rate = participation
        self.state.eligible_fil = self.compute_eligible(circulating)


# Predefined cohort templates
COHORT_TEMPLATES = {
    "retail": Cohort(
        name="retail",
        size_fraction=0.5,
        required_premium=0.08,
        avg_lock_size=1000.0,
        avg_duration_years=2.0,
        risk_tolerance="medium"
    ),
    "institutional": Cohort(
        name="institutional",
        size_fraction=0.2,
        required_premium=0.20,
        avg_lock_size=50000.0,
        avg_duration_years=4.0,
        risk_tolerance="low"
    ),
    "storage_providers": Cohort(
        name="storage_providers",
        size_fraction=0.2,
        required_premium=0.12,
        avg_lock_size=20000.0,
        avg_duration_years=3.0,
        risk_tolerance="medium"
    ),
    "treasuries": Cohort(
        name="treasuries",
        size_fraction=0.1,
        required_premium=0.06,
        avg_lock_size=100000.0,
        avg_duration_years=5.0,
        risk_tolerance="low"
    )
}
