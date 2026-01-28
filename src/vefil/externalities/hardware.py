"""Hardware depreciation mismatch - 5-year locks vs 3-year hardware cycles."""

from dataclasses import dataclass
from typing import List


@dataclass
class HardwareState:
    """Hardware state for an SP."""
    lock_duration_years: float
    hardware_depreciation_years: float
    time_into_lock_years: float
    hardware_age_years: float


class HardwareMismatchAnalyzer:
    """Analyze hardware depreciation mismatch risks."""

    def __init__(self, hardware_depreciation_years: float = 3.0):
        """
        Initialize hardware mismatch analyzer.
        
        Args:
            hardware_depreciation_years: Typical hardware depreciation period
        """
        self.hardware_depreciation_years = hardware_depreciation_years

    def compute_mismatch_risk(
        self,
        lock_duration_years: float,
        hardware_age_at_lock: float = 0.0
    ) -> dict:
        """
        Compute mismatch risk for a given lock duration.
        
        Args:
            lock_duration_years: Lock duration
            hardware_age_at_lock: Hardware age when locked (years)
            
        Returns:
            Mismatch risk analysis
        """
        # Hardware becomes obsolete at depreciation period
        hardware_obsolete_time = self.hardware_depreciation_years - hardware_age_at_lock

        # Lock extends beyond hardware lifecycle
        mismatch_exists = lock_duration_years > hardware_obsolete_time
        mismatch_years = max(0.0, lock_duration_years - hardware_obsolete_time)

        # Risk level
        if mismatch_years <= 0:
            risk_level = "none"
        elif mismatch_years < 1.0:
            risk_level = "low"
        elif mismatch_years < 2.0:
            risk_level = "medium"
        else:
            risk_level = "high"

        # Zombie SP: hardware obsolete but still locked
        is_zombie_risk = mismatch_exists and lock_duration_years > hardware_obsolete_time

        return {
            'lock_duration': lock_duration_years,
            'hardware_depreciation': self.hardware_depreciation_years,
            'hardware_obsolete_time': hardware_obsolete_time,
            'mismatch_exists': mismatch_exists,
            'mismatch_years': mismatch_years,
            'risk_level': risk_level,
            'is_zombie_risk': is_zombie_risk,
            'recommended_max_lock': min(lock_duration_years, hardware_obsolete_time)
        }

    def analyze_sp_portfolio(
        self,
        lock_durations: List[float],
        hardware_ages: List[float] = None
    ) -> dict:
        """
        Analyze mismatch risk for a portfolio of SP locks.
        
        Args:
            lock_durations: List of lock durations
            hardware_ages: List of hardware ages at lock (defaults to 0 for all)
            
        Returns:
            Portfolio analysis
        """
        if hardware_ages is None:
            hardware_ages = [0.0] * len(lock_durations)

        if len(hardware_ages) != len(lock_durations):
            hardware_ages = [0.0] * len(lock_durations)

        mismatch_risks = [
            self.compute_mismatch_risk(duration, age)
            for duration, age in zip(lock_durations, hardware_ages)
        ]

        # Aggregate statistics
        total_locks = len(lock_durations)
        zombie_risks = sum(1 for r in mismatch_risks if r['is_zombie_risk'])
        zombie_percentage = (zombie_risks / total_locks * 100) if total_locks > 0 else 0.0

        risk_distribution = {
            'none': sum(1 for r in mismatch_risks if r['risk_level'] == 'none'),
            'low': sum(1 for r in mismatch_risks if r['risk_level'] == 'low'),
            'medium': sum(1 for r in mismatch_risks if r['risk_level'] == 'medium'),
            'high': sum(1 for r in mismatch_risks if r['risk_level'] == 'high')
        }

        avg_mismatch_years = sum(r['mismatch_years'] for r in mismatch_risks) / total_locks if total_locks > 0 else 0.0

        return {
            'total_locks': total_locks,
            'zombie_risks': zombie_risks,
            'zombie_percentage': zombie_percentage,
            'risk_distribution': risk_distribution,
            'avg_mismatch_years': avg_mismatch_years,
            'individual_risks': mismatch_risks,
            'recommendation': self._generate_recommendation(zombie_percentage, avg_mismatch_years)
        }

    def _generate_recommendation(
        self,
        zombie_percentage: float,
        avg_mismatch_years: float
    ) -> str:
        """Generate recommendation based on analysis."""
        if zombie_percentage > 30 or avg_mismatch_years > 2.0:
            return "High risk: Consider reducing max lock duration to match hardware cycle"
        elif zombie_percentage > 10 or avg_mismatch_years > 1.0:
            return "Medium risk: Monitor hardware depreciation vs lock duration"
        else:
            return "Low risk: Current lock durations align reasonably with hardware cycles"
