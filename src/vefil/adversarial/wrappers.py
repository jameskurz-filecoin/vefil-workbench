"""Liquid wrapper impact analysis - Do wrappers break the commitment premise?"""

from dataclasses import dataclass
from typing import List


@dataclass
class WrapperState:
    """Liquid wrapper state."""
    total_locked: float  # Total FIL locked by wrapper
    concentration_ratio: float  # Fraction of total veFIL locked
    avg_lock_duration: float  # Average lock duration (should be max)
    cvxfil_supply: float  # Supply of liquid wrapper token


class WrapperAnalyzer:
    """Analyze impact of liquid wrappers (convexFIL-style derivatives)."""

    def __init__(self, max_duration_years: float = 5.0):
        """
        Initialize wrapper analyzer.
        
        Args:
            max_duration_years: Maximum lock duration
        """
        self.max_duration_years = max_duration_years

    def compute_wrapper_state(
        self,
        wrapper_locked: float,
        total_vefil_locked: float,
        cvxfil_supply: float = None
    ) -> WrapperState:
        """
        Compute wrapper state metrics.
        
        Args:
            wrapper_locked: FIL locked by wrapper
            total_vefil_locked: Total veFIL locked
            cvxfil_supply: Supply of wrapper token (optional)
            
        Returns:
            Wrapper state
        """
        if total_vefil_locked <= 0:
            concentration_ratio = 0.0
        else:
            concentration_ratio = wrapper_locked / total_vefil_locked

        # Wrappers typically lock at max duration
        avg_lock_duration = self.max_duration_years

        if cvxfil_supply is None:
            # Assume 1:1 ratio for simplicity
            cvxfil_supply = wrapper_locked

        return WrapperState(
            total_locked=wrapper_locked,
            concentration_ratio=concentration_ratio,
            avg_lock_duration=avg_lock_duration,
            cvxfil_supply=cvxfil_supply
        )

    def analyze_wrapper_impact(
        self,
        wrapper_locked: float,
        total_vefil_locked: float,
        individual_lock_distribution: List[float] = None
    ) -> dict:
        """
        Analyze impact of wrapper concentration.
        
        Args:
            wrapper_locked: FIL locked by wrapper
            total_vefil_locked: Total veFIL locked
            individual_lock_distribution: Distribution of individual lock durations (optional)
            
        Returns:
            Impact analysis dictionary
        """
        state = self.compute_wrapper_state(wrapper_locked, total_vefil_locked)

        # Concentration risk
        concentration_risk = "high" if state.concentration_ratio > 0.5 else "medium" if state.concentration_ratio > 0.25 else "low"

        # Duration skew: all wrapper FIL at max duration
        duration_skew = self.max_duration_years  # All at max

        # Effective lock duration: if users can exit via cvxFIL, effective duration drops
        # Simplified: assume wrapper enables ~50% effective duration reduction for users
        effective_duration = self.max_duration_years * 0.5  # Rough estimate

        # Leverage risk: cvxFIL can be re-hypothecated in DeFi
        leverage_multiplier = 1.5  # Estimate: users can leverage 1.5x

        return {
            'wrapper_state': state,
            'concentration_risk': concentration_risk,
            'duration_skew': duration_skew,
            'effective_lock_duration': effective_duration,
            'commitment_premise_broken': state.concentration_ratio > 0.5,
            'leverage_multiplier': leverage_multiplier,
            'voting_power_concentration': state.concentration_ratio,
            'risks': self._identify_risks(state)
        }

    def _identify_risks(self, state: WrapperState) -> List[str]:
        """Identify risks from wrapper state."""
        risks = []

        if state.concentration_ratio > 0.5:
            risks.append("Wrapper controls majority of veFIL voting power")

        if state.concentration_ratio > 0.25:
            risks.append("Significant concentration risk")

        risks.append("Effective lock duration reduced (users can exit via wrapper token)")
        risks.append("Leverage risk from re-hypothecation in DeFi")

        return risks

    def simulate_wrapper_scenario(
        self,
        initial_total_locked: float,
        wrapper_concentration_target: float
    ) -> dict:
        """
        Simulate a wrapper concentration scenario.
        
        Args:
            initial_total_locked: Initial total veFIL locked
            wrapper_concentration_target: Target wrapper concentration (e.g., 0.5 = 50%)
            
        Returns:
            Scenario simulation results
        """
        wrapper_locked = initial_total_locked * wrapper_concentration_target
        impact = self.analyze_wrapper_impact(wrapper_locked, initial_total_locked)

        return {
            'wrapper_locked': wrapper_locked,
            'individual_locked': initial_total_locked - wrapper_locked,
            'concentration': wrapper_concentration_target,
            'impact': impact
        }
