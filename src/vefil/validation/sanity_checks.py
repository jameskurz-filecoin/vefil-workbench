"""Sanity checks and validation for simulation inputs and outputs."""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config.schema import Config
from ..engine.accounting import SystemState




@dataclass
class ValidationWarning:
    """A validation warning with severity and message."""
    severity: str  # "warning" or "error"
    category: str  # e.g., "input", "conservation", "bounds"
    message: str
    details: Optional[str] = None


class SanityChecker:
    """Run sanity checks on configuration and simulation state."""

    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config

    def check_config_inputs(self) -> List[ValidationWarning]:
        """
        Check configuration inputs for implausible values.

        Returns:
            List of validation warnings
        """
        warnings = []

        # Check initial supply conservation
        initial_sum = (
            self.config.initial_supply.circulating +
            self.config.initial_supply.reserve +
            self.config.initial_supply.other_allocations +
            self.config.initial_supply.lending_pool +
            self.config.initial_supply.sp_collateral +
            self.config.initial_supply.locked_vefil
        )
        total = self.config.initial_supply.total
        if abs(initial_sum - total) > 1e-6:
            diff = initial_sum - total
            warnings.append(ValidationWarning(
                severity="error",
                category="conservation",
                message=f"Initial supply doesn't sum to total: {initial_sum:,.0f} vs {total:,.0f}",
                details=f"Difference: {diff:+,.0f} FIL"
            ))

        # Check reserve is positive
        if self.config.initial_supply.reserve <= 0:
            warnings.append(ValidationWarning(
                severity="error",
                category="bounds",
                message="Mining reserve must be positive for reserve-based emissions",
                details=f"Current value: {self.config.initial_supply.reserve:,.0f} FIL"
            ))

        # Check emission rate bounds
        if self.config.yield_source.reserve_annual_rate > 0.20:
            warnings.append(ValidationWarning(
                severity="warning",
                category="bounds",
                message="Reserve emission rate >20% may exhaust reserve quickly",
                details=f"Current rate: {self.config.yield_source.reserve_annual_rate*100:.1f}%"
            ))

        # Check reward curve k bounds
        if self.config.reward_curve.k <= 0:
            warnings.append(ValidationWarning(
                severity="error",
                category="input",
                message="Reward curve exponent k must be positive",
                details=f"Current value: {self.config.reward_curve.k}"
            ))

        # Check duration range validity
        if self.config.reward_curve.min_duration_years >= self.config.reward_curve.max_duration_years:
            warnings.append(ValidationWarning(
                severity="error",
                category="input",
                message="Min lock duration must be less than max duration",
                details=f"Min: {self.config.reward_curve.min_duration_years:.2f} yrs, Max: {self.config.reward_curve.max_duration_years:.2f} yrs"
            ))

        # Check capital flow fractions sum to ~1
        flow_sum = (
            self.config.capital_flow.net_new_fraction +
            self.config.capital_flow.recycled_fraction +
            self.config.capital_flow.cannibalized_fraction
        )
        if abs(flow_sum - 1.0) > 0.01:
            warnings.append(ValidationWarning(
                severity="warning",
                category="input",
                message=f"Capital flow fractions sum to {flow_sum:.2f}, expected ~1.0",
                details="Net-new + Recycled + Cannibalized should equal 1.0"
            ))

        # Check cohort fractions sum to ~1
        cohort_sum = (
            self.config.cohorts.retail.size_fraction +
            self.config.cohorts.institutional.size_fraction +
            self.config.cohorts.storage_providers.size_fraction +
            self.config.cohorts.treasuries.size_fraction
        )
        if abs(cohort_sum - 1.0) > 0.01:
            warnings.append(ValidationWarning(
                severity="warning",
                category="input",
                message=f"Cohort fractions sum to {cohort_sum:.2f}, expected ~1.0",
                details="All cohort size fractions should sum to 1.0"
            ))

        # Check bootstrap APY is reasonable
        if self.config.simulation.bootstrap_apy > 0.50:
            warnings.append(ValidationWarning(
                severity="warning",
                category="bounds",
                message="Bootstrap APY >50% is unusually high",
                details=f"Current value: {self.config.simulation.bootstrap_apy*100:.1f}%"
            ))

        # Check alternative yields are reasonable
        alternatives = [
            ("iFIL (GLIF) APY", self.config.alternatives.ifil_apy),
            ("DeFi APY", self.config.alternatives.defi_apy),
        ]
        for name, apy in alternatives:
            if apy > 0.50:
                warnings.append(ValidationWarning(
                    severity="warning",
                    category="bounds",
                    message=f"{name} of {apy*100:.1f}% is unusually high",
                    details="Consider if this is realistic for sustained periods"
                ))

        return warnings

    def check_state(self, state: SystemState) -> List[ValidationWarning]:
        """
        Check simulation state for issues.

        Args:
            state: Current system state

        Returns:
            List of validation warnings
        """
        warnings = []

        # Check for negative values
        if state.circulating < 0:
            warnings.append(ValidationWarning(
                severity="error",
                category="bounds",
                message=f"Circulating supply went negative at t={state.t:.0f}",
                details=f"Value: {state.circulating:,.0f} FIL"
            ))

        if state.reserve < 0:
            warnings.append(ValidationWarning(
                severity="error",
                category="bounds",
                message=f"Reserve went negative at t={state.t:.0f}",
                details=f"Value: {state.reserve:,.0f} FIL"
            ))

        if state.locked_vefil < 0:
            warnings.append(ValidationWarning(
                severity="error",
                category="bounds",
                message=f"Locked supply went negative at t={state.t:.0f}",
                details=f"Value: {state.locked_vefil:,.0f} FIL"
            ))

        # Check conservation
        is_valid, error_msg = state.validate_conservation()
        if not is_valid:
            warnings.append(ValidationWarning(
                severity="error",
                category="conservation",
                message="Conservation law violated",
                details=error_msg
            ))

        # Check for NaN values
        values_to_check = [
            ("total_supply", state.total_supply),
            ("circulating", state.circulating),
            ("locked_vefil", state.locked_vefil),
            ("reserve", state.reserve),
            ("lending_pool", state.lending_pool),
            ("sp_collateral", state.sp_collateral),
        ]
        for name, value in values_to_check:
            if math.isnan(value) or math.isinf(value):
                warnings.append(ValidationWarning(
                    severity="error",
                    category="nan",
                    message=f"Invalid value detected in {name} at t={state.t:.0f}",
                    details=f"Value: {value}"
                ))

        return warnings

    def check_metrics(self, metrics: Dict[str, Any]) -> List[ValidationWarning]:
        """
        Check computed metrics for issues.

        Args:
            metrics: Computed metrics dictionary

        Returns:
            List of validation warnings
        """
        warnings = []

        # Check for NaN in metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if math.isnan(value) or math.isinf(value):
                    warnings.append(ValidationWarning(
                        severity="error",
                        category="nan",
                        message=f"Invalid metric value for {key}",
                        details=f"Value: {value}"
                    ))

        # Check effective inflation is reasonable
        eff_inflation = metrics.get('effective_inflation', 0)
        if abs(eff_inflation) > 1.0:  # >100% inflation
            warnings.append(ValidationWarning(
                severity="warning",
                category="bounds",
                message=f"Extreme effective inflation: {eff_inflation*100:+.1f}%",
                details="This may indicate parameter misconfiguration"
            ))

        # Check locked-to-emission ratio against internal 10:1 heuristic
        emission = float(metrics.get('emission', 0.0) or 0.0)
        locked = float(metrics.get('locked', 0.0) or 0.0)
        dt_days = float(self.config.simulation.timestep_days)
        if emission > 0 and dt_days > 0:
            annual_emission = emission * (365.25 / dt_days)
            if annual_emission > 0:
                lock_to_emission = locked / annual_emission
                if lock_to_emission < 5.0:
                    warnings.append(ValidationWarning(
                        severity="warning",
                        category="sustainability",
                        message=(
                            "Locked/annual emission ratio is low "
                            f"({lock_to_emission:.1f}x)"
                        ),
                        details=(
                            f"Annualized emission: {annual_emission:,.0f} FIL/yr, "
                            f"Locked: {locked:,.0f} FIL"
                        )
                    ))

        return warnings


def validate_simulation_results(
    config: Config,
    states: List[SystemState],
    metrics_over_time: List[Dict[str, Any]]
) -> List[ValidationWarning]:
    """
    Validate complete simulation results.

    Args:
        config: Simulation configuration
        states: List of system states over time
        metrics_over_time: List of metrics dictionaries

    Returns:
        List of all validation warnings
    """
    checker = SanityChecker(config)
    warnings = []

    # Check config first
    warnings.extend(checker.check_config_inputs())

    # Check each state (sample to avoid too many warnings)
    sample_indices = list(range(0, len(states), max(1, len(states) // 10)))  # ~10 samples
    sample_indices.append(len(states) - 1)  # Always include final state

    for i in set(sample_indices):
        if i < len(states):
            warnings.extend(checker.check_state(states[i]))

    # Check final metrics
    if metrics_over_time:
        warnings.extend(checker.check_metrics(metrics_over_time[-1]))

    # Check for reserve exhaustion
    if states:
        final_state = states[-1]
        initial_reserve = config.initial_supply.reserve
        if final_state.reserve < initial_reserve * 0.1:
            warnings.append(ValidationWarning(
                severity="warning",
                category="sustainability",
                message="Reserve below 10% of initial value",
                details=f"Final reserve: {final_state.reserve:,.0f} FIL ({final_state.reserve/initial_reserve*100:.1f}% remaining)"
            ))

    return warnings
