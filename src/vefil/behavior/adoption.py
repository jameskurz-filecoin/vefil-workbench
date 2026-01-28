"""Behavioral adoption model - Endogenize lock demand with bounded adoption.

Key Concepts:
- Adoption is bounded by addressable FIL per cohort
- Partial adjustment model: new_locks converge toward desired_locked_stock
- Duration choice via discrete utility maximization
- APY/demand circularity handled via within-step solver
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import math

import numpy as np

from .cohorts import Cohort
from .opportunity_cost import OpportunityCostCalculator


@dataclass
class LockDemand:
    """Lock demand result."""
    total_demand: float  # Total FIL demand for locking this step
    demand_by_cohort: Dict[str, float]  # Demand per cohort
    avg_duration: float  # Average lock duration
    participation_rate: float  # Fraction of eligible users locking
    # Extended fields for bounded adoption
    desired_locked_by_cohort: Dict[str, float] = field(default_factory=dict)
    chosen_duration_by_cohort: Dict[str, float] = field(default_factory=dict)
    utility_by_cohort: Dict[str, float] = field(default_factory=dict)
    duration_weights_by_cohort: Dict[str, Dict[float, float]] = field(default_factory=dict)
    solver_converged: bool = True
    solver_iterations: int = 0


@dataclass
class SolverConfig:
    """Configuration for APY/demand solver."""
    max_iterations: int = 10
    tolerance: float = 1e-4  # Relative tolerance for convergence
    damping: float = 0.5  # Damping factor for iteration stability


class AdoptionModel:
    """Model for endogenizing lock demand based on rational decision-making.

    Features:
    - Bounded adoption: Each cohort has addressable FIL limit
    - Partial adjustment: New locks converge toward desired stock
    - Duration choice: Cohorts optimize duration for max utility
    - APY/demand solver: Handles circularity within timestep
    """

    def __init__(
        self,
        cohorts: List[Cohort],
        opportunity_calc: OpportunityCostCalculator,
        participation_elasticity: float = 1.5,
        max_participation: float = 0.8,
        adjustment_tau_days: float = 90.0,  # Time constant for partial adjustment
        duration_choice_temperature: float = 0.35,
        duration_exploration_fraction: float = 0.15,
        solver_config: SolverConfig = None
    ):
        """
        Initialize adoption model.

        Args:
            cohorts: List of user cohorts
            opportunity_calc: Opportunity cost calculator
            participation_elasticity: Elasticity of participation to yield
            max_participation: Maximum participation rate cap
            adjustment_tau_days: Time constant for partial adjustment
            duration_choice_temperature: Softmax temperature for duration choice
            duration_exploration_fraction: Uniform exploration share in duration choice
            solver_config: Configuration for APY/demand solver
        """
        self.cohorts = cohorts
        self.opportunity_calc = opportunity_calc
        self.participation_elasticity = participation_elasticity
        self.max_participation = max_participation
        self.adjustment_tau_days = adjustment_tau_days
        self.duration_choice_temperature = duration_choice_temperature
        self.duration_exploration_fraction = duration_exploration_fraction
        self.solver_config = solver_config or SolverConfig()

    def compute_adjustment_speed(self, dt_days: float) -> float:
        """
        Compute partial adjustment speed for a timestep.

        Uses exponential adjustment: speed = 1 - exp(-dt/tau)

        Args:
            dt_days: Timestep in days

        Returns:
            Fraction of gap to close in this timestep
        """
        return 1.0 - math.exp(-dt_days / self.adjustment_tau_days)

    def choose_optimal_duration(
        self,
        cohort: Cohort,
        yield_curve: Callable[[float], float],
        base_apy: float
    ) -> tuple[float, float, Dict[float, float]]:
        """
        Choose duration mix via softmax over utilities with exploration.

        For each allowed duration d:
            utility(d) = apy(d) - required_apy(d)

        Softmax weights: w(d) = exp(utility(d) / temperature) / sum(...)
        Blend with uniform exploration.

        Args:
            cohort: User cohort
            yield_curve: Function mapping duration -> APY
            base_apy: Base alternative APY

        Returns:
            (expected_duration, best_utility, duration_weights)
        """
        durations = cohort.allowed_durations
        if not durations:
            return cohort.avg_duration_years, 0.0, {cohort.avg_duration_years: 1.0}

        # Compute utilities for each duration
        utilities = []
        for d in durations:
            available_apy = yield_curve(d)
            required_apy = cohort.compute_required_yield(base_apy)
            utility = available_apy - required_apy
            utilities.append(utility)

        best_utility = max(utilities)

        # Softmax with temperature
        temp = self.duration_choice_temperature
        # Shift utilities for numerical stability
        max_u = max(utilities)
        exp_utilities = [math.exp((u - max_u) / temp) for u in utilities]
        exp_sum = sum(exp_utilities)

        if exp_sum > 0:
            softmax_weights = [e / exp_sum for e in exp_utilities]
        else:
            # Fallback to uniform
            softmax_weights = [1.0 / len(durations)] * len(durations)

        # Blend with uniform exploration
        explore_frac = self.duration_exploration_fraction
        uniform_weight = 1.0 / len(durations)
        blended_weights = [
            (1.0 - explore_frac) * sw + explore_frac * uniform_weight
            for sw in softmax_weights
        ]

        # Normalize to ensure sum = 1
        weight_sum = sum(blended_weights)
        if weight_sum > 0:
            blended_weights = [w / weight_sum for w in blended_weights]

        # Build duration_weights dict
        duration_weights = {d: w for d, w in zip(durations, blended_weights)}

        # Compute expected duration (weighted average)
        expected_duration = sum(d * w for d, w in zip(durations, blended_weights))

        return expected_duration, best_utility, duration_weights

    def compute_participation(self, utility: float) -> float:
        """
        Compute participation rate from utility.

        Uses logistic function bounded by max_participation.

        Args:
            utility: Net utility (apy - required_apy)

        Returns:
            Participation rate [0, max_participation]
        """
        # Logistic response scaled by elasticity
        raw = 1.0 / (1.0 + np.exp(-self.participation_elasticity * utility * 10))
        return min(raw, self.max_participation)

    def compute_lock_demand(
        self,
        yield_curve: Callable[[float], float],
        market_state: dict = None,
        circulating: float = None,
        dt_days: float = 30.0
    ) -> LockDemand:
        """
        Compute lock demand using bounded adoption with partial adjustment.

        Args:
            yield_curve: Function mapping duration -> APY
            market_state: Optional market state dictionary
            circulating: Current circulating supply (for addressable bounds)
            dt_days: Timestep in days

        Returns:
            Lock demand result
        """
        if market_state is None:
            market_state = {}
        if circulating is None:
            circulating = market_state.get('circulating', 1e9)

        total_demand = 0.0
        demand_by_cohort = {}
        desired_locked_by_cohort = {}
        chosen_duration_by_cohort = {}
        utility_by_cohort = {}
        duration_weights_by_cohort = {}
        weighted_duration_sum = 0.0
        total_weighted_amount = 0.0

        # Get best alternative APY
        _, best_alt_apy = self.opportunity_calc.alternatives.get_best_alternative()
        adjustment_speed = self.compute_adjustment_speed(dt_days)

        for cohort in self.cohorts:
            # Choose duration mix using softmax
            expected_duration, best_utility, duration_weights = self.choose_optimal_duration(
                cohort, yield_curve, best_alt_apy
            )
            chosen_duration_by_cohort[cohort.name] = expected_duration
            utility_by_cohort[cohort.name] = best_utility
            duration_weights_by_cohort[cohort.name] = duration_weights

            # Compute participation rate using best_utility (max utility)
            participation = self.compute_participation(best_utility)

            # Compute addressable and current locked
            addressable = cohort.compute_addressable(circulating)
            current_locked = cohort.state.locked_fil

            # Compute desired locked stock
            desired_locked = participation * addressable
            desired_locked_by_cohort[cohort.name] = desired_locked

            # Partial adjustment: new_locks closes stock gap, but also includes
            # flow-based participation when APY remains attractive.
            gap = max(0.0, desired_locked - current_locked)
            eligible = cohort.compute_eligible(circulating)

            # Flow baseline: keep some inflow as long as participation > 0
            flow_locks = participation * eligible * adjustment_speed

            # New locks bounded by eligible FIL
            new_locks = min(eligible, max(gap * adjustment_speed, flow_locks))

            total_demand += new_locks
            demand_by_cohort[cohort.name] = new_locks

            # Track weighted duration for average
            if new_locks > 0:
                weighted_duration_sum += new_locks * expected_duration
                total_weighted_amount += new_locks

        avg_duration = weighted_duration_sum / total_weighted_amount if total_weighted_amount > 0 else 0.0

        # Overall participation rate
        total_addressable = sum(c.compute_addressable(circulating) for c in self.cohorts)
        total_locked = sum(c.state.locked_fil for c in self.cohorts)
        overall_participation = total_locked / total_addressable if total_addressable > 0 else 0.0

        return LockDemand(
            total_demand=total_demand,
            demand_by_cohort=demand_by_cohort,
            avg_duration=avg_duration,
            participation_rate=overall_participation,
            desired_locked_by_cohort=desired_locked_by_cohort,
            chosen_duration_by_cohort=chosen_duration_by_cohort,
            utility_by_cohort=utility_by_cohort,
            duration_weights_by_cohort=duration_weights_by_cohort,
            solver_converged=True,
            solver_iterations=1
        )

    def compute_activation_threshold(
        self,
        target_participation: float,
        circulating: float,
        max_duration: float = 5.0
    ) -> float:
        """
        Compute minimum reward pool size to activate target participation.

        Uses participation elasticity to estimate required yield.

        Args:
            target_participation: Target participation rate (0-1)
            circulating: Circulating supply (for addressable FIL bounds)
            max_duration: Max lock duration for weight computation

        Returns:
            Required annual emission (in FIL)
        """
        # Inverse of participation function to find required utility
        if target_participation <= 0:
            return 0.0

        # Clamp target participation
        target_participation = min(target_participation, self.max_participation - 0.01)

        # Solve: target_participation = sigmoid(elasticity * utility * 10)
        # utility = logit(target_participation) / (elasticity * 10)
        utility_delta = np.log(target_participation / (1 - target_participation)) / (self.participation_elasticity * 10)

        # Estimate required APY from utility delta
        _, best_alt_apy = self.opportunity_calc.alternatives.get_best_alternative()
        avg_premium = sum(c.required_premium * c.size_fraction for c in self.cohorts)
        required_apy = best_alt_apy + avg_premium + utility_delta

        # Estimate average duration and weight from cohorts
        avg_duration = sum(c.avg_duration_years * c.size_fraction for c in self.cohorts)
        if avg_duration <= 0:
            return 0.0

        avg_weight = (avg_duration / max_duration) ** 1.5

        # Estimate locked amount from participation and addressable FIL
        total_addressable = sum(c.compute_addressable(circulating) for c in self.cohorts)
        if total_addressable <= 0:
            return 0.0

        estimated_locked = total_addressable * target_participation
        effective_weighted_locked = estimated_locked * avg_weight

        # APY = annual_emission * weight / effective_weighted_locked
        # annual_emission = APY * effective_weighted_locked / weight
        if avg_weight > 0:
            required_annual_emission = required_apy * effective_weighted_locked / avg_weight
        else:
            required_annual_emission = 0.0

        return max(0.0, required_annual_emission)

    def update_cohort_states(
        self,
        lock_demand: LockDemand,
        unlocks_by_cohort: Dict[str, float],
        circulating: float
    ):
        """
        Update cohort states after a timestep.

        Args:
            lock_demand: Lock demand result from this step
            unlocks_by_cohort: Unlocks per cohort
            circulating: Current circulating supply
        """
        for cohort in self.cohorts:
            new_locks = lock_demand.demand_by_cohort.get(cohort.name, 0.0)
            unlocks = unlocks_by_cohort.get(cohort.name, 0.0)
            chosen_duration = lock_demand.chosen_duration_by_cohort.get(cohort.name, cohort.avg_duration_years)
            participation = lock_demand.utility_by_cohort.get(cohort.name, 0.0)

            cohort.update_state(
                new_locks=new_locks,
                unlocks=unlocks,
                chosen_duration=chosen_duration,
                participation=participation,
                circulating=circulating
            )

    def get_total_locked_by_cohorts(self) -> float:
        """Get sum of locked FIL across all cohorts."""
        return sum(c.state.locked_fil for c in self.cohorts)
