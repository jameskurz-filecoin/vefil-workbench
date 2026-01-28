"""Simulation runner - Orchestrate full scenario execution.

Key Features:
- Pure reserve emissions from the 300M mining reserve
- Tracks burned_cumulative in SystemState
- Uses actual effective_weighted_locked from positions (no hard-coded avg_weight)
- Bounded adoption with addressable FIL per cohort
- Cohort-level position tracking
"""

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

import numpy as np

from ..behavior.adoption import AdoptionModel
from ..behavior.cohorts import Cohort
from ..behavior.opportunity_cost import AlternativeYields, OpportunityCostCalculator
from ..config.schema import Config
from ..engine.accounting import AccountingEngine, Flows, SystemState
from ..engine.capital_flow import CapitalFlowModel
from ..engine.lending import LendingMarketModel
from ..engine.rewards import LockPosition, YieldSourceEngine, RewardBudget


@dataclass
class SimulationResult:
    """Complete simulation result."""
    config: Config
    states: List[SystemState]
    metrics_over_time: List[Dict[str, Any]]
    final_metrics: Dict[str, Any]
    conservation_errors: List[str] = field(default_factory=list)
    solver_warnings: List[str] = field(default_factory=list)


class SimulationRunner:
    """Main simulation runner with model realism upgrades."""

    def __init__(self, config: Config):
        """
        Initialize simulation runner.

        Args:
            config: Simulation configuration
        """
        self.config = config
        self._conservation_errors: List[str] = []
        self._solver_warnings: List[str] = []

        # Initialize engines
        self.accounting = AccountingEngine(
            initial_state=SystemState(
                t=0.0,
                total_supply=config.initial_supply.total,
                circulating=config.initial_supply.circulating,
                locked_vefil=config.initial_supply.locked_vefil,
                reserve=config.initial_supply.reserve,
                lending_pool=config.initial_supply.lending_pool,
                sp_collateral=config.initial_supply.sp_collateral,
                other_allocations=config.initial_supply.other_allocations,
                burned_cumulative=0.0  # Initialize burned bucket
            )
        )

        self.yield_source = YieldSourceEngine(
            reserve_annual_rate=config.yield_source.reserve_annual_rate,
            reward_curve_k=config.reward_curve.k,
            max_duration_years=config.reward_curve.max_duration_years
        )

        # Derive cannibalized fraction from net-new + recycled to keep totals consistent.
        derived_cannibalized = max(
            0.0,
            1.0 - config.capital_flow.net_new_fraction - config.capital_flow.recycled_fraction
        )
        self.config.capital_flow.cannibalized_fraction = derived_cannibalized

        self.capital_flow = CapitalFlowModel(
            net_new_fraction=config.capital_flow.net_new_fraction,
            recycled_fraction=config.capital_flow.recycled_fraction,
            cannibalized_fraction=derived_cannibalized,
            liquidity_regime=config.market.liquidity_regime,
            order_book_depth=config.market.order_book_depth
        )

        self.lending_market = LendingMarketModel(
            base_rate=config.lending.base_rate,
            utilization_elasticity=config.lending.utilization_elasticity,
            sp_borrow_demand=config.lending.sp_borrow_demand
        )

        # Initialize behavioral adoption model
        alternatives = AlternativeYields(
            ifil_apy=config.alternatives.ifil_apy,
            glif_apy=config.alternatives.glif_apy,
            defi_apy=config.alternatives.defi_apy,
            risk_free_rate=config.alternatives.risk_free_rate
        )
        # GLIF emits iFIL receipts; keep the config aligned with that invariant.
        self.config.alternatives.glif_apy = self.config.alternatives.ifil_apy

        opportunity_calc = OpportunityCostCalculator(
            alternatives=alternatives,
            volatility=config.market.volatility
        )

        # Create cohorts from config with addressable FIL bounds
        # Compute addressable FIL as fraction of initial circulating
        initial_circulating = config.initial_supply.circulating
        addressable_cap = config.simulation.addressable_cap
        cohorts = [
            Cohort(
                name="retail",
                size_fraction=config.cohorts.retail.size_fraction,
                required_premium=config.cohorts.retail.required_premium,
                avg_lock_size=config.cohorts.retail.avg_lock_size,
                avg_duration_years=config.cohorts.retail.avg_duration_years,
                risk_tolerance="medium",
                # Addressable FIL: fraction of circulating allocated to this cohort
                addressable_fraction=config.cohorts.retail.size_fraction * addressable_cap
            ),
            Cohort(
                name="institutional",
                size_fraction=config.cohorts.institutional.size_fraction,
                required_premium=config.cohorts.institutional.required_premium,
                avg_lock_size=config.cohorts.institutional.avg_lock_size,
                avg_duration_years=config.cohorts.institutional.avg_duration_years,
                risk_tolerance="low",
                addressable_fraction=config.cohorts.institutional.size_fraction * addressable_cap
            ),
            Cohort(
                name="storage_providers",
                size_fraction=config.cohorts.storage_providers.size_fraction,
                required_premium=config.cohorts.storage_providers.required_premium,
                avg_lock_size=config.cohorts.storage_providers.avg_lock_size,
                avg_duration_years=config.cohorts.storage_providers.avg_duration_years,
                risk_tolerance="medium",
                addressable_fraction=config.cohorts.storage_providers.size_fraction * addressable_cap
            ),
            Cohort(
                name="treasuries",
                size_fraction=config.cohorts.treasuries.size_fraction,
                required_premium=config.cohorts.treasuries.required_premium,
                avg_lock_size=config.cohorts.treasuries.avg_lock_size,
                avg_duration_years=config.cohorts.treasuries.avg_duration_years,
                risk_tolerance="low",
                addressable_fraction=config.cohorts.treasuries.size_fraction * addressable_cap
            )
        ]

        # Use participation_elasticity from config
        self.adoption_model = AdoptionModel(
            cohorts=cohorts,
            opportunity_calc=opportunity_calc,
            participation_elasticity=config.simulation.participation_elasticity,
            max_participation=config.simulation.max_participation,
            adjustment_tau_days=config.simulation.adjustment_tau_days,
            duration_choice_temperature=config.simulation.duration_choice_temperature,
            duration_exploration_fraction=config.simulation.duration_exploration_fraction,
        )

        # Initialize emissions policy state
        self._policy_state = {
            "reserve_rate": config.yield_source.reserve_annual_rate,
            "peak_locked": 0.0,  # Track peak locked for drawdown calculation
        }

        # Store alternatives for policy calculations
        self._alternatives = alternatives

    def run(self, random_seed: int = None) -> SimulationResult:
        """
        Run the simulation.

        Args:
            random_seed: Random seed for reproducibility

        Returns:
            Simulation result
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        else:
            np.random.seed(self.config.simulation.random_seed)

        dt_days = self.config.simulation.timestep_days
        num_steps = int((self.config.simulation.time_horizon_months * 30) / dt_days)

        # Get bootstrap parameters from config
        bootstrap_apy = self.config.simulation.bootstrap_apy
        adoption_ramp_fraction = self.config.simulation.adoption_ramp_fraction

        states = []
        metrics_over_time = []
        lock_positions: List[LockPosition] = []
        self._conservation_errors = []
        self._solver_warnings = []

        for step in range(num_steps):
            t = step * dt_days

            # Emissions policy: adjust reserve_rate dynamically
            policy_active = False
            policy_apy_floor = 0.0
            lock_growth_annual = 0.0
            cap_active = False
            max_inflation = 0.0
            current_apy_estimate = 0.0
            lock_progress = 0.0

            if self.config.emissions_policy.enabled:
                policy = self.config.emissions_policy
                cap_active = (
                    t >= policy.cap_start_months * 30.0 or
                    self.accounting.current_state.locked_vefil >= policy.lock_target_fil
                )
                if cap_active and policy.cap_ramp_months > 0:
                    ramp = (t - policy.cap_start_months * 30.0) / (policy.cap_ramp_months * 30.0)
                    ramp = min(1.0, max(0.0, ramp))
                    max_inflation = (
                        policy.early_max_effective_inflation +
                        ramp * (policy.max_effective_inflation - policy.early_max_effective_inflation)
                    )
                else:
                    max_inflation = policy.max_effective_inflation if cap_active else policy.early_max_effective_inflation

                # Compute best_alt_apy and avg_premium
                _, best_alt_apy = self._alternatives.get_best_alternative()
                avg_premium = (
                    self.config.cohorts.retail.required_premium * self.config.cohorts.retail.size_fraction +
                    self.config.cohorts.institutional.required_premium * self.config.cohorts.institutional.size_fraction +
                    self.config.cohorts.storage_providers.required_premium * self.config.cohorts.storage_providers.size_fraction +
                    self.config.cohorts.treasuries.required_premium * self.config.cohorts.treasuries.size_fraction
                )
                policy_apy_floor = best_alt_apy + avg_premium + policy.apy_floor_buffer

                # Compute lock growth over window
                current_locked = self.accounting.current_state.locked_vefil
                if policy.lock_target_fil > 0:
                    lock_progress = min(1.0, current_locked / policy.lock_target_fil)
                window_days = policy.lock_growth_window_months * 30
                if len(states) > 0:
                    # Find state from window_days ago
                    window_start_idx = max(0, len(states) - int(window_days / dt_days))
                    window_start_locked = states[window_start_idx].locked_vefil if window_start_idx < len(states) else 0.0
                    if window_start_locked > 0:
                        lock_growth_window = (current_locked - window_start_locked) / window_start_locked
                        # Annualize the growth rate
                        window_fraction = min(1.0, (len(states) - window_start_idx) * dt_days / 365.25)
                        lock_growth_annual = lock_growth_window / window_fraction if window_fraction > 0 else 0.0
                    else:
                        lock_growth_annual = 0.0
                else:
                    lock_growth_annual = 0.0

                # Update peak locked and compute drawdown
                self._policy_state["peak_locked"] = max(self._policy_state["peak_locked"], current_locked)
                peak_locked = self._policy_state["peak_locked"]
                drawdown = (peak_locked - current_locked) / peak_locked if peak_locked > 0 else 0.0

                # Get previous step's effective inflation
                prev_effective_inflation = 0.0
                if metrics_over_time:
                    prev_effective_inflation = metrics_over_time[-1].get("effective_inflation", 0.0)

                # Estimate current APY (rough estimate before full solver)
                self.yield_source.reserve_annual_rate = self._policy_state["reserve_rate"]
                budget_estimate = self.yield_source.compute_reward_budget(
                    self.accounting.current_state.reserve,
                    dt_days
                )
                annual_budget_estimate = budget_estimate.total_reward_budget * (365.25 / dt_days)
                ewl_estimate = self.yield_source.compute_effective_weighted_locked(lock_positions)
                current_apy_estimate = annual_budget_estimate / ewl_estimate if ewl_estimate > 0 else policy_apy_floor

                # Policy trigger conditions
                trigger_growth = lock_growth_annual < policy.lock_growth_target_annual
                trigger_drawdown = drawdown > policy.lock_drawdown_threshold
                trigger_apy = current_apy_estimate < policy_apy_floor

                if trigger_growth or trigger_drawdown or trigger_apy:
                    policy_active = True

                    # Target APY to restore
                    growth_boost = policy.growth_apy_boost if (trigger_growth or trigger_drawdown) else 0.0
                    target_apy = policy_apy_floor * 1.05 + growth_boost  # Aim above floor with growth boost

                    # Required annual reward budget to hit target APY
                    required_annual_budget = target_apy * ewl_estimate if ewl_estimate > 0 else annual_budget_estimate

                    # Move reserve rate toward required level (bounded)
                    current_reserve_rate = self._policy_state["reserve_rate"]
                    reserve_balance = self.accounting.current_state.reserve
                    if reserve_balance > 0:
                        required_reserve_rate = required_annual_budget / reserve_balance
                        target_reserve_rate = min(policy.reserve_rate_max, max(policy.reserve_rate_min, required_reserve_rate))
                        new_reserve_rate = current_reserve_rate + policy.adjustment_speed * (target_reserve_rate - current_reserve_rate)
                        self._policy_state["reserve_rate"] = new_reserve_rate

                # If effective_inflation > max and not in drawdown, reduce reserve rate
                elif cap_active and prev_effective_inflation > max_inflation and drawdown <= policy.lock_drawdown_threshold:
                    current_reserve_rate = self._policy_state["reserve_rate"]
                    if current_reserve_rate > policy.reserve_rate_min:
                        target_reserve_rate = max(policy.reserve_rate_min, current_reserve_rate * 0.95)
                        new_reserve_rate = current_reserve_rate + policy.adjustment_speed * (target_reserve_rate - current_reserve_rate)
                        self._policy_state["reserve_rate"] = new_reserve_rate

                # During growth phase, don't let reserve rate fall below baseline
                if not cap_active:
                    baseline_rate = self.config.yield_source.reserve_annual_rate
                    if self._policy_state["reserve_rate"] < baseline_rate:
                        self._policy_state["reserve_rate"] = baseline_rate

            # Apply policy-adjusted rate to yield source engine
            self.yield_source.reserve_annual_rate = self._policy_state["reserve_rate"]

            # Compute reward budget from reserve
            reward_budget = self.yield_source.compute_reward_budget(
                self.accounting.current_state.reserve,
                dt_days
            )

            total_locked = self.accounting.current_state.locked_vefil
            circulating = self.accounting.current_state.circulating

            # Compute annual reward budget (used in APY calculations)
            annual_reward_budget = reward_budget.total_reward_budget * (365.25 / dt_days)

            # Compute effective weighted locked from existing positions
            existing_ewl = self.yield_source.compute_effective_weighted_locked(lock_positions)

            # Apply adoption ramp (gradual rollout)
            adoption_progress = min(1.0, step / (num_steps * adoption_ramp_fraction))
            adoption_scaling = 1.0 - np.exp(-3 * adoption_progress)

            # Cap using max_step_lock_fraction from config
            cap = circulating * self.config.simulation.max_step_lock_fraction

            # APY↔demand fixed-point solver
            solver_config = self.adoption_model.solver_config
            expected_new_locks = 0.0
            avg_new_weight = 0.5  # Initial guess
            lock_demand = None
            solver_converged = False
            current_apy = bootstrap_apy

            for iteration in range(solver_config.max_iterations):
                # Estimate ewl with expected new locks
                ewl_guess = existing_ewl + expected_new_locks * avg_new_weight

                # Compute current APY
                if ewl_guess > 0:
                    current_apy = annual_reward_budget / ewl_guess
                else:
                    current_apy = bootstrap_apy

                # Create yield curve function
                def yield_curve(duration_years: float, apy=current_apy, ewl=ewl_guess) -> float:
                    """Compute APY for a given duration."""
                    weight = self.yield_source.compute_lock_weight(duration_years)
                    if ewl > 0:
                        return apy * weight
                    else:
                        return apy * weight / 0.5

                # Compute lock demand
                lock_demand = self.adoption_model.compute_lock_demand(
                    yield_curve=yield_curve,
                    market_state={
                        'emission': reward_budget.total_reward_budget,
                        'locked': total_locked,
                        'circulating': circulating
                    },
                    circulating=circulating,
                    dt_days=dt_days
                )

                # Apply ramp and cap
                target_new_locks = min(lock_demand.total_demand * adoption_scaling, cap)

                # Recompute avg_new_weight from duration_weights_by_cohort
                if lock_demand.total_demand > 0:
                    weighted_weight_sum = 0.0
                    total_weight_amount = 0.0
                    for cohort_name, cohort_demand in lock_demand.demand_by_cohort.items():
                        if cohort_demand > 0:
                            duration_weights = lock_demand.duration_weights_by_cohort.get(cohort_name, {})
                            for duration, weight_frac in duration_weights.items():
                                lock_weight = self.yield_source.compute_lock_weight(duration)
                                weighted_weight_sum += cohort_demand * weight_frac * lock_weight
                                total_weight_amount += cohort_demand * weight_frac
                    if total_weight_amount > 0:
                        avg_new_weight = weighted_weight_sum / total_weight_amount

                # Damped update
                prev_expected = expected_new_locks
                expected_new_locks = (
                    solver_config.damping * target_new_locks +
                    (1 - solver_config.damping) * prev_expected
                )

                # Check convergence
                if prev_expected > 0:
                    rel_change = abs(expected_new_locks - prev_expected) / prev_expected
                    if rel_change < solver_config.tolerance:
                        solver_converged = True
                        break
                elif expected_new_locks == 0:
                    solver_converged = True
                    break

            # Track solver convergence
            if not solver_converged:
                self._solver_warnings.append(f"t={t}: APY/demand solver did not converge")

            # Final new_locks_demand from solver
            new_locks_demand = max(0.0, expected_new_locks)
            base_new_locks_demand = new_locks_demand

            # Process unlocks from expired positions (BEFORE removing them)
            raw_unlocks = self._compute_unlocks(lock_positions, t)
            unlocks_by_cohort = self._compute_unlocks_by_cohort(lock_positions, t)

            # Model relocking explicitly
            relock_fraction = self.config.simulation.relock_fraction_unlocked
            competitive_apy = False
            if self.config.emissions_policy.enabled and policy_apy_floor > 0:
                competitive_apy = current_apy >= policy_apy_floor
            else:
                competitive_apy = current_apy > 0
            if self.config.emissions_policy.enabled:
                if lock_progress >= 1.0:
                    relock_fraction = self.config.simulation.relock_max_fraction
                elif competitive_apy:
                    relock_fraction = self.config.simulation.relock_max_fraction
            relocks_by_cohort = {k: v * relock_fraction for k, v in unlocks_by_cohort.items()}
            effective_unlocks_by_cohort = {
                k: unlocks_by_cohort.get(k, 0.0) - relocks_by_cohort.get(k, 0.0)
                for k in unlocks_by_cohort
            }
            relocks_total = sum(relocks_by_cohort.values())
            effective_unlocks = sum(effective_unlocks_by_cohort.values())

            sell_rate = max(0.0, min(1.0, self.config.capital_flow.sell_rate))

            # Replacement demand: if rewards remain attractive, assume churn is refilled by new entrants
            replacement_by_cohort = {}
            reserve_emission_active = reward_budget.reserve_emission > 0
            if current_apy > 0:
                max_participation = max(self.adoption_model.max_participation, 1e-6)
                for cohort_name, eff_unlock in effective_unlocks_by_cohort.items():
                    utility = lock_demand.utility_by_cohort.get(cohort_name, 0.0)
                    if reserve_emission_active:
                        replacement_frac = 1.0
                    elif competitive_apy:
                        replacement_frac = 1.0
                    elif utility > 0:
                        replacement_frac = 1.0
                    else:
                        replacement_frac = self.adoption_model.compute_participation(utility) / max_participation
                    replacement_by_cohort[cohort_name] = eff_unlock * replacement_frac
            replacement_total = sum(replacement_by_cohort.values())
            new_locks_demand = base_new_locks_demand + replacement_total

            # Compute scaling factor for cohorts (ramp + cap combined)
            # This ensures cohort positions sum to exactly base_new_locks_demand before replacements.
            if lock_demand.total_demand > 0:
                cohort_scaling = base_new_locks_demand / lock_demand.total_demand
            else:
                cohort_scaling = 0.0

            scaled_demand_by_cohort = {
                cohort_name: cohort_demand * cohort_scaling
                for cohort_name, cohort_demand in lock_demand.demand_by_cohort.items()
            }

            # Keep separate copies: one for cohort state (demand only), one for positions (demand + relocks)
            scaled_demand_by_cohort_for_state = dict(scaled_demand_by_cohort)

            # Add replacement demand into both state updates and position creation
            for cohort_name, replacement_amount in replacement_by_cohort.items():
                if cohort_name in scaled_demand_by_cohort:
                    scaled_demand_by_cohort[cohort_name] += replacement_amount
                else:
                    scaled_demand_by_cohort[cohort_name] = replacement_amount
                if cohort_name in scaled_demand_by_cohort_for_state:
                    scaled_demand_by_cohort_for_state[cohort_name] += replacement_amount
                else:
                    scaled_demand_by_cohort_for_state[cohort_name] = replacement_amount

            # Hard inflation cap (post-solver, uses actual net lock change)
            # Only applies when projected inflation would significantly exceed target
            if (
                self.config.emissions_policy.enabled and
                cap_active and
                (current_apy_estimate >= policy_apy_floor or lock_progress >= 1.0)
            ):
                policy = self.config.emissions_policy
                dt_years = dt_days / 365.25

                base_net_lock_change = new_locks_demand - effective_unlocks
                emission_effect_coeff = (2.0 * sell_rate) - 1.0

                # Project effective inflation with current reward budget
                # effective_inflation = (emission_sold - (net_locks + reward_relocks)) / circulating
                # reward_relocks = reserve_emission * (1 - sell_rate)
                projected_emission_effect = reward_budget.reserve_emission * emission_effect_coeff - base_net_lock_change
                projected_inflation = projected_emission_effect / circulating if circulating > 0 else 0.0
                projected_inflation_annual = projected_inflation * (365.25 / dt_days)

                # Apply hard cap whenever we'd exceed the target
                inflation_overshoot = projected_inflation_annual - max_inflation
                if inflation_overshoot > 0.0:
                    policy_active = True

                    # Calculate max reserve emission to hit inflation target
                    # target_inflation = (max_emission - net_lock_change) / circulating * annualization
                    # max_emission = net_lock_change + target_inflation * circulating / annualization
                    if emission_effect_coeff > 0:
                        max_reserve_emission = max(
                            0.0,
                            (base_net_lock_change + max_inflation * circulating * dt_years) / emission_effect_coeff
                        )
                    else:
                        # If reward relocks dominate (coeff <= 0), emission does not increase inflation.
                        max_reserve_emission = reward_budget.reserve_emission

                    # Cap reserve rate
                    reserve_balance = self.accounting.current_state.reserve
                    if reserve_balance > 0 and dt_years > 0:
                        capped_rate = max_reserve_emission / (reserve_balance * dt_years)
                        self._policy_state["reserve_rate"] = min(self._policy_state["reserve_rate"], max(0.0, capped_rate))

                    # Recompute reward budget after policy clamp
                    self.yield_source.reserve_annual_rate = self._policy_state["reserve_rate"]

                    reward_budget = self.yield_source.compute_reward_budget(
                        self.accounting.current_state.reserve, dt_days
                    )
                    annual_reward_budget = reward_budget.total_reward_budget * (365.25 / dt_days)

            # Reward relocks: portion of reserve-funded rewards that get relocked
            reward_relock_fraction = max(0.0, min(1.0, 1.0 - sell_rate))
            reward_relocks_total = reward_budget.reserve_emission * reward_relock_fraction
            reward_relocks_by_cohort = {}
            if reward_relocks_total > 0:
                locked_by_cohort = {}
                for pos in lock_positions:
                    cohort = pos.cohort or "unknown"
                    locked_by_cohort[cohort] = locked_by_cohort.get(cohort, 0.0) + pos.amount
                total_locked_positions = sum(locked_by_cohort.values())
                if total_locked_positions > 0:
                    for cohort_name, locked_amount in locked_by_cohort.items():
                        reward_relocks_by_cohort[cohort_name] = reward_relocks_total * (locked_amount / total_locked_positions)

            reward_relocks_total = sum(reward_relocks_by_cohort.values())
            new_locks_demand += reward_relocks_total

            for cohort_name, reward_amount in reward_relocks_by_cohort.items():
                if cohort_name in scaled_demand_by_cohort:
                    scaled_demand_by_cohort[cohort_name] += reward_amount
                else:
                    scaled_demand_by_cohort[cohort_name] = reward_amount
                if cohort_name in scaled_demand_by_cohort_for_state:
                    scaled_demand_by_cohort_for_state[cohort_name] += reward_amount
                else:
                    scaled_demand_by_cohort_for_state[cohort_name] = reward_amount

            # Add relocks into scaled_demand_by_cohort for position creation
            for cohort_name, relock_amount in relocks_by_cohort.items():
                if cohort_name in scaled_demand_by_cohort:
                    scaled_demand_by_cohort[cohort_name] += relock_amount
                else:
                    scaled_demand_by_cohort[cohort_name] = relock_amount

            # Total new locks for positions = demand + relocks
            new_locks_total = new_locks_demand + relocks_total

            # scaled_lock_demand for cohort state update (without relocks in demand)
            scaled_lock_demand = replace(
                lock_demand,
                total_demand=new_locks_demand,
                demand_by_cohort=scaled_demand_by_cohort_for_state,
            )

            # Decompose capital sources (based on market-driven demand, not relocks or reward relocks)
            market_new_locks = base_new_locks_demand + replacement_total
            capital_sources = self.capital_flow.decompose_capital(
                market_new_locks,
                self.accounting.current_state.lending_pool
            )

            # Create flows with proper reserve vs fee semantics
            # Use new_locks_demand and effective_unlocks for consistent accounting
            # (relocks never really "left" locked state, so we don't count them as new)
            # emission_to_circulating = emission - reward_relocks (what's not relocked goes to circ)
            # If no locked positions to receive relocks, all emission goes to circulating
            emission_to_circulating = reward_budget.reserve_emission - reward_relocks_total
            flows = Flows(
                emission=reward_budget.reserve_emission,  # Total reserve emission (depletes reserve)
                emission_to_circulating=emission_to_circulating,  # Portion that hits circulating
                burns=0.0,  # No fee burns in reserve-only model
                net_locks=new_locks_demand,
                unlocks=effective_unlocks,
                lending_cannibalized=capital_sources.cannibalized,
                net_lending_withdrawals=0.0,
                reward_relocks=reward_relocks_total,  # Rewards that go reserve → locked (bypass circulating)
            )

            # Step accounting with conservation tracking
            try:
                new_state = self.accounting.step(flows, dt_days)
            except ValueError as e:
                self._conservation_errors.append(str(e))
                new_state = self._create_corrected_state(flows, dt_days)

            states.append(new_state)

            # Validate conservation
            is_valid, error_msg = new_state.validate_conservation()
            if not is_valid and error_msg not in self._conservation_errors:
                self._conservation_errors.append(error_msg)

            # Remove expired positions (AFTER computing unlocks)
            lock_positions = [pos for pos in lock_positions if pos.unlock_time > t]

            # Update lock positions with cohort tracking
            # Create multiple positions per cohort using duration weights
            if new_locks_total > 0:
                avg_duration_years = lock_demand.avg_duration if lock_demand.avg_duration > 0 else 3.0

                # Create positions for each cohort split across durations
                for cohort_name, cohort_amount in scaled_demand_by_cohort.items():
                    if cohort_amount > 0:
                        duration_weights = lock_demand.duration_weights_by_cohort.get(cohort_name, {})
                        if duration_weights:
                            # Split across durations using weights
                            for duration, weight_frac in duration_weights.items():
                                position_amount = cohort_amount * weight_frac
                                if position_amount > 0:
                                    lock_positions.append(LockPosition(
                                        amount=position_amount,
                                        duration_years=duration,
                                        lock_time=t,
                                        unlock_time=t + duration * 365.25,
                                        cohort=cohort_name
                                    ))
                        else:
                            # Fallback: single position at chosen duration
                            cohort_duration = lock_demand.chosen_duration_by_cohort.get(
                                cohort_name, avg_duration_years
                            )
                            lock_positions.append(LockPosition(
                                amount=cohort_amount,
                                duration_years=cohort_duration,
                                lock_time=t,
                                unlock_time=t + cohort_duration * 365.25,
                                cohort=cohort_name
                            ))

            # Update cohort states with effective unlocks (not raw unlocks)
            self.adoption_model.update_cohort_states(
                scaled_lock_demand, effective_unlocks_by_cohort, new_state.circulating
            )

            # Recompute effective weighted locked after adding new positions
            effective_weighted_locked_after = self.yield_source.compute_effective_weighted_locked(
                lock_positions
            )

            # Compute metrics including new fields (use post-lock ewl)
            metrics = self._compute_metrics(
                new_state,
                reward_budget,
                new_locks_demand,
                effective_unlocks,
                effective_weighted_locked_after,
                lock_positions,
                relocks=relocks_total,
                policy_active=policy_active,
                policy_reserve_rate=self._policy_state["reserve_rate"],
                policy_apy_floor=policy_apy_floor,
                lock_growth_annual=lock_growth_annual,
            )
            metrics_over_time.append(metrics)

        # Final metrics
        final_metrics = self._compute_final_metrics(
            latest_metrics=metrics_over_time[-1] if metrics_over_time else None,
            lock_positions=lock_positions
        )

        return SimulationResult(
            config=self.config,
            states=states,
            metrics_over_time=metrics_over_time,
            final_metrics=final_metrics,
            conservation_errors=self._conservation_errors,
            solver_warnings=self._solver_warnings
        )

    def _create_corrected_state(self, flows: Flows, dt_days: float) -> SystemState:
        """Create a corrected state when conservation is violated."""
        prev = self.accounting.current_state

        new_reserve = prev.reserve - flows.emission
        new_locked = prev.locked_vefil + flows.net_locks - flows.unlocks
        new_lending = prev.lending_pool - flows.net_lending_withdrawals - flows.lending_cannibalized
        new_sp_collateral = prev.sp_collateral + flows.sp_collateral_change
        new_burned = prev.burned_cumulative + flows.burns
        # other_allocations carried forward unchanged
        new_other_allocations = prev.other_allocations

        if new_lending < 0:
            new_lending = 0.0
        if new_sp_collateral < 0:
            new_sp_collateral = 0.0

        # Solve for circulating (max supply constant, subtract other_allocations)
        new_circulating = (
            prev.total_supply - new_locked - new_reserve -
            new_lending - new_sp_collateral - new_other_allocations - new_burned
        )

        corrected_state = SystemState(
            t=prev.t + dt_days,
            total_supply=prev.total_supply,
            circulating=new_circulating,
            locked_vefil=new_locked,
            reserve=new_reserve,
            lending_pool=new_lending,
            sp_collateral=new_sp_collateral,
            other_allocations=new_other_allocations,
            burned_cumulative=new_burned
        )

        self.accounting.current_state = corrected_state
        self.accounting.history.append(corrected_state)

        return corrected_state

    def _compute_unlocks(self, positions: List[LockPosition], current_time: float) -> float:
        """Compute total unlocks from expired positions."""
        total_unlocks = 0.0
        for pos in positions:
            if pos.unlock_time <= current_time:
                total_unlocks += pos.amount
        return total_unlocks

    def _compute_unlocks_by_cohort(
        self, positions: List[LockPosition], current_time: float
    ) -> Dict[str, float]:
        """Compute unlocks per cohort from expired positions."""
        unlocks = {}
        for pos in positions:
            if pos.unlock_time <= current_time:
                cohort = pos.cohort or "unknown"
                unlocks[cohort] = unlocks.get(cohort, 0.0) + pos.amount
        return unlocks

    def _compute_metrics(
        self,
        state: SystemState,
        reward_budget: RewardBudget,
        new_locks: float,
        unlocks: float,
        effective_weighted_locked: float,
        positions: List[LockPosition],
        relocks: float = 0.0,
        policy_active: bool = False,
        policy_reserve_rate: float = 0.03,
        policy_apy_floor: float = 0.0,
        lock_growth_annual: float = 0.0
    ) -> Dict[str, Any]:
        """Compute metrics for a timestep."""
        inflation_metrics = self.accounting.compute_inflation_metrics(
            self.config.simulation.timestep_days
        )

        # Compute total locked from positions for consistency check
        total_locked_positions = sum(p.amount for p in positions)

        return {
            't': state.t,
            'circulating': state.circulating,
            'locked': state.locked_vefil,
            'reserve': state.reserve,
            'burned_cumulative': state.burned_cumulative,
            'outstanding_supply': state.outstanding_supply,
            # Reward budget (reserve emissions only)
            'emission': reward_budget.reserve_emission,
            'total_reward_budget': reward_budget.total_reward_budget,
            # Lock flows
            'new_locks': new_locks,
            'relocks': relocks,
            'new_locks_total': new_locks + relocks,
            'unlocks': unlocks,
            # Yield curve metrics
            'effective_weighted_locked': effective_weighted_locked,
            'total_locked_positions': total_locked_positions,
            # Inflation metrics
            'net_inflation_rate': inflation_metrics['net_inflation_rate'],
            'gross_emission_rate': inflation_metrics['gross_emission_rate'],
            'effective_inflation': inflation_metrics['effective_inflation'],
            # Policy metrics
            'policy_active': policy_active,
            'policy_reserve_rate': policy_reserve_rate,
            'policy_apy_floor': policy_apy_floor,
            'lock_growth_annual': lock_growth_annual
        }

    def _compute_final_metrics(
        self,
        latest_metrics: Optional[Dict[str, Any]] = None,
        lock_positions: List[LockPosition] = None
    ) -> Dict[str, Any]:
        """Compute final summary metrics."""
        final_state = self.accounting.current_state
        runway_years = self.accounting.get_reserve_runway_years()
        dt_days = float(self.config.simulation.timestep_days)

        emission_latest = float(latest_metrics.get('emission', 0.0)) if latest_metrics else 0.0
        if dt_days > 0:
            annual_emission_latest = emission_latest * (365.25 / dt_days)
        else:
            annual_emission_latest = 0.0

        if annual_emission_latest > 0:
            lock_to_emission_ratio = final_state.locked_vefil / annual_emission_latest
        else:
            lock_to_emission_ratio = math.inf

        # Compute final effective weighted locked
        effective_weighted_locked = 0.0
        if lock_positions:
            effective_weighted_locked = self.yield_source.compute_effective_weighted_locked(
                lock_positions
            )

        return {
            'final_circulating': final_state.circulating,
            'final_locked': final_state.locked_vefil,
            'final_reserve': final_state.reserve,
            'final_burned': final_state.burned_cumulative,
            'final_outstanding': final_state.outstanding_supply,
            'reserve_runway_years': runway_years,
            'total_supply': final_state.total_supply,
            'annual_emission_latest': annual_emission_latest,
            'lock_to_emission_ratio': lock_to_emission_ratio,
            'effective_weighted_locked': effective_weighted_locked,
            'solver_warnings_count': len(self._solver_warnings)
        }
