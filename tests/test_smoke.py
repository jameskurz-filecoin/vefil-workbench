"""Smoke tests for core veFIL modules.

These tests verify basic functionality without deep validation.
Run these first to catch obvious breakage.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vefil.config.loader import load_config
from vefil.config.schema import Config
from vefil.simulation.runner import SimulationRunner, SimulationResult
from vefil.engine.accounting import AccountingEngine, SystemState, Flows
from vefil.engine.rewards import YieldSourceEngine


class TestConfigLoading:
    """Smoke tests for configuration loading."""

    def test_load_default_config(self):
        """Config loads without errors."""
        config = load_config()
        assert config is not None
        assert isinstance(config, Config)

    def test_config_has_required_sections(self):
        """Config contains all expected sections."""
        config = load_config()
        assert hasattr(config, 'initial_supply')
        assert hasattr(config, 'yield_source')
        assert hasattr(config, 'reward_curve')
        assert hasattr(config, 'market')
        assert hasattr(config, 'capital_flow')
        assert hasattr(config, 'alternatives')
        assert hasattr(config, 'cohorts')
        assert hasattr(config, 'simulation')

    def test_config_hash_is_deterministic(self):
        """Same config produces same hash."""
        config1 = load_config()
        config2 = load_config()
        assert config1.compute_hash() == config2.compute_hash()

    def test_initial_supply_conservation(self):
        """Initial supply components sum correctly."""
        config = load_config()
        supply = config.initial_supply
        # Components should be <= total (some may be unallocated)
        component_sum = (
            supply.circulating +
            supply.reserve +
            supply.lending_pool +
            supply.sp_collateral +
            supply.locked_vefil
        )
        assert component_sum <= supply.total


class TestAccountingEngine:
    """Smoke tests for accounting engine."""

    def test_create_initial_state(self):
        """SystemState can be created."""
        state = SystemState(
            t=0.0,
            total_supply=2_000_000_000,
            circulating=600_000_000,
            locked_vefil=0,
            reserve=300_000_000,
            lending_pool=50_000_000,
            sp_collateral=100_000_000
        )
        assert state.total_supply == 2_000_000_000

    def test_conservation_validation(self):
        """Conservation check works."""
        state = SystemState(
            t=0.0,
            total_supply=1_050_000_000,
            circulating=600_000_000,
            locked_vefil=0,
            reserve=300_000_000,
            lending_pool=50_000_000,
            sp_collateral=100_000_000
        )
        is_valid, error = state.validate_conservation()
        assert is_valid is True

    def test_conservation_detects_violation(self):
        """Conservation check catches errors."""
        state = SystemState(
            t=0.0,
            total_supply=2_000_000_000,  # Much larger than components
            circulating=100,
            locked_vefil=0,
            reserve=100,
            lending_pool=0,
            sp_collateral=0
        )
        is_valid, error = state.validate_conservation()
        assert is_valid is False

    def test_accounting_engine_step(self):
        """AccountingEngine can process a step."""
        config = load_config()
        initial_state = SystemState(
            t=0.0,
            total_supply=config.initial_supply.total,
            circulating=config.initial_supply.circulating,
            locked_vefil=config.initial_supply.locked_vefil,
            reserve=config.initial_supply.reserve,
            lending_pool=config.initial_supply.lending_pool,
            sp_collateral=config.initial_supply.sp_collateral,
            other_allocations=config.initial_supply.other_allocations,
        )
        engine = AccountingEngine(initial_state=initial_state)

        flows = Flows(
            emission=1_000_000,
            emission_to_circulating=1_000_000,  # All emission goes to circulating (no relocks)
            net_locks=500_000,
            unlocks=0,
            lending_cannibalized=100_000,
            net_lending_withdrawals=0,
            reward_relocks=0,  # No reward relocks
        )

        new_state = engine.step(flows, dt=30)
        assert new_state.t == 30
        assert new_state.reserve < initial_state.reserve


class TestRewardsEngine:
    """Smoke tests for rewards/yield engine."""

    def test_compute_lock_weight(self):
        """Lock weight computation works."""
        config = load_config()
        engine = YieldSourceEngine(
            reserve_annual_rate=config.yield_source.reserve_annual_rate,
            reward_curve_k=config.reward_curve.k,
            max_duration_years=config.reward_curve.max_duration_years
        )

        weight_1yr = engine.compute_lock_weight(1.0)
        weight_5yr = engine.compute_lock_weight(5.0)

        # Longer duration should have higher weight
        assert weight_5yr > weight_1yr
        # Weight at max duration should be 1.0
        assert abs(weight_5yr - 1.0) < 0.01

    def test_weight_monotonic(self):
        """Longer locks always have higher weight (for k > 0)."""
        config = load_config()
        engine = YieldSourceEngine(
            reserve_annual_rate=config.yield_source.reserve_annual_rate,
            reward_curve_k=config.reward_curve.k,
            max_duration_years=config.reward_curve.max_duration_years
        )

        prev_weight = 0
        for duration in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]:
            weight = engine.compute_lock_weight(duration)
            assert weight >= prev_weight
            prev_weight = weight

    def test_compute_emission(self):
        """Emission computation produces positive value."""
        config = load_config()
        engine = YieldSourceEngine(
            reserve_annual_rate=0.05,
            reward_curve_k=1.5,
            max_duration_years=5.0
        )

        emission = engine.compute_emission(
            reserve_balance=300_000_000,
            network_fee_revenue=0,
            dt_days=30
        )

        assert emission > 0
        # Should be approximately 5% annual rate for 30 days
        expected_monthly = 300_000_000 * 0.05 * (30 / 365.25)
        assert abs(emission - expected_monthly) / expected_monthly < 0.01


class TestSimulationRunner:
    """Smoke tests for full simulation."""

    def test_run_simulation(self):
        """Simulation completes without errors."""
        config = load_config()
        # Use short horizon for speed
        config.simulation.time_horizon_months = 12
        config.simulation.monte_carlo_runs = 1

        runner = SimulationRunner(config)
        result = runner.run(random_seed=42)

        assert result is not None
        assert isinstance(result, SimulationResult)
        assert len(result.states) > 0
        assert len(result.metrics_over_time) > 0

    def test_simulation_deterministic(self):
        """Same seed produces same results."""
        config = load_config()
        config.simulation.time_horizon_months = 6

        runner1 = SimulationRunner(config)
        result1 = runner1.run(random_seed=42)

        runner2 = SimulationRunner(config)
        result2 = runner2.run(random_seed=42)

        assert result1.final_metrics['final_locked'] == result2.final_metrics['final_locked']
        assert result1.final_metrics['final_reserve'] == result2.final_metrics['final_reserve']

    def test_simulation_final_metrics(self):
        """Final metrics contain expected keys."""
        config = load_config()
        config.simulation.time_horizon_months = 6

        runner = SimulationRunner(config)
        result = runner.run(random_seed=42)

        assert 'final_circulating' in result.final_metrics
        assert 'final_locked' in result.final_metrics
        assert 'final_reserve' in result.final_metrics
        assert 'reserve_runway_years' in result.final_metrics

    def test_reserve_decreases_over_time(self):
        """Reserve should decrease as emissions occur."""
        config = load_config()
        config.simulation.time_horizon_months = 12
        config.yield_source.type = "reserve"
        config.yield_source.reserve_annual_rate = 0.05

        runner = SimulationRunner(config)
        result = runner.run(random_seed=42)

        initial_reserve = config.initial_supply.reserve
        final_reserve = result.final_metrics['final_reserve']

        assert final_reserve < initial_reserve


class TestAdoptionModel:
    """Smoke tests for behavioral adoption."""

    def test_adoption_model_import(self):
        """AdoptionModel can be imported."""
        from vefil.behavior.adoption import AdoptionModel, LockDemand
        assert AdoptionModel is not None

    def test_lock_demand_computation(self):
        """Lock demand can be computed."""
        from vefil.behavior.adoption import AdoptionModel
        from vefil.behavior.cohorts import Cohort
        from vefil.behavior.opportunity_cost import OpportunityCostCalculator, AlternativeYields

        alternatives = AlternativeYields(
            ifil_apy=0.10,
            glif_apy=0.10,
            defi_apy=0.12,
            risk_free_rate=0.04
        )

        opp_calc = OpportunityCostCalculator(
            alternatives=alternatives,
            volatility=0.6
        )

        cohorts = [
            Cohort(
                name="retail",
                size_fraction=0.5,
                required_premium=0.08,
                avg_lock_size=1000,
                avg_duration_years=2.0,
                risk_tolerance="medium"
            )
        ]

        model = AdoptionModel(
            cohorts=cohorts,
            opportunity_calc=opp_calc,
            participation_elasticity=1.5
        )

        def yield_curve(duration: float) -> float:
            return 0.15  # 15% APY

        demand = model.compute_lock_demand(yield_curve=yield_curve)

        assert demand.total_demand >= 0
        assert demand.participation_rate >= 0

    def test_glif_is_alias_of_ifil(self):
        """GLIF APY should align with iFIL since GLIF issues iFIL receipts."""
        from vefil.behavior.opportunity_cost import AlternativeYields

        alternatives = AlternativeYields(
            ifil_apy=0.11,
            glif_apy=0.07,  # Should be overridden to match iFIL
            defi_apy=0.09,
            risk_free_rate=0.04,
        )

        assert alternatives.glif_apy == alternatives.ifil_apy
        best_name, _ = alternatives.get_best_alternative()
        assert best_name != "glif"


class TestFinalMetrics:
    """Smoke tests for final metric diagnostics."""

    def test_lock_to_emission_ratio_metric(self):
        """Annualized emission and lock/emission ratio should be internally consistent."""
        from vefil.config.loader import load_config
        from vefil.simulation.runner import SimulationRunner

        config = load_config()
        runner = SimulationRunner(config)

        state = runner.accounting.current_state
        locked_amount = 50_000_000.0  # Test with 50M locked
        state.locked_vefil = locked_amount
        state.circulating = max(0.0, state.circulating - locked_amount)

        dt_days = float(config.simulation.timestep_days)
        emission_step = state.reserve * config.yield_source.reserve_annual_rate * (dt_days / 365.25)

        final = runner._compute_final_metrics(latest_metrics={"emission": emission_step})
        expected_annual = state.reserve * config.yield_source.reserve_annual_rate

        assert expected_annual > 0
        assert abs(final["annual_emission_latest"] - expected_annual) / expected_annual < 1e-9
        assert final["lock_to_emission_ratio"] > 0


class TestAnalysisModules:
    """Smoke tests for analysis modules."""

    def test_sensitivity_import(self):
        """Sensitivity analysis can be imported."""
        from vefil.analysis.sensitivity import SensitivityAnalyzer
        assert SensitivityAnalyzer is not None

    def test_scenarios_import(self):
        """Scenarios can be imported."""
        from vefil.analysis.scenarios import ScenarioRunner, SCENARIO_LIBRARY
        assert ScenarioRunner is not None
        assert len(SCENARIO_LIBRARY) > 0

    def test_calibration_import(self):
        """Calibration module can be imported."""
        from vefil.analysis.calibration import CalibrationPanel, MODEL_LIMITS
        assert CalibrationPanel is not None
        assert len(MODEL_LIMITS) > 0

    def test_monte_carlo_import(self):
        """Monte Carlo can be imported."""
        from vefil.simulation.monte_carlo import MonteCarloRunner
        assert MonteCarloRunner is not None


class TestValidation:
    """Smoke tests for validation module."""

    def test_validation_import(self):
        """Validation module can be imported."""
        from vefil.validation.sanity_checks import SanityChecker, ValidationWarning
        assert SanityChecker is not None

    def test_sanity_checker_runs(self):
        """SanityChecker can check a config."""
        from vefil.validation.sanity_checks import SanityChecker

        config = load_config()
        checker = SanityChecker(config)

        warnings = checker.check_config_inputs()
        assert isinstance(warnings, list)


class TestGuardrails:
    """Guardrail tests for consistency checks."""

    def test_cohort_locked_matches_accounting_locked(self):
        """Cohort-level locked sum approximately matches accounting locked.

        Note: With relocking, small discrepancies can occur due to timing
        of position expiration vs cohort state updates. Allow 1% tolerance.
        """
        config = load_config()
        runner = SimulationRunner(config)
        result = runner.run(random_seed=config.simulation.random_seed)
        cohort_locked = sum(float(c.state.locked_fil) for c in runner.adoption_model.cohorts)
        accounting_locked = result.final_metrics["final_locked"]
        # Allow 1% tolerance for relocking-related discrepancies
        assert abs(cohort_locked - accounting_locked) <= 0.01 * max(1.0, accounting_locked)

    def test_year1_locked_supply_hits_target_band(self):
        """Year-1 locked supply reaches target band."""
        config = load_config()
        runner = SimulationRunner(config)
        result = runner.run(random_seed=config.simulation.random_seed)
        # Monthly steps -> index 11 is end of month 12
        locked_year1 = result.states[11].locked_vefil
        assert locked_year1 >= 95_000_000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
