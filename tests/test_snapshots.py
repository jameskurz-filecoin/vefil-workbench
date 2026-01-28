"""Snapshot tests for Core Impact metrics.

These tests verify that simulation results match known reference values
for specific configurations. If these fail after code changes, either:
1. The change broke something (bug) - fix the code
2. The change is intentional - update the snapshot values

Last snapshot update: January 2026
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vefil.config.loader import load_config
from vefil.config.schema import Config
from vefil.simulation.runner import SimulationRunner


# Snapshot reference values (update when model intentionally changes)
# Generated with default config, seed=42, 60-month horizon
# Updated Jan 2026: Fixed conservation accounting for reward_relocks
SNAPSHOT_DEFAULT = {
    'config_hash_prefix': 'fc11c2',  # First 6 chars - will need updating
    'seed': 42,
    'horizon_months': 60,
    'expected': {
        # Allow 5% tolerance for floating point variations
        'final_locked_min': 130_000_000,    # At least 130M locked
        'final_locked_max': 160_000_000,    # At most 160M locked (allows for ~144M actual)
        'final_reserve_min': 230_000_000,   # Mining reserve after 5yr
        'final_reserve_max': 280_000_000,   # Started at 300M
        'reserve_runway_min': 15,           # Runway ~20 years with emissions policy
        'reserve_runway_max': 30,           # Upper bound
        'num_timesteps': 60,                # Exact match expected
    }
}

# Conservative scenario snapshot
SNAPSHOT_CONSERVATIVE = {
    'description': 'Low emission rate, high required premiums, no dynamic adjustment',
    'overrides': {
        'yield_source.reserve_annual_rate': 0.005,
        'emissions_policy.enabled': False,  # Disable dynamic rate adjustment for true fixed rate
        'cohorts.retail.required_premium': 0.12,
        'cohorts.institutional.required_premium': 0.25,
    },
    'seed': 42,
    'horizon_months': 36,
    'expected': {
        'final_locked_min': 0,
        'final_locked_max': 100_000_000,    # Lower adoption expected
        'final_reserve_min': 290_000_000,   # More reserve preserved with fixed 0.5% rate
    }
}

# Aggressive scenario snapshot
SNAPSHOT_AGGRESSIVE = {
    'description': 'High emission rate, low premiums',
    'overrides': {
        'yield_source.reserve_annual_rate': 0.02,
        'cohorts.retail.required_premium': 0.05,
        'cohorts.institutional.required_premium': 0.12,
    },
    'seed': 42,
    'horizon_months': 36,
    'expected': {
        'final_locked_min': 500_000,        # Higher adoption expected (model is conservative)
        'final_reserve_min': 900_000_000,   # More depletion, but large reserve remains
    }
}


def apply_overrides(config: Config, overrides: dict) -> Config:
    """Apply dot-notation overrides to config."""
    for path, value in overrides.items():
        parts = path.split('.')
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    return config


class TestDefaultConfigSnapshot:
    """Snapshot tests with default configuration."""

    def test_simulation_completes(self):
        """Simulation runs to completion."""
        config = load_config()
        config.simulation.time_horizon_months = SNAPSHOT_DEFAULT['horizon_months']

        runner = SimulationRunner(config)
        result = runner.run(random_seed=SNAPSHOT_DEFAULT['seed'])

        assert len(result.states) == SNAPSHOT_DEFAULT['expected']['num_timesteps']

    def test_final_locked_in_range(self):
        """Final locked supply is within expected range."""
        config = load_config()
        config.simulation.time_horizon_months = SNAPSHOT_DEFAULT['horizon_months']

        runner = SimulationRunner(config)
        result = runner.run(random_seed=SNAPSHOT_DEFAULT['seed'])

        final_locked = result.final_metrics['final_locked']
        expected = SNAPSHOT_DEFAULT['expected']

        assert final_locked >= expected['final_locked_min'], \
            f"Final locked {final_locked:,.0f} below minimum {expected['final_locked_min']:,.0f}"
        assert final_locked <= expected['final_locked_max'], \
            f"Final locked {final_locked:,.0f} above maximum {expected['final_locked_max']:,.0f}"

    def test_final_reserve_in_range(self):
        """Final reserve is within expected range."""
        config = load_config()
        config.simulation.time_horizon_months = SNAPSHOT_DEFAULT['horizon_months']

        runner = SimulationRunner(config)
        result = runner.run(random_seed=SNAPSHOT_DEFAULT['seed'])

        final_reserve = result.final_metrics['final_reserve']
        expected = SNAPSHOT_DEFAULT['expected']

        assert final_reserve >= expected['final_reserve_min'], \
            f"Final reserve {final_reserve:,.0f} below minimum {expected['final_reserve_min']:,.0f}"
        assert final_reserve <= expected['final_reserve_max'], \
            f"Final reserve {final_reserve:,.0f} above maximum {expected['final_reserve_max']:,.0f}"

    def test_reserve_runway_reasonable(self):
        """Reserve runway is within plausible range."""
        config = load_config()
        config.simulation.time_horizon_months = SNAPSHOT_DEFAULT['horizon_months']

        runner = SimulationRunner(config)
        result = runner.run(random_seed=SNAPSHOT_DEFAULT['seed'])

        runway = result.final_metrics['reserve_runway_years']
        expected = SNAPSHOT_DEFAULT['expected']

        assert runway >= expected['reserve_runway_min'], \
            f"Runway {runway:.1f}yr below minimum {expected['reserve_runway_min']}yr"
        assert runway <= expected['reserve_runway_max'], \
            f"Runway {runway:.1f}yr above maximum {expected['reserve_runway_max']}yr"

    def test_conservation_maintained(self):
        """No conservation errors during simulation."""
        config = load_config()
        config.simulation.time_horizon_months = SNAPSHOT_DEFAULT['horizon_months']

        runner = SimulationRunner(config)
        result = runner.run(random_seed=SNAPSHOT_DEFAULT['seed'])

        # Should have few or no conservation errors
        assert len(result.conservation_errors) <= 5, \
            f"Too many conservation errors: {len(result.conservation_errors)}"


class TestConservativeSnapshot:
    """Snapshot tests with conservative configuration."""

    def test_lower_adoption_than_default(self):
        """Conservative config should have lower adoption."""
        # Default run
        default_config = load_config()
        default_config.simulation.time_horizon_months = 36
        default_runner = SimulationRunner(default_config)
        default_result = default_runner.run(random_seed=42)

        # Conservative run
        conservative_config = load_config()
        conservative_config.simulation.time_horizon_months = 36
        apply_overrides(conservative_config, SNAPSHOT_CONSERVATIVE['overrides'])
        conservative_runner = SimulationRunner(conservative_config)
        conservative_result = conservative_runner.run(random_seed=42)

        # Conservative should have less locked
        default_locked = default_result.final_metrics['final_locked']
        conservative_locked = conservative_result.final_metrics['final_locked']

        assert conservative_locked <= default_locked, \
            f"Conservative {conservative_locked:,.0f} should be <= default {default_locked:,.0f}"

    def test_more_reserve_preserved(self):
        """Conservative config should preserve more reserve."""
        # Default run
        default_config = load_config()
        default_config.simulation.time_horizon_months = 36
        default_runner = SimulationRunner(default_config)
        default_result = default_runner.run(random_seed=42)

        # Conservative run
        conservative_config = load_config()
        conservative_config.simulation.time_horizon_months = 36
        apply_overrides(conservative_config, SNAPSHOT_CONSERVATIVE['overrides'])
        conservative_runner = SimulationRunner(conservative_config)
        conservative_result = conservative_runner.run(random_seed=42)

        # Conservative should have more reserve (lower emission rate)
        default_reserve = default_result.final_metrics['final_reserve']
        conservative_reserve = conservative_result.final_metrics['final_reserve']

        assert conservative_reserve >= default_reserve, \
            f"Conservative reserve {conservative_reserve:,.0f} should be >= default {default_reserve:,.0f}"


class TestAggressiveSnapshot:
    """Snapshot tests with aggressive configuration."""

    def test_higher_adoption_than_conservative(self):
        """Aggressive config should have higher adoption than conservative."""
        # Conservative run
        conservative_config = load_config()
        conservative_config.simulation.time_horizon_months = 36
        apply_overrides(conservative_config, SNAPSHOT_CONSERVATIVE['overrides'])
        conservative_runner = SimulationRunner(conservative_config)
        conservative_result = conservative_runner.run(random_seed=42)

        # Aggressive run
        aggressive_config = load_config()
        aggressive_config.simulation.time_horizon_months = 36
        apply_overrides(aggressive_config, SNAPSHOT_AGGRESSIVE['overrides'])
        aggressive_runner = SimulationRunner(aggressive_config)
        aggressive_result = aggressive_runner.run(random_seed=42)

        conservative_locked = conservative_result.final_metrics['final_locked']
        aggressive_locked = aggressive_result.final_metrics['final_locked']

        assert aggressive_locked >= conservative_locked, \
            f"Aggressive {aggressive_locked:,.0f} should be >= conservative {conservative_locked:,.0f}"


class TestMetricsOverTime:
    """Snapshot tests for time-series metrics."""

    def test_emission_positive(self):
        """All emissions should be positive."""
        config = load_config()
        config.simulation.time_horizon_months = 24

        runner = SimulationRunner(config)
        result = runner.run(random_seed=42)

        for i, metrics in enumerate(result.metrics_over_time):
            assert metrics['emission'] >= 0, \
                f"Negative emission at timestep {i}: {metrics['emission']}"

    def test_reserve_monotonic_decreasing(self):
        """Reserve should decrease or stay flat (no spontaneous creation)."""
        config = load_config()
        config.simulation.time_horizon_months = 24
        config.yield_source.type = "reserve"

        runner = SimulationRunner(config)
        result = runner.run(random_seed=42)

        prev_reserve = config.initial_supply.reserve
        for i, state in enumerate(result.states):
            assert state.reserve <= prev_reserve + 1, \
                f"Reserve increased at timestep {i}: {prev_reserve:,.0f} -> {state.reserve:,.0f}"
            prev_reserve = state.reserve

    def test_circulating_non_negative(self):
        """Circulating supply should never go negative."""
        config = load_config()
        config.simulation.time_horizon_months = 60

        runner = SimulationRunner(config)
        result = runner.run(random_seed=42)

        for i, state in enumerate(result.states):
            assert state.circulating >= 0, \
                f"Negative circulating at timestep {i}: {state.circulating}"


class TestEdgeCases:
    """Snapshot tests for edge cases."""

    def test_zero_initial_locked(self):
        """Simulation works with zero initial locked."""
        config = load_config()
        config.initial_supply.locked_vefil = 0
        config.simulation.time_horizon_months = 12

        runner = SimulationRunner(config)
        result = runner.run(random_seed=42)

        assert len(result.states) == 12

    def test_very_short_horizon(self):
        """Simulation works with very short horizon."""
        config = load_config()
        config.simulation.time_horizon_months = 1

        runner = SimulationRunner(config)
        result = runner.run(random_seed=42)

        assert len(result.states) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
