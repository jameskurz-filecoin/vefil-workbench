"""Predefined scenario library for veFIL simulation comparison."""

import copy
from dataclasses import dataclass
from typing import Any, Dict, List

from ..config.schema import Config
from ..simulation.runner import SimulationResult, SimulationRunner


@dataclass
class Scenario:
    """A named scenario with configuration overrides."""
    name: str
    description: str
    category: str  # "yield_source", "market", "stress_test", "calibration"
    overrides: Dict[str, Any]  # Config path -> value


@dataclass
class ScenarioComparison:
    """Result of comparing multiple scenarios."""
    scenarios: Dict[str, Scenario]
    results: Dict[str, SimulationResult]
    summary: Dict[str, Dict[str, Any]]  # scenario_name -> metrics summary


# ============================================================================
# PREDEFINED SCENARIOS
# ============================================================================

SCENARIO_LIBRARY = {
    # === Adoption Outlook Scenarios ===
    "conservative": Scenario(
        name="Conservative",
        description="Lower adoption: higher required premiums, smaller addressable market, lower relock rates (~82M @ 12mo, ~109M @ 5yr)",
        category="calibration",
        overrides={
            # Higher required premiums (harder to convince people to lock)
            "cohorts.retail.required_premium": 0.04,  # Was 0.02
            "cohorts.institutional.required_premium": 0.10,  # Was 0.06
            "cohorts.storage_providers.required_premium": 0.07,  # Was 0.04
            "cohorts.treasuries.required_premium": 0.03,  # Was 0.015
            # Smaller addressable market
            "simulation.addressable_cap": 0.12,  # Was 0.17
            # Lower relock rate
            "simulation.relock_fraction_unlocked": 0.52,  # Was 0.70
            # Slower adoption response
            "simulation.participation_elasticity": 2.2,  # Was 2.8
        }
    ),

    "optimistic": Scenario(
        name="Optimistic",
        description="Higher adoption: lower required premiums, larger addressable market, higher relock rates (~156M @ 12mo, ~194M @ 5yr)",
        category="calibration",
        overrides={
            # Lower required premiums (easier to convince people to lock)
            "cohorts.retail.required_premium": 0.01,  # Was 0.02
            "cohorts.institutional.required_premium": 0.04,  # Was 0.06
            "cohorts.storage_providers.required_premium": 0.025,  # Was 0.04
            "cohorts.treasuries.required_premium": 0.01,  # Was 0.015
            # Larger addressable market
            "simulation.addressable_cap": 0.25,  # Was 0.17
            # Higher relock rate
            "simulation.relock_fraction_unlocked": 0.85,  # Was 0.70
            # Faster adoption response
            "simulation.participation_elasticity": 3.5,  # Was 2.8
        }
    ),

    # === Reward Curve Scenarios ===
    "linear_rewards": Scenario(
        name="Linear Rewards (k=1)",
        description="Linear duration-weight relationship (k=1)",
        category="calibration",
        overrides={
            "reward_curve.k": 1.0,
        }
    ),

    "convex_rewards": Scenario(
        name="Convex Rewards (k=2)",
        description="Strongly convex curve favoring long locks (k=2)",
        category="calibration",
        overrides={
            "reward_curve.k": 2.0,
        }
    ),

    "short_max_duration": Scenario(
        name="Short Max Duration (3yr)",
        description="Cap maximum lock duration at 3 years",
        category="calibration",
        overrides={
            "reward_curve.max_duration_years": 3.0,
        }
    ),

    "long_max_duration": Scenario(
        name="Long Max Duration (7yr)",
        description="Allow lock durations up to 7 years",
        category="calibration",
        overrides={
            "reward_curve.max_duration_years": 7.0,
        }
    ),

    # === Market Condition Scenarios (secondary factors) ===
    "bull_market": Scenario(
        name="Bull Market Conditions",
        description="High net-new capital inflow, low volatility, strong network growth (secondary overlay)",
        category="market",
        overrides={
            "capital_flow.net_new_fraction": 0.6,
            "capital_flow.recycled_fraction": 0.3,
            "capital_flow.cannibalized_fraction": 0.1,
            "market.volatility": 0.4,
            "externalities.network_growth_rate": 0.25,
        }
    ),

    "bear_market": Scenario(
        name="Bear Market Conditions",
        description="Low net-new capital, high volatility, weak network growth (secondary overlay)",
        category="market",
        overrides={
            "capital_flow.net_new_fraction": 0.2,
            "capital_flow.recycled_fraction": 0.4,
            "capital_flow.cannibalized_fraction": 0.4,
            "market.volatility": 0.8,
            "externalities.network_growth_rate": 0.05,
        }
    ),

    # === Stress Test Scenarios ===
    "reserve_exhaustion_cliff": Scenario(
        name="Reserve Exhaustion Cliff",
        description="High emission + long horizon to test reserve depletion behavior",
        category="stress_test",
        overrides={
            "yield_source.reserve_annual_rate": 0.10,
            "simulation.time_horizon_months": 120,  # 10 years
        }
    ),

    "high_competition": Scenario(
        name="High Alternative Competition",
        description="Alternative yields spike (iFIL 15%, DeFi 20%)",
        category="stress_test",
        overrides={
            "alternatives.ifil_apy": 0.15,
            "alternatives.glif_apy": 0.15,
            "alternatives.defi_apy": 0.20,
        }
    ),

    # === Calibration Scenarios ===
    "sp_focused": Scenario(
        name="SP-Focused Design",
        description="Parameters tuned for Storage Provider participation",
        category="calibration",
        overrides={
            "cohorts.storage_providers.size_fraction": 0.4,
            "cohorts.storage_providers.required_premium": 0.08,
            "cohorts.retail.size_fraction": 0.3,
            "cohorts.institutional.size_fraction": 0.2,
            "cohorts.treasuries.size_fraction": 0.1,
        }
    ),

    "retail_focused": Scenario(
        name="Retail-Focused Design",
        description="Parameters tuned for retail participant accessibility",
        category="calibration",
        overrides={
            "cohorts.retail.size_fraction": 0.6,
            "cohorts.retail.required_premium": 0.05,
            "reward_curve.min_duration_years": 0.0833,  # 1 month minimum
            "reward_curve.k": 1.2,  # Less convex to help short lockers
        }
    ),
}


class ScenarioRunner:
    """Run and compare predefined scenarios."""

    def __init__(self, base_config: Config):
        """
        Initialize scenario runner.

        Args:
            base_config: Base configuration to apply overrides to
        """
        self.base_config = base_config

    def get_available_scenarios(self) -> Dict[str, Scenario]:
        """Get all available scenarios."""
        return SCENARIO_LIBRARY.copy()

    def get_scenarios_by_category(self, category: str) -> Dict[str, Scenario]:
        """Get scenarios filtered by category."""
        return {
            name: scenario
            for name, scenario in SCENARIO_LIBRARY.items()
            if scenario.category == category
        }

    def apply_scenario(self, scenario: Scenario) -> Config:
        """
        Apply scenario overrides to base config.

        Args:
            scenario: Scenario with overrides

        Returns:
            Modified config
        """
        config = copy.deepcopy(self.base_config)

        for path, value in scenario.overrides.items():
            self._set_config_value(config, path, value)

        # GLIF emits iFIL receipts; keep GLIF aligned with iFIL.
        config.alternatives.glif_apy = config.alternatives.ifil_apy

        return config

    def run_scenario(
        self,
        scenario_name: str,
        random_seed: int = None
    ) -> SimulationResult:
        """
        Run a single scenario.

        Args:
            scenario_name: Name of scenario from library
            random_seed: Random seed for reproducibility

        Returns:
            Simulation result
        """
        if scenario_name not in SCENARIO_LIBRARY:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        scenario = SCENARIO_LIBRARY[scenario_name]
        config = self.apply_scenario(scenario)

        runner = SimulationRunner(config)
        seed = random_seed or config.simulation.random_seed
        return runner.run(random_seed=seed)

    def compare_scenarios(
        self,
        scenario_names: List[str],
        include_base: bool = True,
        random_seed: int = None
    ) -> ScenarioComparison:
        """
        Run and compare multiple scenarios.

        Args:
            scenario_names: List of scenario names to compare
            include_base: Whether to include base case
            random_seed: Random seed for reproducibility

        Returns:
            ScenarioComparison result
        """
        seed = random_seed or self.base_config.simulation.random_seed
        scenarios = {}
        results = {}
        summary = {}

        # Run base case if requested
        if include_base:
            base_scenario = Scenario(
                name="Base Case",
                description="Default configuration without modifications",
                category="base",
                overrides={}
            )
            scenarios["base"] = base_scenario

            runner = SimulationRunner(self.base_config)
            results["base"] = runner.run(random_seed=seed)
            summary["base"] = self._extract_summary(results["base"])

        # Run each requested scenario
        for name in scenario_names:
            if name not in SCENARIO_LIBRARY:
                continue

            scenarios[name] = SCENARIO_LIBRARY[name]
            results[name] = self.run_scenario(name, seed)
            summary[name] = self._extract_summary(results[name])

        return ScenarioComparison(
            scenarios=scenarios,
            results=results,
            summary=summary
        )

    def compare_market_conditions(self, random_seed: int = None) -> ScenarioComparison:
        """Compare bull vs bear market conditions."""
        return self.compare_scenarios(
            ["bull_market", "bear_market"],
            include_base=True,
            random_seed=random_seed
        )

    def compare_adoption_outlooks(self, random_seed: int = None) -> ScenarioComparison:
        """Compare conservative vs optimistic adoption scenarios."""
        return self.compare_scenarios(
            ["conservative", "optimistic"],
            include_base=True,
            random_seed=random_seed
        )

    def run_stress_tests(self, random_seed: int = None) -> ScenarioComparison:
        """Run all stress test scenarios."""
        stress_scenarios = list(self.get_scenarios_by_category("stress_test").keys())
        return self.compare_scenarios(
            stress_scenarios,
            include_base=True,
            random_seed=random_seed
        )

    def _extract_summary(self, result: SimulationResult) -> Dict[str, Any]:
        """Extract key metrics summary from simulation result."""
        final_metrics = result.final_metrics

        # Get final effective inflation
        effective_inflation = 0.0
        if result.metrics_over_time:
            effective_inflation = result.metrics_over_time[-1].get('effective_inflation', 0)

        # Compute locked supply share
        total_supply = final_metrics.get('total_supply', 1)
        locked_share = final_metrics.get('final_locked', 0) / total_supply if total_supply > 0 else 0

        return {
            'final_locked': final_metrics.get('final_locked', 0),
            'final_locked_share': locked_share,
            'final_reserve': final_metrics.get('final_reserve', 0),
            'reserve_runway_years': final_metrics.get('reserve_runway_years', 0),
            'final_circulating': final_metrics.get('final_circulating', 0),
            'effective_inflation': effective_inflation,
            'conservation_errors': len(result.conservation_errors) if result.conservation_errors else 0
        }

    def _set_config_value(self, config: Config, path: str, value: Any) -> None:
        """Set a value in config using dot-notation path."""
        parts = path.split('.')
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)


def format_comparison_table(comparison: ScenarioComparison) -> str:
    """
    Format scenario comparison as a text table.

    Args:
        comparison: ScenarioComparison result

    Returns:
        Formatted table string
    """
    lines = []
    headers = ["Scenario", "Locked (M)", "Locked %", "Reserve (M)", "Runway (yr)", "Eff. Infl."]
    lines.append(" | ".join(f"{h:>12}" for h in headers))
    lines.append("-" * 85)

    for name, summary in comparison.summary.items():
        scenario = comparison.scenarios.get(name)
        display_name = scenario.name if scenario else name

        row = [
            f"{display_name[:12]:>12}",
            f"{summary['final_locked']/1e6:>12,.1f}",
            f"{summary['final_locked_share']*100:>11.1f}%",
            f"{summary['final_reserve']/1e6:>12,.1f}",
            f"{summary['reserve_runway_years']:>12.1f}",
            f"{summary['effective_inflation']*100:>11.2f}%"
        ]
        lines.append(" | ".join(row))

    return "\n".join(lines)
