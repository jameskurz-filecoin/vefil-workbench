"""Monte Carlo simulation for uncertainty analysis."""

import copy
from typing import Any, Callable, Dict, List

import numpy as np

from ..config.schema import Config
from .runner import SimulationResult, SimulationRunner


class MonteCarloRunner:
    """Run Monte Carlo simulations with parameter uncertainty."""

    def __init__(
        self,
        config: Config,
        parameter_distributions: Dict[str, Callable] = None
    ):
        """
        Initialize Monte Carlo runner.

        Args:
            config: Base configuration
            parameter_distributions: Dict mapping parameter paths to distribution functions
                Each function should take no arguments and return a sampled value.
                Example: {'network_growth_rate': lambda: np.random.normal(0.15, 0.05)}
        """
        self.config = config
        self.parameter_distributions = parameter_distributions or self._default_distributions()

    def run(
        self,
        num_runs: int = None,
        random_seed: int = None
    ) -> List[SimulationResult]:
        """
        Run Monte Carlo simulation.

        Args:
            num_runs: Number of runs (defaults to config value)
            random_seed: Random seed (defaults to config value)

        Returns:
            List of simulation results
        """
        if num_runs is None:
            num_runs = self.config.simulation.monte_carlo_runs

        if random_seed is None:
            random_seed = self.config.simulation.random_seed

        results = []

        for run_idx in range(num_runs):
            # Sample parameters
            sampled_config = self._sample_config(random_seed + run_idx)

            # Run simulation
            runner = SimulationRunner(sampled_config)
            result = runner.run(random_seed=random_seed + run_idx)
            results.append(result)

        return results

    def _default_distributions(self) -> Dict[str, Callable]:
        """
        Create default parameter distributions based on base config.

        Includes expanded parameter set for P3 rigor requirements:
        - Market multiplier
        - Participation elasticity
        - Cohort premiums
        - Hedging costs
        - Fee parameters

        Returns:
            Dict mapping parameter names to sampling functions
        """
        base = self.config

        return {
            # === Original parameters ===

            # Network growth with ±5% uncertainty
            'network_growth_rate': lambda: np.random.normal(
                base.externalities.network_growth_rate, 0.05
            ),

            # Alternative yields with ±2% uncertainty
            'ifil_apy': lambda: max(0.01, np.random.normal(
                base.alternatives.ifil_apy, 0.02
            )),
            'glif_apy': lambda: max(0.01, np.random.normal(
                base.alternatives.glif_apy, 0.02
            )),
            'defi_apy': lambda: max(0.01, np.random.normal(
                base.alternatives.defi_apy, 0.03
            )),

            # Capital flow fractions with ±10% relative uncertainty
            'net_new_fraction': lambda: np.clip(
                np.random.normal(base.capital_flow.net_new_fraction, 0.05),
                0.1, 0.7
            ),
            'recycled_fraction': lambda: np.clip(
                np.random.normal(base.capital_flow.recycled_fraction, 0.05),
                0.1, 0.7
            ),

            # Market volatility with ±20% relative uncertainty
            'volatility': lambda: max(0.2, np.random.normal(
                base.market.volatility, 0.12
            )),

            # Reserve annual rate with ±1% uncertainty
            'reserve_annual_rate': lambda: np.clip(
                np.random.normal(base.yield_source.reserve_annual_rate, 0.01),
                0.01, 0.15
            ),

            # Transaction volume with ±30% uncertainty
            'transaction_volume_base': lambda: max(100000, np.random.normal(
                base.network.transaction_volume_base, base.network.transaction_volume_base * 0.3
            )),

            # === NEW P3 Parameters: Market multiplier ===

            # Market multiplier (price impact) with ±25% relative uncertainty
            'market_multiplier': lambda: max(0.5, np.random.normal(
                base.market.market_multiplier, base.market.market_multiplier * 0.25
            )),

            # === NEW P3 Parameters: Participation elasticity ===

            # Participation elasticity with ±30% relative uncertainty
            'participation_elasticity': lambda: np.clip(
                np.random.normal(base.simulation.participation_elasticity, 0.45),
                0.5, 4.0
            ),

            # === NEW P3 Parameters: Cohort premiums ===

            # Retail required premium with ±30% relative uncertainty
            'retail_required_premium': lambda: np.clip(
                np.random.normal(base.cohorts.retail.required_premium, 0.024),
                0.02, 0.20
            ),

            # Institutional required premium with ±25% relative uncertainty
            'institutional_required_premium': lambda: np.clip(
                np.random.normal(base.cohorts.institutional.required_premium, 0.05),
                0.05, 0.40
            ),

            # Storage providers required premium with ±25% relative uncertainty
            'sp_required_premium': lambda: np.clip(
                np.random.normal(base.cohorts.storage_providers.required_premium, 0.03),
                0.03, 0.25
            ),

            # Treasuries required premium with ±30% relative uncertainty
            'treasuries_required_premium': lambda: np.clip(
                np.random.normal(base.cohorts.treasuries.required_premium, 0.018),
                0.01, 0.15
            ),

            # === NEW P3 Parameters: Hedging costs ===

            # Hedging funding rate with ±50% relative uncertainty
            'hedging_funding_rate': lambda: max(0.00001, np.random.normal(
                base.adversarial.hedging_funding_rate, base.adversarial.hedging_funding_rate * 0.5
            )),

            # Hedging borrow rate with ±50% relative uncertainty
            'hedging_borrow_rate': lambda: max(0.0001, np.random.normal(
                base.adversarial.hedging_borrow_rate, base.adversarial.hedging_borrow_rate * 0.5
            )),

            # === NEW P3 Parameters: Fee parameters ===

            # Fee multiplier with ±30% relative uncertainty
            'fee_multiplier': lambda: max(0.1, np.random.normal(
                base.yield_source.fee_multiplier, base.yield_source.fee_multiplier * 0.3
            )),

            # Fee per transaction with ±40% relative uncertainty
            'fee_per_transaction': lambda: max(0.0001, np.random.normal(
                base.network.fee_per_transaction, base.network.fee_per_transaction * 0.4
            )),
        }

    def _sample_config(self, seed: int) -> Config:
        """
        Sample a configuration from parameter distributions.

        Args:
            seed: Random seed for this sample

        Returns:
            New config with sampled parameters
        """
        np.random.seed(seed)

        # Deep copy the base config
        sampled_config = copy.deepcopy(self.config)

        # Sample each parameter
        for param_name, sample_func in self.parameter_distributions.items():
            sampled_value = sample_func()

            # Update the config based on parameter name
            # === Original parameters ===
            if param_name == 'network_growth_rate':
                sampled_config.externalities.network_growth_rate = sampled_value
            elif param_name == 'ifil_apy':
                sampled_config.alternatives.ifil_apy = sampled_value
            elif param_name == 'glif_apy':
                # GLIF emits iFIL receipts; keep GLIF aligned with iFIL.
                sampled_config.alternatives.glif_apy = sampled_config.alternatives.ifil_apy
            elif param_name == 'defi_apy':
                sampled_config.alternatives.defi_apy = sampled_value
            elif param_name == 'net_new_fraction':
                # Adjust other fractions to maintain sum ≤ 1
                sampled_config.capital_flow.net_new_fraction = sampled_value
                remaining = 1.0 - sampled_value
                # Split remaining between recycled and cannibalized proportionally
                total_other = sampled_config.capital_flow.recycled_fraction + sampled_config.capital_flow.cannibalized_fraction
                if total_other > 0:
                    scale = min(1.0, remaining / total_other)
                    sampled_config.capital_flow.recycled_fraction *= scale
                    sampled_config.capital_flow.cannibalized_fraction *= scale
            elif param_name == 'recycled_fraction':
                sampled_config.capital_flow.recycled_fraction = sampled_value
                # Ensure sum ≤ 1
                total = (sampled_config.capital_flow.net_new_fraction +
                        sampled_value +
                        sampled_config.capital_flow.cannibalized_fraction)
                if total > 1.0:
                    sampled_config.capital_flow.cannibalized_fraction = max(0, 1.0 - sampled_config.capital_flow.net_new_fraction - sampled_value)
            elif param_name == 'volatility':
                sampled_config.market.volatility = sampled_value
            elif param_name == 'reserve_annual_rate':
                sampled_config.yield_source.reserve_annual_rate = sampled_value
            elif param_name == 'transaction_volume_base':
                sampled_config.network.transaction_volume_base = sampled_value

            # === NEW P3 Parameters ===
            elif param_name == 'market_multiplier':
                sampled_config.market.market_multiplier = sampled_value
            elif param_name == 'participation_elasticity':
                sampled_config.simulation.participation_elasticity = sampled_value
            elif param_name == 'retail_required_premium':
                sampled_config.cohorts.retail.required_premium = sampled_value
            elif param_name == 'institutional_required_premium':
                sampled_config.cohorts.institutional.required_premium = sampled_value
            elif param_name == 'sp_required_premium':
                sampled_config.cohorts.storage_providers.required_premium = sampled_value
            elif param_name == 'treasuries_required_premium':
                sampled_config.cohorts.treasuries.required_premium = sampled_value
            elif param_name == 'hedging_funding_rate':
                sampled_config.adversarial.hedging_funding_rate = sampled_value
            elif param_name == 'hedging_borrow_rate':
                sampled_config.adversarial.hedging_borrow_rate = sampled_value
            elif param_name == 'fee_multiplier':
                sampled_config.yield_source.fee_multiplier = sampled_value
            elif param_name == 'fee_per_transaction':
                sampled_config.network.fee_per_transaction = sampled_value

        # Enforce GLIF/iFIL alignment after all sampling passes.
        sampled_config.alternatives.glif_apy = sampled_config.alternatives.ifil_apy

        return sampled_config

    def analyze_results(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """
        Analyze Monte Carlo results.

        Args:
            results: List of simulation results

        Returns:
            Statistical analysis
        """
        if not results:
            return {}

        # Extract final metrics
        final_circulating = [r.final_metrics['final_circulating'] for r in results]
        final_locked = [r.final_metrics['final_locked'] for r in results]
        final_reserve = [r.final_metrics['final_reserve'] for r in results]
        runway_years = [r.final_metrics['reserve_runway_years'] for r in results]

        def compute_stats(values: List[float]) -> Dict[str, float]:
            """Compute statistical summary."""
            return {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'p5': np.percentile(values, 5),
                'p25': np.percentile(values, 25),
                'p50': np.percentile(values, 50),
                'p75': np.percentile(values, 75),
                'p95': np.percentile(values, 95)
            }

        return {
            'num_runs': len(results),
            'final_circulating': compute_stats(final_circulating),
            'final_locked': compute_stats(final_locked),
            'final_reserve': compute_stats(final_reserve),
            'reserve_runway_years': compute_stats(runway_years)
        }

    def compute_run_confidence(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """
        Compute run confidence summary based on Monte Carlo dispersion.

        This provides a qualitative assessment of result reliability based on
        the spread of outcomes across Monte Carlo runs.

        Args:
            results: List of simulation results

        Returns:
            Confidence assessment with metrics and qualitative ratings
        """
        if not results or len(results) < 2:
            return {
                'confidence_level': 'insufficient_data',
                'message': 'Insufficient Monte Carlo runs for confidence assessment',
                'metrics': {}
            }

        stats = self.analyze_results(results)

        # Compute coefficient of variation (CV) for key metrics
        cv_locked = stats['final_locked']['std'] / max(stats['final_locked']['mean'], 1e-6)
        cv_reserve = stats['final_reserve']['std'] / max(stats['final_reserve']['mean'], 1e-6)
        cv_runway = stats['reserve_runway_years']['std'] / max(stats['reserve_runway_years']['mean'], 1e-6)

        # P5 to P95 range as fraction of median
        range_locked = (stats['final_locked']['p95'] - stats['final_locked']['p5']) / max(stats['final_locked']['p50'], 1e-6)
        range_reserve = (stats['final_reserve']['p95'] - stats['final_reserve']['p5']) / max(stats['final_reserve']['p50'], 1e-6)

        # Average CV across key metrics
        avg_cv = (cv_locked + cv_reserve + cv_runway) / 3

        # Determine confidence level based on dispersion
        if avg_cv < 0.1:
            confidence_level = 'high'
            confidence_message = 'Results are tightly clustered; high confidence in point estimates'
        elif avg_cv < 0.25:
            confidence_level = 'medium'
            confidence_message = 'Moderate outcome variance; consider the range of scenarios'
        elif avg_cv < 0.5:
            confidence_level = 'low'
            confidence_message = 'High outcome variance; results are sensitive to assumptions'
        else:
            confidence_level = 'very_low'
            confidence_message = 'Very high dispersion; results are highly uncertain'

        # Flag specific concerns
        flags = []
        if cv_runway > 0.3:
            flags.append('Reserve runway shows high sensitivity to assumptions')
        if cv_locked > 0.4:
            flags.append('Final locked supply is highly variable')
        if range_reserve > 1.0:
            flags.append('Reserve depletion outcomes vary widely (P95 differs from P5 by >100%)')

        return {
            'confidence_level': confidence_level,
            'message': confidence_message,
            'flags': flags,
            'metrics': {
                'cv_final_locked': cv_locked,
                'cv_final_reserve': cv_reserve,
                'cv_runway_years': cv_runway,
                'avg_cv': avg_cv,
                'p5_p95_range_locked': range_locked,
                'p5_p95_range_reserve': range_reserve,
                'num_runs': len(results)
            }
        }
