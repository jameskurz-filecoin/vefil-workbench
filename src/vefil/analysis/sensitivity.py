"""Sensitivity analysis for veFIL simulation parameters."""

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from ..config.schema import Config
from ..simulation.runner import SimulationRunner


@dataclass
class ParameterSweep:
    """Result of a single parameter sweep."""
    parameter_name: str
    parameter_label: str
    base_value: float
    sweep_values: List[float]
    metric_values: Dict[str, List[float]]  # metric_name -> values at each sweep point


@dataclass
class TornadoEntry:
    """Single entry in a tornado chart."""
    parameter_name: str
    parameter_label: str
    base_value: float
    low_value: float
    high_value: float
    metric_at_low: float
    metric_at_high: float
    impact_range: float  # high - low metric value


@dataclass
class SensitivityResult:
    """Complete sensitivity analysis result."""
    base_config: Config
    base_metrics: Dict[str, Any]
    sweeps: Dict[str, ParameterSweep]
    tornado_data: Dict[str, List[TornadoEntry]]  # metric_name -> sorted entries


class SensitivityAnalyzer:
    """Perform sensitivity analysis on simulation parameters."""

    # Default parameters to analyze with their labels and relative variation
    DEFAULT_PARAMETERS = {
        # (config_path, label, low_mult, high_mult)
        'reserve_annual_rate': ('yield_source.reserve_annual_rate', 'Reserve Emission Rate', 0.5, 1.5),
        'fee_multiplier': ('yield_source.fee_multiplier', 'Fee Multiplier', 0.5, 2.0),
        'reward_curve_k': ('reward_curve.k', 'Reward Curve k', 0.67, 1.5),
        'max_duration_years': ('reward_curve.max_duration_years', 'Max Lock Duration', 0.6, 1.4),
        'market_multiplier': ('market.market_multiplier', 'Market Multiplier', 0.5, 2.0),
        'volatility': ('market.volatility', 'Volatility', 0.5, 1.5),
        'net_new_fraction': ('capital_flow.net_new_fraction', 'Net-New Capital Fraction', 0.5, 1.5),
        'participation_elasticity': ('simulation.participation_elasticity', 'Participation Elasticity', 0.5, 2.0),
        'ifil_apy': ('alternatives.ifil_apy', 'iFIL APY', 0.5, 2.0),
        'defi_apy': ('alternatives.defi_apy', 'DeFi APY', 0.5, 2.0),
        'network_growth_rate': ('externalities.network_growth_rate', 'Network Growth Rate', 0.33, 2.0),
        'retail_premium': ('cohorts.retail.required_premium', 'Retail Required Premium', 0.5, 2.0),
        'institutional_premium': ('cohorts.institutional.required_premium', 'Institutional Required Premium', 0.5, 1.5),
    }

    # Core metrics to track
    CORE_METRICS = [
        'final_locked',
        'final_reserve',
        'reserve_runway_years',
        'effective_inflation_final'
    ]

    def __init__(self, config: Config, parameters: Dict[str, Tuple] = None):
        """
        Initialize sensitivity analyzer.

        Args:
            config: Base configuration
            parameters: Optional custom parameter definitions
                Format: {name: (config_path, label, low_mult, high_mult)}
        """
        self.config = config
        self.parameters = parameters or self.DEFAULT_PARAMETERS

    def run_sweep(
        self,
        parameter_name: str,
        num_points: int = 11,
        random_seed: int = None
    ) -> ParameterSweep:
        """
        Run one-at-a-time sweep for a single parameter.

        Args:
            parameter_name: Name of parameter to sweep
            num_points: Number of sweep points
            random_seed: Random seed for reproducibility

        Returns:
            ParameterSweep result
        """
        if parameter_name not in self.parameters:
            raise ValueError(f"Unknown parameter: {parameter_name}")

        config_path, label, low_mult, high_mult = self.parameters[parameter_name]
        base_value = self._get_config_value(self.config, config_path)

        # Generate sweep values
        low_value = base_value * low_mult
        high_value = base_value * high_mult
        sweep_values = list(np.linspace(low_value, high_value, num_points))

        # Run simulation at each point
        metric_values = {metric: [] for metric in self.CORE_METRICS}

        for val in sweep_values:
            # Create modified config
            modified_config = copy.deepcopy(self.config)
            self._set_config_value(modified_config, config_path, val)

            # Run simulation
            runner = SimulationRunner(modified_config)
            seed = random_seed or self.config.simulation.random_seed
            result = runner.run(random_seed=seed)

            # Extract metrics
            metric_values['final_locked'].append(result.final_metrics['final_locked'])
            metric_values['final_reserve'].append(result.final_metrics['final_reserve'])
            metric_values['reserve_runway_years'].append(result.final_metrics['reserve_runway_years'])

            # Get final effective inflation
            if result.metrics_over_time:
                metric_values['effective_inflation_final'].append(
                    result.metrics_over_time[-1].get('effective_inflation', 0)
                )
            else:
                metric_values['effective_inflation_final'].append(0)

        return ParameterSweep(
            parameter_name=parameter_name,
            parameter_label=label,
            base_value=base_value,
            sweep_values=sweep_values,
            metric_values=metric_values
        )

    def compute_tornado(
        self,
        target_metric: str = 'final_locked',
        random_seed: int = None
    ) -> List[TornadoEntry]:
        """
        Compute tornado chart data for a target metric.

        Args:
            target_metric: Metric to analyze (default: final_locked)
            random_seed: Random seed for reproducibility

        Returns:
            List of TornadoEntry sorted by impact (largest first)
        """
        # First run base case
        runner = SimulationRunner(self.config)
        seed = random_seed or self.config.simulation.random_seed
        base_result = runner.run(random_seed=seed)

        # Get base metric value
        if target_metric == 'effective_inflation_final':
            base_metric = base_result.metrics_over_time[-1].get('effective_inflation', 0) if base_result.metrics_over_time else 0
        else:
            base_metric = base_result.final_metrics.get(target_metric, 0)

        entries = []

        for param_name, (config_path, label, low_mult, high_mult) in self.parameters.items():
            base_value = self._get_config_value(self.config, config_path)
            low_value = base_value * low_mult
            high_value = base_value * high_mult

            # Run at low value
            low_config = copy.deepcopy(self.config)
            self._set_config_value(low_config, config_path, low_value)
            low_runner = SimulationRunner(low_config)
            low_result = low_runner.run(random_seed=seed)

            if target_metric == 'effective_inflation_final':
                metric_at_low = low_result.metrics_over_time[-1].get('effective_inflation', 0) if low_result.metrics_over_time else 0
            else:
                metric_at_low = low_result.final_metrics.get(target_metric, 0)

            # Run at high value
            high_config = copy.deepcopy(self.config)
            self._set_config_value(high_config, config_path, high_value)
            high_runner = SimulationRunner(high_config)
            high_result = high_runner.run(random_seed=seed)

            if target_metric == 'effective_inflation_final':
                metric_at_high = high_result.metrics_over_time[-1].get('effective_inflation', 0) if high_result.metrics_over_time else 0
            else:
                metric_at_high = high_result.final_metrics.get(target_metric, 0)

            impact_range = abs(metric_at_high - metric_at_low)

            entries.append(TornadoEntry(
                parameter_name=param_name,
                parameter_label=label,
                base_value=base_value,
                low_value=low_value,
                high_value=high_value,
                metric_at_low=metric_at_low,
                metric_at_high=metric_at_high,
                impact_range=impact_range
            ))

        # Sort by impact (largest first)
        entries.sort(key=lambda e: e.impact_range, reverse=True)

        return entries

    def run_full_analysis(
        self,
        num_sweep_points: int = 11,
        random_seed: int = None
    ) -> SensitivityResult:
        """
        Run complete sensitivity analysis.

        Args:
            num_sweep_points: Number of points per parameter sweep
            random_seed: Random seed for reproducibility

        Returns:
            Complete SensitivityResult
        """
        seed = random_seed or self.config.simulation.random_seed

        # Run base case
        runner = SimulationRunner(self.config)
        base_result = runner.run(random_seed=seed)
        base_metrics = {
            'final_locked': base_result.final_metrics['final_locked'],
            'final_reserve': base_result.final_metrics['final_reserve'],
            'reserve_runway_years': base_result.final_metrics['reserve_runway_years'],
            'effective_inflation_final': base_result.metrics_over_time[-1].get('effective_inflation', 0) if base_result.metrics_over_time else 0
        }

        # Run sweeps for all parameters
        sweeps = {}
        for param_name in self.parameters:
            sweeps[param_name] = self.run_sweep(param_name, num_sweep_points, seed)

        # Compute tornado charts for each core metric
        tornado_data = {}
        for metric in self.CORE_METRICS:
            tornado_data[metric] = self.compute_tornado(metric, seed)

        return SensitivityResult(
            base_config=self.config,
            base_metrics=base_metrics,
            sweeps=sweeps,
            tornado_data=tornado_data
        )

    def _get_config_value(self, config: Config, path: str) -> float:
        """Get a value from config using dot-notation path."""
        parts = path.split('.')
        obj = config
        for part in parts:
            obj = getattr(obj, part)
        return float(obj)

    def _set_config_value(self, config: Config, path: str, value: float) -> None:
        """Set a value in config using dot-notation path."""
        parts = path.split('.')
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)


def compute_parameter_importance(
    tornado_entries: List[TornadoEntry],
    base_metric: float
) -> Dict[str, float]:
    """
    Compute normalized parameter importance scores.

    Args:
        tornado_entries: List of tornado entries
        base_metric: Base case metric value

    Returns:
        Dict mapping parameter name to importance score (0-1)
    """
    if not tornado_entries:
        return {}

    # Normalize by base metric to get relative importance
    max_impact = max(e.impact_range for e in tornado_entries)

    importance = {}
    for entry in tornado_entries:
        if max_impact > 0:
            importance[entry.parameter_name] = entry.impact_range / max_impact
        else:
            importance[entry.parameter_name] = 0.0

    return importance
