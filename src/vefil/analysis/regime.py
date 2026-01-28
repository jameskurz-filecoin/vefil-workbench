"""Regime analysis for veFIL tokenomics - windowed deflationary/inflationary classification.

This module provides time-frame regime analysis that classifies periods as deflationary
or inflationary based on aggregated flows across configurable windows.

Key Concepts:
- Regime is determined by effective inflation: negative = deflationary, positive = inflationary
- Effective inflation = (emission - net_lock_change) / circulating, annualized
- Net lock change = new_locks - unlocks
- Windows aggregate flows across multiple timesteps for robust classification

Denominator Choice:
We use the circulating supply at the START of each window as the denominator.
This provides a stable reference point and avoids the circularity of using
a denominator that changes due to the flows being measured.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class RegimeType(Enum):
    """Regime classification."""
    DEFLATIONARY = "deflationary"
    INFLATIONARY = "inflationary"
    NEUTRAL = "neutral"


@dataclass
class WindowMetrics:
    """Aggregated metrics for a time window."""
    window_name: str
    window_months: int
    start_day: float
    end_day: float

    # Aggregated flows
    emission_sum: float
    new_locks_sum: float
    unlocks_sum: float
    net_lock_change: float  # new_locks_sum - unlocks_sum

    # Denominator (circulating at window start)
    circulating_at_start: float

    # Computed metrics
    effective_inflation_annualized: float  # As a decimal (e.g., 0.05 = 5%)
    regime: RegimeType

    # Additional context
    locked_at_start: float
    locked_at_end: float
    locked_change: float
    locked_share_at_end: float  # locked / total_supply at window end


@dataclass
class LockGuardrails:
    """Near-term lock guardrails and target tracking."""
    # Near-term locked amounts
    locked_at_3_months: float
    locked_at_6_months: float
    locked_at_12_months: float

    # Warnings
    warning_3m_below_1m: bool
    warning_6m_below_1m: bool

    # Lock tracking
    current_locked: float = 0.0

    # Share metrics
    locked_share_of_circulating: float = 0.0
    locked_share_of_total: float = 0.0


@dataclass
class RegimeAnalysisResult:
    """Complete regime analysis result."""
    windows: List[WindowMetrics]
    guardrails: LockGuardrails

    # Summary
    early_regime: RegimeType  # First 6 months
    mid_regime: RegimeType    # 6-24 months
    late_regime: RegimeType   # After 24 months (if available)
    full_horizon_regime: RegimeType


# Tolerance for regime classification (in annualized decimal form)
# Values within this band are classified as NEUTRAL
REGIME_TOLERANCE = 0.005  # 0.5% annual


def classify_regime(effective_inflation: float, tolerance: float = REGIME_TOLERANCE) -> RegimeType:
    """
    Classify regime based on effective inflation.

    Args:
        effective_inflation: Annualized effective inflation as decimal (e.g., 0.05 = 5%)
        tolerance: Band around zero for neutral classification

    Returns:
        RegimeType classification
    """
    if effective_inflation < -tolerance:
        return RegimeType.DEFLATIONARY
    elif effective_inflation > tolerance:
        return RegimeType.INFLATIONARY
    else:
        return RegimeType.NEUTRAL


def compute_window_metrics(
    metrics_over_time: List[Dict[str, Any]],
    states: List[Any],  # List[SystemState]
    window_months: int,
    window_name: str,
    dt_days: float = 30.0
) -> Optional[WindowMetrics]:
    """
    Compute aggregated metrics for a specific time window.

    Args:
        metrics_over_time: List of per-step metrics from simulation
        states: List of SystemState objects
        window_months: Window duration in months
        window_name: Human-readable name for window
        dt_days: Timestep in days

    Returns:
        WindowMetrics for the window, or None if insufficient data
    """
    if not metrics_over_time or not states:
        return None

    window_days = window_months * 30.0

    # Find steps within window
    steps_in_window = []
    for i, m in enumerate(metrics_over_time):
        t = m.get('t', 0)
        if t <= window_days:
            steps_in_window.append((i, m))

    if not steps_in_window:
        return None

    # Aggregate flows across window
    emission_sum = sum(m.get('emission', 0) for _, m in steps_in_window)
    new_locks_sum = sum(m.get('new_locks', 0) for _, m in steps_in_window)
    unlocks_sum = sum(m.get('unlocks', 0) for _, m in steps_in_window)
    net_lock_change = new_locks_sum - unlocks_sum

    # Get circulating at window start (from first state or initial)
    # states[0] is after first step, so we need to be careful
    first_step_idx = steps_in_window[0][0]
    if first_step_idx > 0 and first_step_idx - 1 < len(states):
        circulating_at_start = states[first_step_idx - 1].circulating
    elif states:
        # Use first state as approximation
        circulating_at_start = states[0].circulating
    else:
        return None

    # Get locked values
    last_step_idx = steps_in_window[-1][0]
    locked_at_start = states[first_step_idx].locked_vefil if first_step_idx < len(states) else 0.0
    locked_at_end = states[last_step_idx].locked_vefil if last_step_idx < len(states) else 0.0
    total_supply_at_end = states[last_step_idx].total_supply if last_step_idx < len(states) else 1.0

    # Compute effective inflation for window
    # Formula: ((emission - net_lock_change) / circulating_at_start) * (365.25 / window_days)
    if circulating_at_start > 0 and window_days > 0:
        effective_inflation_raw = (emission_sum - net_lock_change) / circulating_at_start
        effective_inflation_annualized = effective_inflation_raw * (365.25 / window_days)
    else:
        effective_inflation_annualized = 0.0

    regime = classify_regime(effective_inflation_annualized)

    return WindowMetrics(
        window_name=window_name,
        window_months=window_months,
        start_day=0.0,
        end_day=window_days,
        emission_sum=emission_sum,
        new_locks_sum=new_locks_sum,
        unlocks_sum=unlocks_sum,
        net_lock_change=net_lock_change,
        circulating_at_start=circulating_at_start,
        effective_inflation_annualized=effective_inflation_annualized,
        regime=regime,
        locked_at_start=locked_at_start,
        locked_at_end=locked_at_end,
        locked_change=locked_at_end - locked_at_start,
        locked_share_at_end=locked_at_end / total_supply_at_end if total_supply_at_end > 0 else 0.0
    )


def compute_lock_guardrails(
    metrics_over_time: List[Dict[str, Any]],
    states: List[Any],
    total_supply: float,
    circulating: float,
    dt_days: float = 30.0
) -> LockGuardrails:
    """
    Compute near-term lock guardrails.

    Args:
        metrics_over_time: List of per-step metrics from simulation
        states: List of SystemState objects
        total_supply: Total FIL supply
        circulating: Current circulating supply
        dt_days: Timestep in days

    Returns:
        LockGuardrails with warnings and lock metrics
    """
    # Find locked values at key time points
    locked_3m = 0.0
    locked_6m = 0.0
    locked_12m = 0.0
    current_locked = 0.0

    days_3m = 90.0
    days_6m = 180.0
    days_12m = 365.0

    for state in states:
        t = state.t
        current_locked = state.locked_vefil

        if t <= days_3m:
            locked_3m = state.locked_vefil
        if t <= days_6m:
            locked_6m = state.locked_vefil
        if t <= days_12m:
            locked_12m = state.locked_vefil

    # Check warnings
    warning_3m = locked_3m < 1_000_000
    warning_6m = locked_6m < 1_000_000

    # Compute share metrics
    share_of_circulating = current_locked / circulating if circulating > 0 else 0.0
    share_of_total = current_locked / total_supply if total_supply > 0 else 0.0

    return LockGuardrails(
        locked_at_3_months=locked_3m,
        locked_at_6_months=locked_6m,
        locked_at_12_months=locked_12m,
        warning_3m_below_1m=warning_3m,
        warning_6m_below_1m=warning_6m,
        current_locked=current_locked,
        locked_share_of_circulating=share_of_circulating,
        locked_share_of_total=share_of_total
    )


def analyze_regime(
    metrics_over_time: List[Dict[str, Any]],
    states: List[Any],
    total_supply: float,
    circulating: float,
    dt_days: float = 30.0
) -> RegimeAnalysisResult:
    """
    Perform complete regime analysis across multiple time windows.

    Required windows:
    - 3 months
    - 6 months
    - 12 months
    - 24 months
    - Full horizon

    Args:
        metrics_over_time: List of per-step metrics from simulation
        states: List of SystemState objects
        total_supply: Total FIL supply
        circulating: Current circulating supply
        dt_days: Timestep in days

    Returns:
        Complete RegimeAnalysisResult
    """
    # Define windows
    window_configs = [
        (3, "3 months"),
        (6, "6 months"),
        (12, "12 months"),
        (24, "24 months"),
    ]

    # Compute full horizon months
    if states:
        full_horizon_days = states[-1].t
        full_horizon_months = int(full_horizon_days / 30)
    else:
        full_horizon_months = 60  # Default 5 years

    window_configs.append((full_horizon_months, f"Full horizon ({full_horizon_months}mo)"))

    # Compute window metrics
    windows = []
    for months, name in window_configs:
        window = compute_window_metrics(
            metrics_over_time=metrics_over_time,
            states=states,
            window_months=months,
            window_name=name,
            dt_days=dt_days
        )
        if window:
            windows.append(window)

    # Compute guardrails
    guardrails = compute_lock_guardrails(
        metrics_over_time=metrics_over_time,
        states=states,
        total_supply=total_supply,
        circulating=circulating,
        dt_days=dt_days
    )

    # Determine summary regimes
    early_regime = RegimeType.NEUTRAL
    mid_regime = RegimeType.NEUTRAL
    late_regime = RegimeType.NEUTRAL
    full_horizon_regime = RegimeType.NEUTRAL

    for w in windows:
        if w.window_months <= 6:
            early_regime = w.regime
        elif w.window_months <= 24:
            mid_regime = w.regime
        else:
            late_regime = w.regime
            full_horizon_regime = w.regime

    return RegimeAnalysisResult(
        windows=windows,
        guardrails=guardrails,
        early_regime=early_regime,
        mid_regime=mid_regime,
        late_regime=late_regime,
        full_horizon_regime=full_horizon_regime
    )


def format_regime_for_display(regime: RegimeType) -> Tuple[str, str]:
    """
    Format regime for UI display.

    Returns:
        (label, css_color_class)
    """
    if regime == RegimeType.DEFLATIONARY:
        return ("Deflationary", "green")
    elif regime == RegimeType.INFLATIONARY:
        return ("Inflationary", "red")
    else:
        return ("Neutral", "amber")


def format_inflation_for_display(inflation: float) -> str:
    """
    Format effective inflation for UI display.

    Args:
        inflation: Annualized effective inflation as decimal

    Returns:
        Formatted string with sign and percentage
    """
    pct = inflation * 100
    if pct >= 0:
        return f"+{pct:.2f}%"
    else:
        return f"{pct:.2f}%"


# ============================================================================
# Credibility Levers: Sensitivity Analysis and Break Conditions
# ============================================================================

@dataclass
class SensitivityResult:
    """Result of sensitivity analysis for a single parameter perturbation."""
    parameter_name: str
    baseline_value: float
    perturbed_value: float
    perturbation_pct: float
    baseline_regime: RegimeType
    perturbed_regime: RegimeType
    regime_flipped: bool
    baseline_inflation: float
    perturbed_inflation: float
    inflation_delta: float


@dataclass
class BreakCondition:
    """A condition that would flip the regime from current state."""
    description: str
    parameter: str
    current_value: float
    break_value: float
    change_required_pct: float
    direction: str  # "increase" or "decrease"


@dataclass
class CredibilityAnalysis:
    """Complete credibility analysis including sensitivity and break conditions."""
    sensitivity_results: List[SensitivityResult]
    break_conditions: List[BreakCondition]
    most_sensitive_parameter: Optional[str]
    regime_stability: str  # "stable", "marginal", or "fragile"


def compute_sensitivity(
    window: WindowMetrics,
    perturbation_pct: float = 0.10
) -> List[SensitivityResult]:
    """
    Compute regime sensitivity to small perturbations in key parameters.

    This helps readers understand how robust the regime classification is
    by showing what happens under small changes to the underlying flows.

    Args:
        window: WindowMetrics for a specific time window
        perturbation_pct: Percentage to perturb each parameter (default 10%)

    Returns:
        List of SensitivityResult for each perturbed parameter
    """
    results = []
    baseline_inflation = window.effective_inflation_annualized
    baseline_regime = window.regime

    # Parameters to perturb: emission, new_locks, unlocks
    parameters = [
        ("Emissions", window.emission_sum),
        ("New Locks", window.new_locks_sum),
        ("Unlocks", window.unlocks_sum),
    ]

    window_days = window.end_day - window.start_day
    if window_days <= 0:
        window_days = window.window_months * 30.0

    for param_name, baseline_val in parameters:
        # Skip if baseline is zero (can't perturb meaningfully)
        if baseline_val <= 0:
            continue

        # Compute perturbed value
        delta = baseline_val * perturbation_pct
        perturbed_val = baseline_val + delta

        # Recompute effective inflation with perturbed value
        if param_name == "Emissions":
            new_emission = perturbed_val
            new_net_lock = window.net_lock_change
        elif param_name == "New Locks":
            new_emission = window.emission_sum
            new_net_lock = perturbed_val - window.unlocks_sum
        else:  # Unlocks
            new_emission = window.emission_sum
            new_net_lock = window.new_locks_sum - perturbed_val

        if window.circulating_at_start > 0:
            perturbed_raw = (new_emission - new_net_lock) / window.circulating_at_start
            perturbed_inflation = perturbed_raw * (365.25 / window_days)
        else:
            perturbed_inflation = 0.0

        perturbed_regime = classify_regime(perturbed_inflation)

        results.append(SensitivityResult(
            parameter_name=param_name,
            baseline_value=baseline_val,
            perturbed_value=perturbed_val,
            perturbation_pct=perturbation_pct * 100,
            baseline_regime=baseline_regime,
            perturbed_regime=perturbed_regime,
            regime_flipped=(perturbed_regime != baseline_regime),
            baseline_inflation=baseline_inflation,
            perturbed_inflation=perturbed_inflation,
            inflation_delta=perturbed_inflation - baseline_inflation
        ))

    return results


def compute_break_conditions(
    window: WindowMetrics,
    tolerance: float = REGIME_TOLERANCE
) -> List[BreakCondition]:
    """
    Compute what conditions would flip the regime from its current state.

    This answers: "What would need to be true for this to become inflationary?"
    (or deflationary, depending on current state).

    Args:
        window: WindowMetrics for a specific time window
        tolerance: Regime tolerance band

    Returns:
        List of BreakCondition describing what would flip the regime
    """
    conditions = []
    current_inflation = window.effective_inflation_annualized
    current_regime = window.regime

    window_days = window.end_day - window.start_day
    if window_days <= 0:
        window_days = window.window_months * 30.0

    # Determine target inflation to flip regime
    if current_regime == RegimeType.DEFLATIONARY:
        # Need to become inflationary (> tolerance)
        target_inflation = tolerance + 0.001
        target_label = "become inflationary"
    elif current_regime == RegimeType.INFLATIONARY:
        # Need to become deflationary (< -tolerance)
        target_inflation = -tolerance - 0.001
        target_label = "become deflationary"
    else:
        # Neutral - could go either way, show both
        target_inflation = tolerance + 0.001
        target_label = "become inflationary"

    # Effective inflation = (emission - net_lock_change) / circulating * annualize
    # Solve for each parameter

    circulating = window.circulating_at_start
    if circulating <= 0:
        return conditions

    # Convert target to raw (un-annualized) value
    target_raw = target_inflation * (window_days / 365.25)
    current_raw = current_inflation * (window_days / 365.25)

    # 1. What emission would flip it?
    # target_raw = (emission - net_lock) / circulating
    # emission = target_raw * circulating + net_lock
    required_emission = target_raw * circulating + window.net_lock_change
    if required_emission > 0:
        change_pct = ((required_emission - window.emission_sum) / window.emission_sum * 100) if window.emission_sum > 0 else float('inf')
        direction = "increase" if required_emission > window.emission_sum else "decrease"
        if abs(change_pct) < 1000:  # Only show reasonable changes
            conditions.append(BreakCondition(
                description=f"Emissions would need to {direction} by {abs(change_pct):.1f}% to {target_label}",
                parameter="Emissions",
                current_value=window.emission_sum,
                break_value=required_emission,
                change_required_pct=change_pct,
                direction=direction
            ))

    # 2. What net lock change would flip it?
    # target_raw = (emission - net_lock) / circulating
    # net_lock = emission - target_raw * circulating
    required_net_lock = window.emission_sum - target_raw * circulating
    current_net_lock = window.net_lock_change
    if current_net_lock != 0:
        change_pct = ((required_net_lock - current_net_lock) / abs(current_net_lock) * 100)
    else:
        change_pct = float('inf') if required_net_lock != 0 else 0

    direction = "increase" if required_net_lock > current_net_lock else "decrease"
    if abs(change_pct) < 1000:
        conditions.append(BreakCondition(
            description=f"Net locks would need to {direction} by {abs(change_pct):.1f}% to {target_label}",
            parameter="Net Locks",
            current_value=current_net_lock,
            break_value=required_net_lock,
            change_required_pct=change_pct,
            direction=direction
        ))

    return conditions


def analyze_credibility(
    windows: List[WindowMetrics],
    primary_window_name: str = "12 months",
    perturbation_pct: float = 0.10
) -> CredibilityAnalysis:
    """
    Perform comprehensive credibility analysis on regime classification.

    This helps readers assess how robust the conclusions are by:
    1. Showing sensitivity to small parameter changes
    2. Computing explicit break conditions
    3. Assessing overall regime stability

    Args:
        windows: List of WindowMetrics from regime analysis
        primary_window_name: Which window to analyze for credibility
        perturbation_pct: Percentage perturbation for sensitivity (default 10%)

    Returns:
        CredibilityAnalysis with sensitivity results and break conditions
    """
    # Find the primary window
    primary_window = None
    for w in windows:
        if w.window_name == primary_window_name:
            primary_window = w
            break

    if primary_window is None and windows:
        primary_window = windows[-1]  # Use last (full horizon) if not found

    if primary_window is None:
        return CredibilityAnalysis(
            sensitivity_results=[],
            break_conditions=[],
            most_sensitive_parameter=None,
            regime_stability="unknown"
        )

    # Compute sensitivity
    sensitivity_results = compute_sensitivity(primary_window, perturbation_pct)

    # Compute break conditions
    break_conditions = compute_break_conditions(primary_window)

    # Find most sensitive parameter (smallest perturbation causing regime flip)
    flipping_results = [r for r in sensitivity_results if r.regime_flipped]
    most_sensitive = None
    if flipping_results:
        most_sensitive = min(flipping_results, key=lambda r: abs(r.inflation_delta))
        most_sensitive = most_sensitive.parameter_name

    # Assess regime stability based on how close we are to tolerance boundary
    inflation = abs(primary_window.effective_inflation_annualized)
    if inflation < REGIME_TOLERANCE * 0.5:
        stability = "marginal"
    elif any(r.regime_flipped for r in sensitivity_results):
        stability = "fragile"
    else:
        stability = "stable"

    return CredibilityAnalysis(
        sensitivity_results=sensitivity_results,
        break_conditions=break_conditions,
        most_sensitive_parameter=most_sensitive,
        regime_stability=stability
    )
