"""Analysis tools for veFIL simulation."""

from .calibration import (
    HIGH_LEVERAGE_ASSUMPTIONS,
    MODEL_LIMITS,
    CalibrationNote,
    CalibrationPanel,
    ModelLimit,
    format_calibration_markdown,
)
from .changelog import (
    ChangelogEntry,
    ChangelogManager,
    ConfigDiff,
    compare_configs,
    format_changelog_markdown,
)
from .regime import (
    REGIME_TOLERANCE,
    LockGuardrails,
    RegimeAnalysisResult,
    RegimeType,
    WindowMetrics,
    analyze_regime,
    classify_regime,
    compute_lock_guardrails,
    compute_window_metrics,
    format_inflation_for_display,
    format_regime_for_display,
)
from .scenarios import (
    SCENARIO_LIBRARY,
    Scenario,
    ScenarioComparison,
    ScenarioRunner,
    format_comparison_table,
)
from .sensitivity import (
    ParameterSweep,
    SensitivityAnalyzer,
    SensitivityResult,
    TornadoEntry,
    compute_parameter_importance,
)

__all__ = [
    # Sensitivity analysis
    "SensitivityAnalyzer",
    "SensitivityResult",
    "ParameterSweep",
    "TornadoEntry",
    "compute_parameter_importance",
    # Scenario comparison
    "Scenario",
    "ScenarioComparison",
    "ScenarioRunner",
    "SCENARIO_LIBRARY",
    "format_comparison_table",
    # Calibration
    "CalibrationNote",
    "ModelLimit",
    "CalibrationPanel",
    "HIGH_LEVERAGE_ASSUMPTIONS",
    "MODEL_LIMITS",
    "format_calibration_markdown",
    # Changelog
    "ChangelogEntry",
    "ConfigDiff",
    "ChangelogManager",
    "compare_configs",
    "format_changelog_markdown",
    # Regime analysis
    "RegimeType",
    "WindowMetrics",
    "LockGuardrails",
    "RegimeAnalysisResult",
    "REGIME_TOLERANCE",
    "classify_regime",
    "compute_window_metrics",
    "compute_lock_guardrails",
    "analyze_regime",
    "format_regime_for_display",
    "format_inflation_for_display",
]
