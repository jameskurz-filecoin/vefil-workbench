"""Calibration notes and model documentation for veFIL simulation."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config.schema import Config


@dataclass
class CalibrationNote:
    """A single calibration note for a parameter or assumption."""
    parameter: str
    label: str
    current_value: Any
    source: str  # e.g., "assumption", "empirical", "derived", "external_data"
    confidence: str  # "high", "medium", "low"
    notes: str
    empirical_range: Optional[tuple] = None  # (low, high) if known
    references: Optional[List[str]] = None


@dataclass
class ModelLimit:
    """A documented model limitation."""
    category: str  # e.g., "price", "behavior", "network", "structural"
    title: str
    description: str
    impact: str  # How this affects results
    mitigation: Optional[str] = None  # Suggested approach


# ============================================================================
# HIGH-LEVERAGE ASSUMPTIONS
# These are the assumptions that most significantly affect model outcomes
# ============================================================================

HIGH_LEVERAGE_ASSUMPTIONS = [
    CalibrationNote(
        parameter="yield_source.reserve_annual_rate",
        label="Reserve Emission Rate",
        current_value=None,  # Will be populated from config
        source="policy_choice",
        confidence="high",
        notes="Core protocol design parameter. Directly determines reward budget and reserve runway. "
              "Higher rates attract more lockers but deplete reserve faster.",
        empirical_range=(0.03, 0.10),
        references=["FIP proposal discussions", "ve-token analysis comparisons"]
    ),
    CalibrationNote(
        parameter="reward_curve.k",
        label="Reward Curve Exponent (k)",
        current_value=None,
        source="policy_choice",
        confidence="medium",
        notes="Controls convexity of duration→weight mapping. k>1 favors long-term lockers. "
              "This significantly affects lock duration distribution and capital efficiency.",
        empirical_range=(1.0, 2.5),
        references=["Curve Finance (k=1.0)", "Balancer veBAL (k~1.5)"]
    ),
    CalibrationNote(
        parameter="capital_flow.net_new_fraction",
        label="Net-New Capital Fraction",
        current_value=None,
        source="assumption",
        confidence="low",
        notes="Fraction of lock inflows that represent genuinely new capital entering the ecosystem. "
              "Highly uncertain—depends on market conditions and alternative opportunity costs. "
              "This is a HIGH-LEVERAGE assumption: small changes significantly affect net inflation.",
        empirical_range=(0.2, 0.6),
        references=["No direct empirical data; estimated from DeFi capital flow patterns"]
    ),
    CalibrationNote(
        parameter="simulation.participation_elasticity",
        label="Participation Elasticity",
        current_value=None,
        source="assumption",
        confidence="low",
        notes="How responsive users are to yield premiums. Higher elasticity means small yield changes "
              "cause large participation swings. This is behaviorally uncertain and context-dependent.",
        empirical_range=(0.5, 3.0),
        references=["Estimated from DeFi yield farming response patterns"]
    ),
    CalibrationNote(
        parameter="cohorts.institutional.required_premium",
        label="Institutional Required Premium",
        current_value=None,
        source="assumption",
        confidence="low",
        notes="Minimum yield premium institutions need above alternatives to participate. "
              "Institutions have higher opportunity costs and stricter risk frameworks. "
              "This parameter gates institutional adoption—a key uncertainty.",
        empirical_range=(0.10, 0.35),
        references=["Institutional crypto investment surveys"]
    ),
    CalibrationNote(
        parameter="alternatives.defi_apy",
        label="DeFi Alternative APY",
        current_value=None,
        source="external_data",
        confidence="medium",
        notes="Competing yield opportunity in DeFi markets. This is STATIC per simulation run—"
              "real DeFi yields are highly variable. Consider running scenarios with different values.",
        empirical_range=(0.05, 0.25),
        references=["DeFiLlama yield aggregator", "Historical lending rates"]
    ),
    CalibrationNote(
        parameter="externalities.network_growth_rate",
        label="Network Growth Rate",
        current_value=None,
        source="assumption",
        confidence="low",
        notes="Annual growth rate of Filecoin network activity (transaction volume). "
              "Only relevant for fee-based yield sources. Historically volatile and uncertain.",
        empirical_range=(0.0, 0.30),
        references=["Filecoin network statistics"]
    ),
]


# ============================================================================
# MODEL LIMITATIONS
# ============================================================================

MODEL_LIMITS = [
    ModelLimit(
        category="price",
        title="No Endogenous FIL Price",
        description="The model does not simulate FIL price dynamics. It assumes constant purchasing power "
                    "and ignores price impact from lock/unlock flows.",
        impact="Cannot capture reflexive dynamics where locks affect price, which affects lock attractiveness. "
               "Results may over/underestimate adoption in bull/bear markets.",
        mitigation="Run scenarios with different alternative yield assumptions to proxy price regime effects."
    ),
    ModelLimit(
        category="behavior",
        title="Static Alternative Yields",
        description="Alternative yield rates (iFIL, GLIF, DeFi) are constant within each simulation run.",
        impact="Real alternative yields are dynamic and may adjust competitively to veFIL. "
               "Model cannot capture yield competition dynamics.",
        mitigation="Use Monte Carlo with yield uncertainty or compare multiple scenario configurations."
    ),
    ModelLimit(
        category="behavior",
        title="Simplified Rational Actors",
        description="Users are modeled as yield-maximizing rational agents with fixed cohort characteristics.",
        impact="Ignores irrational behavior, herd effects, information asymmetry, and behavioral biases. "
               "May overestimate responsiveness to yield changes.",
        mitigation="Participation elasticity parameter provides some behavioral flexibility."
    ),
    ModelLimit(
        category="network",
        title="Simplified Pledge Mechanics",
        description="Storage Provider pledge requirements are not dynamically modeled.",
        impact="May miss interactions between SP collateral needs and veFIL locking decisions.",
        mitigation="SP cohort parameters can be adjusted to proxy different pledge scenarios."
    ),
    ModelLimit(
        category="structural",
        title="No Secondary Markets",
        description="No modeling of veFIL derivatives, wrapped tokens, or secondary liquidity.",
        impact="Ignores potential for liquid staking wrappers that could increase effective supply velocity.",
        mitigation="Adversarial parameters (wrapper_concentration_limit) provide some stress testing."
    ),
    ModelLimit(
        category="structural",
        title="Discrete Time Steps",
        description="Simulation uses discrete monthly time steps rather than continuous dynamics.",
        impact="May miss intra-month dynamics and rapid response scenarios.",
        mitigation="Use shorter timesteps for higher-resolution analysis (increases computation)."
    ),
    ModelLimit(
        category="network",
        title="Exogenous Network Growth",
        description="Network transaction volume grows at a fixed rate, independent of veFIL dynamics.",
        impact="Cannot capture potential positive feedback where successful veFIL increases network utility.",
        mitigation="Run scenarios with different network growth assumptions."
    ),
]


class CalibrationPanel:
    """Generate calibration notes and warnings for a configuration."""

    def __init__(self, config: Config):
        """
        Initialize calibration panel.

        Args:
            config: Current simulation configuration
        """
        self.config = config

    def get_calibration_notes(self) -> List[CalibrationNote]:
        """
        Get calibration notes with current values populated.

        Returns:
            List of calibration notes with current config values
        """
        notes = []
        for note in HIGH_LEVERAGE_ASSUMPTIONS:
            # Create copy with current value populated
            current_value = self._get_config_value(note.parameter)
            updated_note = CalibrationNote(
                parameter=note.parameter,
                label=note.label,
                current_value=current_value,
                source=note.source,
                confidence=note.confidence,
                notes=note.notes,
                empirical_range=note.empirical_range,
                references=note.references
            )
            notes.append(updated_note)
        return notes

    def get_model_limits(self) -> List[ModelLimit]:
        """Get all documented model limitations."""
        return MODEL_LIMITS

    def get_high_confidence_params(self) -> List[CalibrationNote]:
        """Get only high-confidence calibration notes."""
        return [n for n in self.get_calibration_notes() if n.confidence == "high"]

    def get_low_confidence_params(self) -> List[CalibrationNote]:
        """Get only low-confidence (uncertain) calibration notes."""
        return [n for n in self.get_calibration_notes() if n.confidence == "low"]

    def check_empirical_bounds(self) -> List[Dict[str, Any]]:
        """
        Check if current values are within empirical ranges.

        Returns:
            List of warnings for out-of-range values
        """
        warnings = []
        for note in self.get_calibration_notes():
            if note.empirical_range is None:
                continue

            low, high = note.empirical_range
            if note.current_value is not None:
                if note.current_value < low:
                    warnings.append({
                        "parameter": note.label,
                        "current": note.current_value,
                        "expected_range": note.empirical_range,
                        "issue": f"Below empirical range ({note.current_value} < {low})"
                    })
                elif note.current_value > high:
                    warnings.append({
                        "parameter": note.label,
                        "current": note.current_value,
                        "expected_range": note.empirical_range,
                        "issue": f"Above empirical range ({note.current_value} > {high})"
                    })
        return warnings

    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate calibration summary.

        Returns:
            Summary dict with counts and key info
        """
        notes = self.get_calibration_notes()
        bounds_warnings = self.check_empirical_bounds()

        return {
            "total_assumptions": len(notes),
            "high_confidence": len([n for n in notes if n.confidence == "high"]),
            "medium_confidence": len([n for n in notes if n.confidence == "medium"]),
            "low_confidence": len([n for n in notes if n.confidence == "low"]),
            "out_of_range_warnings": len(bounds_warnings),
            "model_limitations": len(MODEL_LIMITS),
            "bounds_warnings": bounds_warnings,
        }

    def _get_config_value(self, path: str) -> Any:
        """Get a value from config using dot-notation path."""
        try:
            parts = path.split('.')
            obj = self.config
            for part in parts:
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            return None


def format_calibration_markdown(panel: CalibrationPanel) -> str:
    """
    Format calibration panel as markdown.

    Args:
        panel: CalibrationPanel instance

    Returns:
        Markdown formatted string
    """
    lines = ["# Calibration Notes\n"]

    # Summary
    summary = panel.generate_summary()
    lines.append("## Summary\n")
    lines.append(f"- **Total documented assumptions**: {summary['total_assumptions']}")
    lines.append(f"- **High confidence**: {summary['high_confidence']}")
    lines.append(f"- **Medium confidence**: {summary['medium_confidence']}")
    lines.append(f"- **Low confidence**: {summary['low_confidence']}")
    lines.append(f"- **Out-of-range warnings**: {summary['out_of_range_warnings']}\n")

    # Warnings
    if summary['bounds_warnings']:
        lines.append("## ⚠️ Out-of-Range Warnings\n")
        for w in summary['bounds_warnings']:
            lines.append(f"- **{w['parameter']}**: {w['issue']}")
        lines.append("")

    # Low confidence parameters (most important to flag)
    low_conf = panel.get_low_confidence_params()
    if low_conf:
        lines.append("## 🔴 Low Confidence Parameters\n")
        lines.append("*These parameters significantly affect results but have high uncertainty:*\n")
        for note in low_conf:
            lines.append(f"### {note.label}")
            lines.append(f"- **Current value**: {note.current_value}")
            if note.empirical_range:
                lines.append(f"- **Empirical range**: {note.empirical_range[0]} – {note.empirical_range[1]}")
            lines.append(f"- {note.notes}")
            lines.append("")

    # Model limitations
    lines.append("## Model Limitations\n")
    for limit in MODEL_LIMITS:
        lines.append(f"### {limit.title}")
        lines.append(f"*Category: {limit.category}*\n")
        lines.append(limit.description)
        lines.append(f"\n**Impact**: {limit.impact}")
        if limit.mitigation:
            lines.append(f"\n**Mitigation**: {limit.mitigation}")
        lines.append("")

    return "\n".join(lines)
