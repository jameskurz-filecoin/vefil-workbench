"""OpenAI-powered expert assistant for the veFIL workbench.

This module focuses on three things:
1) Building a rigorous, auditable context bundle.
2) Providing stable assumption/equation references for citation.
3) Calling the OpenAI Responses API when configured.

The assistant is intentionally explanatory (not predictive) and is designed
so that every claim can be tied back to explicit assumptions or equations.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config.schema import Config
from ..simulation.runner import SimulationResult
from ..validation.sanity_checks import ValidationWarning

# Stable assumption and equation references used for citations.
ASSUMPTION_REFS: List[Dict[str, str]] = [
    {
        "id": "A1",
        "text": "Lock demand is driven by APY relative to alternative yields and risk tolerance.",
    },
    {
        "id": "A2",
        "text": "Unlocks occur only at maturity; there is no early exit behavior in the base model.",
    },
    {
        "id": "A3",
        "text": "Effective inflation is computed from emissions minus net locks, annualized by timestep.",
    },
    {
        "id": "A4",
        "text": "Market impact and participation are approximated via liquidity regime and elasticity parameters.",
    },
    {
        "id": "A5",
        "text": "Reserve runway assumes the current emission pace continues without endogenous price feedback.",
    },
]

EQUATION_REFS: List[Dict[str, str]] = [
    {
        "id": "E1",
        "text": "lock_weight(d) = (d / max_duration)^k",
    },
    {
        "id": "E2",
        "text": "effective_inflation = (emission_sold - new_locks + unlocks) / circulating × (365.25 / dt)",
    },
    {
        "id": "E3",
        "text": "net_inflation_rate = Δcirculating / circulating × (365.25 / dt)",
    },
    {
        "id": "E4",
        "text": "reserve_runway_years ≈ reserve_balance / annualized_emission",
    },
]

GLOSSARY_REFS: List[Dict[str, str]] = [
    {
        "id": "G1",
        "term": "Effective Inflation",
        "definition": "Annualized emissions minus net locks, scaled by circulating supply.",
    },
    {
        "id": "G2",
        "term": "Net Locks",
        "definition": "New locks minus unlocks within a timestep.",
    },
    {
        "id": "G3",
        "term": "Reserve Runway",
        "definition": "Estimated years the reserve can sustain current emission levels.",
    },
    {
        "id": "G4",
        "term": "Participation Elasticity",
        "definition": "How strongly participation rates respond to yield premiums.",
    },
    {
        "id": "G5",
        "term": "Market Multiplier",
        "definition": "A price-impact proxy that scales market sensitivity in scenarios.",
    },
]


@dataclass
class MonteCarloSummary:
    """Small, assistant-friendly Monte Carlo summary."""

    confidence_level: str
    message: str
    flags: Sequence[str]
    metrics: Dict[str, Any]
    stats: Optional[Dict[str, Any]] = None


@dataclass
class AssistantContext:
    """Structured context passed to the LLM."""

    config_hash: str
    core_metrics: Dict[str, Any]
    latest_metrics: Dict[str, Any]
    config_summary: Dict[str, Any]
    monte_carlo: Optional[MonteCarloSummary]
    validation_summary: Dict[str, Any]
    assumptions: Sequence[Dict[str, str]]
    equations: Sequence[Dict[str, str]]
    glossary: Sequence[Dict[str, str]]
    config_full: Optional[Dict[str, Any]] = None

    def to_prompt_payload(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable prompt payload."""
        payload: Dict[str, Any] = {
            "config_hash": self.config_hash,
            "core_metrics": self.core_metrics,
            "latest_metrics": self.latest_metrics,
            "config_summary": self.config_summary,
            "validation_summary": self.validation_summary,
            "assumptions": list(self.assumptions),
            "equations": list(self.equations),
            "glossary": list(self.glossary),
        }
        if self.monte_carlo is not None:
            payload["monte_carlo"] = {
                "confidence_level": self.monte_carlo.confidence_level,
                "message": self.monte_carlo.message,
                "flags": list(self.monte_carlo.flags),
                "metrics": self.monte_carlo.metrics,
                "stats": self.monte_carlo.stats,
            }
        if self.config_full is not None:
            payload["config_full"] = self.config_full
        return payload


def _compute_core_metrics(result: Optional[SimulationResult]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute core and latest metrics in a stable shape for prompting."""
    if result is None or not result.metrics_over_time:
        empty = {
            "final_locked": 0.0,
            "locked_share": 0.0,
            "net_inflation_pct": 0.0,
            "effective_inflation_pct": 0.0,
            "net_locks_latest": 0.0,
            "reserve_runway_years": 0.0,
            "emission_latest": 0.0,
        }
        return empty, {
            "t": 0.0,
            "circulating": 0.0,
            "locked": 0.0,
            "reserve": 0.0,
            "emission": 0.0,
            "new_locks": 0.0,
            "unlocks": 0.0,
            "net_inflation_rate": 0.0,
            "effective_inflation": 0.0,
        }

    latest = result.metrics_over_time[-1]
    final_locked = float(result.final_metrics.get("final_locked", 0.0))
    total_supply = float(result.final_metrics.get("total_supply", 0.0) or 0.0)
    locked_share = (final_locked / total_supply) if total_supply > 0 else 0.0

    net_inflation_pct = float(latest.get("net_inflation_rate", 0.0)) * 100.0
    effective_inflation_pct = float(latest.get("effective_inflation", 0.0)) * 100.0
    net_locks_latest = float(latest.get("new_locks", 0.0)) - float(latest.get("unlocks", 0.0))

    core = {
        "final_locked": final_locked,
        "locked_share": locked_share,
        "net_inflation_pct": net_inflation_pct,
        "effective_inflation_pct": effective_inflation_pct,
        "net_locks_latest": net_locks_latest,
        "reserve_runway_years": float(result.final_metrics.get("reserve_runway_years", 0.0)),
        "emission_latest": float(latest.get("emission", 0.0)),
    }

    latest_metrics = {
        "t": float(latest.get("t", 0.0)),
        "circulating": float(latest.get("circulating", 0.0)),
        "locked": float(latest.get("locked", 0.0)),
        "reserve": float(latest.get("reserve", 0.0)),
        "emission": float(latest.get("emission", 0.0)),
        "new_locks": float(latest.get("new_locks", 0.0)),
        "unlocks": float(latest.get("unlocks", 0.0)),
        "net_inflation_rate": float(latest.get("net_inflation_rate", 0.0)),
        "effective_inflation": float(latest.get("effective_inflation", 0.0)),
    }

    return core, latest_metrics


def _summarize_config(config: Config) -> Dict[str, Any]:
    """Create a focused config summary for the assistant."""
    return {
        "yield_source": config.yield_source.type,
        "reserve_annual_rate": config.yield_source.reserve_annual_rate,
        "reward_curve_k": config.reward_curve.k,
        "min_duration_years": config.reward_curve.min_duration_years,
        "max_duration_years": config.reward_curve.max_duration_years,
        "time_horizon_months": config.simulation.time_horizon_months,
        "timestep_days": config.simulation.timestep_days,
        "participation_elasticity": config.simulation.participation_elasticity,
        "market_multiplier": config.market.market_multiplier,
        "liquidity_regime": config.market.liquidity_regime,
        "order_book_depth": config.market.order_book_depth,
        "alternatives": {
            "ifil_apy": config.alternatives.ifil_apy,
            "glif_apy": config.alternatives.glif_apy,
            "defi_apy": config.alternatives.defi_apy,
            "risk_free_rate": config.alternatives.risk_free_rate,
        },
    }


def _summarize_validation(warnings: Sequence[ValidationWarning]) -> Dict[str, Any]:
    """Summarize validation warnings in a compact format."""
    errors = [w for w in warnings if w.severity == "error"]
    warns = [w for w in warnings if w.severity == "warning"]
    return {
        "num_errors": len(errors),
        "num_warnings": len(warns),
        "top_items": [
            {
                "severity": w.severity,
                "category": w.category,
                "message": w.message,
            }
            for w in (errors + warns)[:6]
        ],
    }


def build_assistant_context(
    config: Config,
    result: Optional[SimulationResult],
    validation_warnings: Sequence[ValidationWarning],
    monte_carlo_confidence: Optional[Dict[str, Any]] = None,
    monte_carlo_stats: Optional[Dict[str, Any]] = None,
    show_workings: bool = False,
) -> AssistantContext:
    """Build the assistant context bundle."""
    core_metrics, latest_metrics = _compute_core_metrics(result)
    config_summary = _summarize_config(config)
    validation_summary = _summarize_validation(validation_warnings)

    mc_summary: Optional[MonteCarloSummary] = None
    if monte_carlo_confidence:
        mc_summary = MonteCarloSummary(
            confidence_level=str(monte_carlo_confidence.get("confidence_level", "unknown")),
            message=str(monte_carlo_confidence.get("message", "")),
            flags=monte_carlo_confidence.get("flags", []) or [],
            metrics=monte_carlo_confidence.get("metrics", {}) or {},
            stats=monte_carlo_stats,
        )

    config_full = config.to_dict() if show_workings else None

    return AssistantContext(
        config_hash=config.compute_hash(),
        core_metrics=core_metrics,
        latest_metrics=latest_metrics,
        config_summary=config_summary,
        monte_carlo=mc_summary,
        validation_summary=validation_summary,
        assumptions=ASSUMPTION_REFS,
        equations=EQUATION_REFS,
        glossary=GLOSSARY_REFS,
        config_full=config_full,
    )


def _format_assumptions_section() -> str:
    lines = ["Assumptions (cite by id):"]
    for ref in ASSUMPTION_REFS:
        lines.append(f"- [{ref['id']}] {ref['text']}")
    return "\n".join(lines)


def _format_equations_section() -> str:
    lines = ["Equations (cite by id):"]
    for ref in EQUATION_REFS:
        lines.append(f"- [{ref['id']}] {ref['text']}")
    return "\n".join(lines)


def _format_glossary_section() -> str:
    lines = ["Glossary (cite by id when defining terms):"]
    for ref in GLOSSARY_REFS:
        lines.append(f"- [{ref['id']}] {ref['term']}: {ref['definition']}")
    return "\n".join(lines)


def build_system_prompt(show_workings: bool) -> str:
    """Create the system prompt that enforces rigor and citations."""
    workings_instruction = (
        "Include a short 'Workings' section that shows the key equations and parameter values you used."
        if show_workings
        else "Do not include a 'Workings' section unless the user explicitly asks for it."
    )

    return "\n".join(
        [
            "You are the veFIL Expert Assistant inside a tokenomics workbench.",
            "You must be rigorous, auditable, and explanatory (not predictive).",
            "Only use the provided context. If something is missing, say what is missing.",
            "Cite assumptions and equations using the provided reference ids like [A1] or [E2].",
            "When Monte Carlo confidence is low or flags exist, explicitly call that out.",
            workings_instruction,
            "Keep answers structured and decision-oriented.",
        ]
    )


def build_user_prompt(context: AssistantContext, user_question: str, show_workings: bool) -> str:
    """Build the user prompt that includes context and reference tables."""
    payload = context.to_prompt_payload()
    payload_json = json.dumps(payload, indent=2, sort_keys=True)

    sections = [
        "Context (JSON):",
        payload_json,
        "",
        _format_assumptions_section(),
        "",
        _format_equations_section(),
        "",
        _format_glossary_section(),
        "",
        "Instructions:",
        "- Use citations like [A1], [E2], [G3] inline when making claims.",
        "- If Monte Carlo flags exist, add a short 'Sensitivity Flags' section.",
    ]

    if show_workings:
        sections.append("- Include a 'Workings' section with equations and the concrete parameter values you used.")

    sections.extend(["", "User question:", user_question])

    return "\n".join(sections)


def _get_default_model() -> str:
    """Resolve a reasonable default model name without hardcoding secrets."""
    return (
        os.getenv("VEFIL_OPENAI_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4.1-mini"
    )


def _create_openai_client(api_key: Optional[str] = None):
    """Create an OpenAI client if the dependency and key are available."""
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover - import failure depends on environment
        return None, f"OpenAI SDK not installed: {exc}"

    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        return None, "Missing OPENAI_API_KEY environment variable."

    try:
        client = OpenAI(api_key=resolved_key)
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"Failed to initialize OpenAI client: {exc}"

    return client, None


def generate_response(
    *,
    user_question: str,
    config: Config,
    result: Optional[SimulationResult],
    validation_warnings: Sequence[ValidationWarning],
    monte_carlo_confidence: Optional[Dict[str, Any]] = None,
    monte_carlo_stats: Optional[Dict[str, Any]] = None,
    show_workings: bool = False,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Generate an assistant response.

    Returns:
        (response_text, error_message)
    """
    context = build_assistant_context(
        config=config,
        result=result,
        validation_warnings=validation_warnings,
        monte_carlo_confidence=monte_carlo_confidence,
        monte_carlo_stats=monte_carlo_stats,
        show_workings=show_workings,
    )

    system_prompt = build_system_prompt(show_workings=show_workings)
    user_prompt = build_user_prompt(context=context, user_question=user_question, show_workings=show_workings)

    client, client_error = _create_openai_client(api_key=api_key)
    if client is None:
        return None, client_error

    resolved_model = model or _get_default_model()

    try:
        response = client.responses.create(
            model=resolved_model,
            temperature=0.2,
            max_output_tokens=900,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": system_prompt,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": user_prompt,
                        }
                    ],
                },
            ],
        )
    except Exception as exc:  # pragma: no cover - network and API errors are environment-dependent
        return None, f"OpenAI API call failed: {exc}"

    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text, None

    # Fallback: try to extract text from the structured output.
    try:
        text_parts: List[str] = []
        for item in response.output:  # type: ignore[attr-defined]
            for content in getattr(item, "content", []):
                if getattr(content, "type", "") == "output_text":
                    text_parts.append(getattr(content, "text", ""))
        combined = "\n".join(part for part in text_parts if part)
        if combined:
            return combined, None
    except Exception:
        pass

    return None, "OpenAI response did not include text output."


def get_default_model() -> str:
    """Public helper for UI defaults."""
    return _get_default_model()
