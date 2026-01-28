"""Tests for the OpenAI assistant context builder.

These tests avoid network calls and focus on prompt safety and structure.
"""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vefil.config.loader import load_config
from vefil.simulation.runner import SimulationRunner
from vefil.assistant import build_assistant_context, build_system_prompt, build_user_prompt


def _make_small_result():
    """Create a small simulation result for fast tests."""
    config = load_config()
    config.simulation.time_horizon_months = 12
    config.simulation.timestep_days = 30
    runner = SimulationRunner(config)
    result = runner.run(random_seed=42)
    return config, result


def test_assistant_context_includes_core_metrics():
    config, result = _make_small_result()
    ctx = build_assistant_context(
        config=config,
        result=result,
        validation_warnings=[],
        monte_carlo_confidence=None,
        monte_carlo_stats=None,
        show_workings=False,
    )

    assert ctx.core_metrics["final_locked"] >= 0
    assert "effective_inflation_pct" in ctx.core_metrics
    assert ctx.config_full is None


def test_show_workings_includes_full_config():
    config, result = _make_small_result()
    ctx = build_assistant_context(
        config=config,
        result=result,
        validation_warnings=[],
        show_workings=True,
    )

    assert isinstance(ctx.config_full, dict)
    assert "yield_source" in ctx.config_full


def test_prompts_reference_citation_ids():
    config, result = _make_small_result()
    ctx = build_assistant_context(config, result, validation_warnings=[])
    system_prompt = build_system_prompt(show_workings=False)
    user_prompt = build_user_prompt(ctx, "Why is inflation positive?", show_workings=False)

    assert "[A1]" in user_prompt
    assert "[E2]" in user_prompt
    assert "auditable" in system_prompt.lower()
