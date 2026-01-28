"""
Streamlit application for veFIL Tokenomics Workbench.

This is the main entry point for the web-based UI. It provides interactive
controls for configuring simulations, visualizing results, and exploring
various tokenomics scenarios.

Run locally with: streamlit run streamlit_app.py
Deploy to Streamlit Cloud by connecting this repository.
"""

import math
import re
from datetime import date
import html
from typing import Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from vefil.adversarial.crisis import CrisisBehaviorModel, CrisisState
from vefil.adversarial.hedging import DeltaNeutralAnalyzer
from vefil.assistant import generate_response, get_default_model
from vefil.behavior.opportunity_cost import AlternativeYields, OpportunityCostCalculator

# Import from the vefil package (installed via pip install -e .)
from vefil.config.loader import load_config
from vefil.config.schema import Config
from vefil.engine.lending import LendingMarketModel
from vefil.externalities.hardware import HardwareMismatchAnalyzer
from vefil.externalities.network_growth import NetworkGrowthModel
from vefil.externalities.storage_pricing import StoragePricingModel
from vefil.analysis.scenarios import ScenarioRunner, SCENARIO_LIBRARY
from vefil.analysis.sensitivity import SensitivityAnalyzer
from vefil.reporting.charts import (
    create_apy_curve_chart,
    create_capital_flow_chart,
    create_inflation_chart,
    create_locked_impact_chart,
    create_reserve_runway_chart,
    create_supply_chart,
)
from vefil.reporting.export import export_csv, export_json
from vefil.simulation.monte_carlo import MonteCarloRunner
from vefil.simulation.runner import SimulationRunner
from vefil.validation.sanity_checks import (
    SanityChecker,
    ValidationWarning,
    validate_simulation_results,
)
from vefil.analysis.regime import (
    RegimeType,
    analyze_regime,
    analyze_credibility,
    format_inflation_for_display,
    format_regime_for_display,
)

# Page config
st.set_page_config(
    page_title="veFIL Tokenomics Workbench",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Visual System v3.1
# Typography: Geist (unified system) + Geist Mono (data only)
# Aesthetic: Industrial/Utilitarian - compact, sharp, professional
# Colors: Near-black with electric cyan accent, warm amber for warnings
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --void: #08090a;
        --surface: #0e1012;
        --surface-raised: #141618;
        --border: #1e2124;
        --border-accent: #2a2e32;
        --text-primary: #e8eaed;
        --text-secondary: #9aa0a6;
        --text-tertiary: #5f6368;
        --cyan: #00d4ff;
        --cyan-dim: rgba(0, 212, 255, 0.15);
        --amber: #ffab00;
        --amber-dim: rgba(255, 171, 0, 0.12);
        --red: #ff5252;
        --red-dim: rgba(255, 82, 82, 0.12);
        --green: #00e676;
        --green-dim: rgba(0, 230, 118, 0.12);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: var(--void);
        color: var(--text-primary);
        font-size: 13px;
    }

    .main {
        padding: 0.75rem 1rem 2rem 1rem;
        background: var(--void);
    }

    .block-container {
        padding-top: 2.5rem;
        max-width: 100%;
    }

    h1, h2, h3 {
        font-family: 'Inter', -apple-system, sans-serif;
        letter-spacing: -0.02em;
    }

    h1 {
        color: var(--text-primary);
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.5rem;
    }

    h2 {
        color: var(--text-primary);
        font-size: 0.8rem;
        font-weight: 600;
        margin: 1.25rem 0 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    h3 {
        color: var(--text-secondary);
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.75rem 0 0.35rem 0;
    }

    /* Compact header block */
    .hero-shell {
        background: var(--surface);
        border: 1px solid var(--border);
        border-left: 3px solid var(--cyan);
        padding: 0.75rem 1rem;
        margin-bottom: 1rem;
    }

    .hero-kicker {
        font-family: 'SF Mono', 'Consolas', monospace;
        text-transform: uppercase;
        font-size: 0.65rem;
        letter-spacing: 0.1em;
        color: var(--cyan);
        font-weight: 500;
    }

    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.25rem;
        font-weight: 600;
        margin: 0.15rem 0 0.25rem 0;
        color: var(--text-primary);
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        color: var(--text-secondary);
        font-size: 0.8rem;
        line-height: 1.5;
    }

    /* Tight panel styling */
    .panel {
        background: var(--surface);
        border: 1px solid var(--border);
        padding: 0.6rem 0.8rem;
    }

    .panel-title {
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
        color: var(--text-tertiary);
    }

    .stat-value {
        font-family: 'SF Mono', 'Consolas', monospace;
        font-size: 1rem;
        font-weight: 500;
        color: var(--text-primary);
        letter-spacing: -0.01em;
    }

    .stat-label {
        font-size: 0.65rem;
        font-weight: 500;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.15rem;
    }

    .stat-delta {
        font-size: 0.7rem;
        color: var(--text-secondary);
    }

    /* Override Streamlit metrics for compactness */
    [data-testid="stMetricValue"] {
        font-family: 'SF Mono', 'Consolas', monospace;
        font-size: 1.1rem;
        font-weight: 500;
        color: var(--text-primary);
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.65rem;
        font-weight: 600;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.7rem;
    }

    [data-testid="metric-container"] {
        background: var(--surface);
        border: 1px solid var(--border);
        padding: 0.5rem 0.65rem;
    }

    /* Sharp buttons */
    .stButton>button {
        border-radius: 2px;
        border: 1px solid var(--cyan);
        background: var(--cyan-dim);
        color: var(--cyan);
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding: 0.4rem 0.8rem;
        transition: all 0.15s ease;
    }

    .stButton>button:hover {
        background: var(--cyan);
        color: var(--void);
    }

    .stButton>button[kind="primary"] {
        background: var(--cyan);
        color: var(--void);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding: 0.5rem 0.75rem;
        border-radius: 0;
        background: transparent;
        border-bottom: 2px solid transparent;
    }

    .stTabs [aria-selected="true"] {
        border-bottom: 2px solid var(--cyan);
        color: var(--cyan);
    }

    /* Why panel - compact */
    .why-panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-left: 3px solid var(--amber);
        padding: 0.6rem 0.85rem;
        margin-top: 0.75rem;
    }

    .why-panel-title {
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--amber);
        margin-bottom: 0.35rem;
    }

    .why-panel-content {
        color: var(--text-secondary);
        font-size: 0.8rem;
        line-height: 1.55;
    }

    .why-panel-content p {
        margin: 0 0 0.5rem 0;
    }

    .why-panel-content strong {
        color: var(--text-primary);
        font-weight: 600;
    }

    /* Causal chain - inline compact */
    .causal-chain {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.25rem;
        margin-top: 0.5rem;
        font-size: 0.7rem;
    }

    .causal-step {
        background: var(--surface-raised);
        border: 1px solid var(--border-accent);
        padding: 0.2rem 0.45rem;
        color: var(--text-primary);
    }

    .causal-arrow {
        color: var(--cyan);
        font-weight: 600;
    }

    /* Chart disclosure - minimal */
    .chart-disclosure {
        background: var(--surface);
        border: 1px solid var(--border);
        padding: 0.4rem 0.65rem;
        margin-top: 0.35rem;
        font-size: 0.75rem;
    }

    .chart-disclosure summary {
        cursor: pointer;
        color: var(--text-tertiary);
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .chart-disclosure summary:hover {
        color: var(--text-secondary);
    }

    .chart-disclosure[open] summary {
        color: var(--cyan);
        margin-bottom: 0.35rem;
    }

    .chart-disclosure code {
        font-family: 'SF Mono', 'Consolas', monospace;
        background: var(--void);
        padding: 0.1rem 0.3rem;
        font-size: 0.7rem;
        color: var(--cyan);
        border: 1px solid var(--border);
    }

    /* Validation states */
    .validation-warning {
        background: var(--amber-dim);
        border: 1px solid var(--amber);
        border-left: 3px solid var(--amber);
        padding: 0.4rem 0.65rem;
        margin: 0.35rem 0;
        font-size: 0.75rem;
        color: var(--amber);
    }

    .validation-error {
        background: var(--red-dim);
        border: 1px solid var(--red);
        border-left: 3px solid var(--red);
        padding: 0.4rem 0.65rem;
        margin: 0.35rem 0;
        font-size: 0.75rem;
        color: var(--red);
    }

    /* Sidebar tightening */
    [data-testid="stSidebar"] {
        background: var(--surface);
    }

    [data-testid="stSidebar"] .block-container {
        padding: 0.5rem;
    }

    /* Input elements */
    .stSlider [data-baseweb="slider"] {
        margin-top: 0.25rem;
    }

    .stSelectbox [data-baseweb="select"] {
        font-size: 0.8rem;
    }

    .stNumberInput input {
        font-family: 'SF Mono', 'Consolas', monospace;
        font-size: 0.85rem;
    }

    /* Expander compact */
    .streamlit-expanderHeader {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--text-secondary);
    }

    /* DataFrame styling */
    .stDataFrame {
        font-family: 'SF Mono', 'Consolas', monospace;
        font-size: 0.75rem;
    }

    /* Info/Warning/Success boxes */
    .stAlert {
        font-size: 0.8rem;
        padding: 0.5rem 0.75rem;
        border-radius: 2px;
    }

    /* Radio buttons inline */
    .stRadio > div {
        gap: 0.5rem;
    }

    .stRadio label {
        font-size: 0.85rem;
    }

    /* Caption text */
    .stCaption {
        font-size: 0.7rem;
        color: var(--text-tertiary);
    }

    /* Dividers */
    hr {
        border: none;
        border-top: 1px solid var(--border);
        margin: 0.75rem 0;
    }

    /* Chat styling */
    .stChatMessage {
        background: var(--surface);
        border: 1px solid var(--border);
        padding: 0.5rem 0.75rem;
    }

    /* Regime analysis styling */
    .regime-row {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.5rem 0;
        border-bottom: 1px solid var(--border);
    }

    .regime-window {
        width: 140px;
        font-weight: 600;
        color: var(--text-primary);
    }

    .regime-label {
        width: 100px;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
    }

    .regime-label.deflationary { color: var(--green); }
    .regime-label.inflationary { color: var(--red); }
    .regime-label.neutral { color: var(--amber); }

    .regime-inflation {
        width: 80px;
        font-family: 'SF Mono', 'Consolas', monospace;
        font-size: 0.8rem;
    }

    .regime-details {
        flex: 1;
        color: var(--text-secondary);
        font-size: 0.75rem;
    }

    /* Guardrail warning */
    .guardrail-critical {
        background: var(--red-dim);
        border: 2px solid var(--red);
        border-radius: 2px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
    }

    .guardrail-critical-title {
        color: var(--red);
        font-weight: 700;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }

    .guardrail-critical-text {
        color: var(--text-primary);
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'config' not in st.session_state:
    st.session_state.config = load_config()
if 'simulation_result' not in st.session_state:
    st.session_state.simulation_result = None
if 'validation_warnings' not in st.session_state:
    st.session_state.validation_warnings = []
if 'assistant_messages' not in st.session_state:
    st.session_state.assistant_messages = []
if 'assistant_show_workings' not in st.session_state:
    st.session_state.assistant_show_workings = False
if 'assistant_model' not in st.session_state:
    st.session_state.assistant_model = get_default_model()
if 'assistant_mc_confidence' not in st.session_state:
    st.session_state.assistant_mc_confidence = None
if 'assistant_mc_stats' not in st.session_state:
    st.session_state.assistant_mc_stats = None
if 'assistant_mc_hash' not in st.session_state:
    st.session_state.assistant_mc_hash = None
if 'assistant_mc_runs' not in st.session_state:
    st.session_state.assistant_mc_runs = None
if 'baseline_minting_fil_per_year' not in st.session_state:
    st.session_state.baseline_minting_fil_per_year = 0.0

# Display-only circulating denominator adjustments based on Filecoin issuance.
# Sources: Filecoin spec (token allocation + simple minting), Filecoin docs (simple minting totals).
FILECOIN_MAINNET_LAUNCH = date(2020, 10, 15)
FILECOIN_FIL_BASE = 2_000_000_000
FILECOIN_PL_FF_TOTAL = 0.20 * FILECOIN_FIL_BASE  # 15% PL + 5% FF
FILECOIN_SIMPLE_MINT_TOTAL = 330_000_000  # Simple minting total allocation
FILECOIN_SIMPLE_MINT_HALF_LIFE_YEARS = 6.0
FILECOIN_VESTING_START = date(2020, 10, 15)
FILECOIN_VESTING_END = date(2026, 10, 15)
DISPLAY_START_DATE = date(2026, 1, 27)


def _years_since_launch(d: date) -> float:
    if d <= FILECOIN_MAINNET_LAUNCH:
        return 0.0
    return (d - FILECOIN_MAINNET_LAUNCH).days / 365.0


def _simple_minting_cumulative(t_years: float) -> float:
    if t_years <= 0:
        return 0.0
    lam = math.log(2) / FILECOIN_SIMPLE_MINT_HALF_LIFE_YEARS
    return FILECOIN_SIMPLE_MINT_TOTAL * (1 - math.exp(-lam * t_years))


def _linear_vesting_fraction(d: date) -> float:
    if d <= FILECOIN_VESTING_START:
        return 0.0
    if d >= FILECOIN_VESTING_END:
        return 1.0
    total_days = (FILECOIN_VESTING_END - FILECOIN_VESTING_START).days
    if total_days <= 0:
        return 0.0
    return (d - FILECOIN_VESTING_START).days / total_days


def compute_circulating_adjustment(
    start_date: date,
    horizon_months: int,
    baseline_minting_fil_per_year: float = 0.0
) -> Dict[str, float]:
    """Compute display-only circulating adjustment from simple minting + PL/FF vesting."""
    end_ts = pd.Timestamp(start_date) + pd.DateOffset(months=horizon_months)
    end_date = date(end_ts.year, end_ts.month, end_ts.day)

    # Simple minting delta between start and end
    t_start = _years_since_launch(start_date)
    t_end = _years_since_launch(end_date)
    simple_delta = max(0.0, _simple_minting_cumulative(t_end) - _simple_minting_cumulative(t_start))

    # PL + FF vesting delta between start and end
    vest_start = _linear_vesting_fraction(start_date)
    vest_end = _linear_vesting_fraction(end_date)
    vest_delta = max(0.0, (vest_end - vest_start) * FILECOIN_PL_FF_TOTAL)

    baseline_delta = max(0.0, baseline_minting_fil_per_year) * (horizon_months / 12.0)

    return {
        "total": simple_delta + vest_delta + baseline_delta,
        "simple_minting": simple_delta,
        "vesting": vest_delta,
        "baseline": baseline_delta,
    }


@st.cache_data(show_spinner=False)
def _run_monte_carlo_cached(config_dict: Dict[str, Any], num_runs: int, seed: int):
    """Run Monte Carlo with caching to avoid repeated heavy compute."""
    config = Config.from_dict(config_dict)
    runner = MonteCarloRunner(config)
    results = runner.run(num_runs=num_runs, random_seed=seed)
    stats = runner.analyze_results(results)
    confidence = runner.compute_run_confidence(results)
    return stats, confidence


@st.cache_data(show_spinner=False)
def _run_sensitivity_sweep_cached(
    config_dict: Dict[str, Any],
    parameter_name: str,
    num_points: int,
    seed: int
):
    """Run a single parameter sweep with caching."""
    config = Config.from_dict(config_dict)
    analyzer = SensitivityAnalyzer(config)
    return analyzer.run_sweep(parameter_name, num_points=num_points, random_seed=seed)


@st.cache_data(show_spinner=False)
def _run_tornado_cached(config_dict: Dict[str, Any], metric: str, seed: int):
    """Compute tornado chart data with caching."""
    config = Config.from_dict(config_dict)
    analyzer = SensitivityAnalyzer(config)
    return analyzer.compute_tornado(target_metric=metric, random_seed=seed)


def run_simulation():
    """Run the simulation with current config and validation."""
    with st.spinner("Running simulation..."):
        try:
            # Run pre-simulation validation
            checker = SanityChecker(st.session_state.config)
            pre_warnings = checker.check_config_inputs()

            runner = SimulationRunner(st.session_state.config)
            st.session_state.simulation_result = runner.run()

            # Run post-simulation validation
            result = st.session_state.simulation_result
            post_warnings = validate_simulation_results(
                result.config,
                result.states,
                result.metrics_over_time
            )

            # Add conservation errors from simulation as ValidationWarning objects
            conservation_warnings = []
            for error_msg in result.conservation_errors:
                conservation_warnings.append(ValidationWarning(
                    severity="error",
                    category="conservation",
                    message="Per-timestep conservation violation detected",
                    details=error_msg
                ))

            # Combine all warnings (avoid duplicates)
            all_warnings = pre_warnings + [w for w in post_warnings if w not in pre_warnings]
            all_warnings.extend(conservation_warnings)
            st.session_state.validation_warnings = all_warnings

            errors = [w for w in all_warnings if w.severity == "error"]
            if errors:
                st.warning(f"Simulation completed with {len(errors)} validation error(s).")
            else:
                st.success("Simulation completed.")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")


def render_validation_panel():
    """Render validation warnings and errors if any exist."""
    if not st.session_state.validation_warnings:
        return

    warnings = st.session_state.validation_warnings
    errors = [w for w in warnings if w.severity == "error"]
    warns = [w for w in warnings if w.severity == "warning"]

    if errors or warns:
        with st.expander(f"Validation Issues ({len(errors)} errors, {len(warns)} warnings)", expanded=len(errors) > 0):
            if errors:
                st.markdown("### Errors")
                for err in errors:
                    st.markdown(
                        f"""<div class="validation-error">
                            <strong>{err.category.upper()}:</strong> {err.message}
                            {f'<br/><small>{err.details}</small>' if err.details else ''}
                        </div>""",
                        unsafe_allow_html=True
                    )

            if warns:
                st.markdown("### Warnings")
                for warn in warns:
                    st.markdown(
                        f"""<div class="validation-warning">
                            <strong>{warn.category.upper()}:</strong> {warn.message}
                            {f'<br/><small>{warn.details}</small>' if warn.details else ''}
                        </div>""",
                        unsafe_allow_html=True
                    )


def render_chart_disclosure(title: str, assumptions: list, equations: list = None, time_unit: str = "months"):
    """Render an 'Assumptions & Equations' disclosure panel below a chart."""
    def _sanitize_items(items: list) -> list:
        sanitized = []
        for item in items or []:
            text = str(item).strip()
            if re.fullmatch(r"</?\\w+[^>]*>", text):
                continue
            sanitized.append(text)
        return sanitized

    assumptions_clean = _sanitize_items(assumptions)
    equations_clean = _sanitize_items(equations or [])

    assumptions_html = "".join([f"<li>{html.escape(str(a))}</li>" for a in assumptions_clean])
    equations_html = ""
    if equations_clean:
        equations_html = "<div style='margin-top: 0.5rem;'><strong>Equations:</strong><ul style='margin: 0.25rem 0; padding-left: 1.2rem;'>"
        equations_html += "".join([f"<li><code>{html.escape(str(eq))}</code></li>" for eq in equations_clean])
        equations_html += "</ul></div>"

    st.markdown(
        f"""
        <details class="chart-disclosure">
            <summary>ASSUMPTIONS & EQUATIONS</summary>
            <div style="padding: 0.5rem 0 0.25rem 0;">
                <div><strong>Time Unit:</strong> {time_unit}</div>
                <div style="margin-top: 0.35rem;"><strong>Assumptions:</strong></div>
                <ul style="margin: 0.25rem 0; padding-left: 1.2rem; color: var(--text-secondary);">
                    {assumptions_html}
                </ul>
                {equations_html}
            </div>
        </details>
        """,
        unsafe_allow_html=True
    )


def render_chart_with_download(fig, chart_id: str, filename: str = "chart"):
    """Render a Plotly chart with a download button for PNG export."""
    import base64

    st.plotly_chart(fig, width="stretch", key=chart_id)

    try:
        img_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
        b64 = base64.b64encode(img_bytes).decode()

        st.markdown(
            f"""
            <a href="data:image/png;base64,{b64}" download="{filename}.png"
               style="display: inline-block; padding: 0.25rem 0.6rem;
                      background: var(--cyan-dim);
                      border: 1px solid var(--cyan);
                      color: var(--cyan);
                      text-decoration: none;
                      font-size: 0.7rem;
                      font-weight: 600;
                      text-transform: uppercase;
                      letter-spacing: 0.04em;
                      margin-top: 0.35rem;">
                Download PNG
            </a>
            """,
            unsafe_allow_html=True
        )
    except Exception:
        st.caption("Install kaleido for PNG export: pip install kaleido")


def render_header():
    """Render polished header with branding and run button."""
    col1, col2 = st.columns([3, 1], vertical_alignment="bottom")
    with col1:
        st.markdown(
            """
            <div class="hero-shell">
                <div class="hero-kicker">Tokenomics Workbench</div>
                <div class="hero-title">veFIL Parameter Lab</div>
                <div class="hero-subtitle">
                    Design parameters with a clear line of sight to lock demand, inflation dynamics,
                    and reserve sustainability. Every output is traceable to explicit assumptions.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        # Run simulation button at top
        if st.button("Run Simulation", type="primary", width="stretch"):
            run_simulation()


def render_sidebar():
    """Render polished sidebar configuration panel."""
    with st.sidebar:
        st.markdown("## Controls")

        st.markdown("### Scenario Presets")
        preset = st.selectbox(
            "Load Preset",
            ["Base Case", "Conservative", "Optimistic"],
            help="Select a pre-configured scenario. Base Case uses pure reserve emissions from the 300M mining reserve."
        )

        preset_map = {
            "Conservative": "conservative",
            "Optimistic": "optimistic",
        }
        if preset != "Base Case":
            scenario_key = preset_map.get(preset)
            scenario_desc = SCENARIO_LIBRARY.get(scenario_key).description if scenario_key else ""
            st.info(f"Preset '{preset}': {scenario_desc}")
            if st.button("Apply Preset"):
                runner = ScenarioRunner(st.session_state.config)
                scenario = SCENARIO_LIBRARY.get(scenario_key)
                if scenario:
                    st.session_state.config = runner.apply_scenario(scenario)
                    st.session_state.simulation_result = None
                    st.rerun()

        st.markdown("---")

        with st.expander("Advanced Configuration", expanded=False):
            st.markdown("#### Market & Liquidity")
            market_regime_options = {
                "low": "Low Liquidity",
                "medium": "Medium Liquidity",
                "high": "High Liquidity"
            }
            market_regime = st.selectbox(
                "Market Regime",
                list(market_regime_options.keys()),
                format_func=lambda x: market_regime_options[x],
                index=list(market_regime_options.keys()).index(st.session_state.config.market.liquidity_regime)
            )
            st.session_state.config.market.liquidity_regime = market_regime

            volatility_pct = st.slider(
                "Market Volatility (%)",
                0.0, 200.0, float(st.session_state.config.market.volatility) * 100, 5.0,
                help="Annual price volatility"
            )
            st.session_state.config.market.volatility = volatility_pct / 100.0

            growth_rate_pct = st.slider(
                "Network Growth Rate (%)",
                0.0, 50.0, float(st.session_state.config.externalities.network_growth_rate) * 100, 1.0,
                help="Annual network transaction growth rate"
            )
            st.session_state.config.externalities.network_growth_rate = growth_rate_pct / 100.0

            st.markdown("#### Capital Flow")
            net_new_pct = st.slider(
                "Net New Capital (%)",
                0.0, 100.0, float(st.session_state.config.capital_flow.net_new_fraction) * 100, 5.0,
                help="Fraction of lock demand from new capital entering the ecosystem"
            )
            st.session_state.config.capital_flow.net_new_fraction = net_new_pct / 100.0

            recycled_pct = st.slider(
                "Recycled Capital (%)",
                0.0, 100.0, float(st.session_state.config.capital_flow.recycled_fraction) * 100, 5.0,
                help="Fraction of lock demand from existing circulating FIL"
            )
            st.session_state.config.capital_flow.recycled_fraction = recycled_pct / 100.0

            # Cannibalized fraction is derived from net-new + recycled
            total_fraction = (
                st.session_state.config.capital_flow.net_new_fraction +
                st.session_state.config.capital_flow.recycled_fraction
            )
            cannibalized_fraction = max(0.0, 1.0 - total_fraction)
            st.session_state.config.capital_flow.cannibalized_fraction = cannibalized_fraction

            st.caption(f"Cannibalized (derived): {cannibalized_fraction*100:.0f}%")
            if total_fraction > 1.0:
                st.warning("Net-new + recycled exceed 100%. Cannibalized set to 0%.")

            sell_rate_pct = st.slider(
                "Reward Sell Rate (%)",
                0.0, 100.0, float(st.session_state.config.capital_flow.sell_rate) * 100, 5.0,
                help="Fraction of veFIL rewards sold immediately"
            )
            st.session_state.config.capital_flow.sell_rate = sell_rate_pct / 100.0

            st.markdown("#### Alternative Yields")
            ifil_apy_pct = st.slider(
                "iFIL (GLIF) APY (%)",
                0.0, 50.0, float(st.session_state.config.alternatives.ifil_apy) * 100, 1.0,
                help="Competing yield from GLIF liquid staking"
            )
            st.session_state.config.alternatives.ifil_apy = ifil_apy_pct / 100.0
            # GLIF emits iFIL receipts; keep these aligned.
            st.session_state.config.alternatives.glif_apy = st.session_state.config.alternatives.ifil_apy

            defi_apy_pct = st.slider(
                "DeFi APY (%)",
                0.0, 50.0, float(st.session_state.config.alternatives.defi_apy) * 100, 1.0,
                help="Competing yield from DeFi lending protocols"
            )
            st.session_state.config.alternatives.defi_apy = defi_apy_pct / 100.0

            risk_free_pct = st.slider(
                "Risk-Free Rate (%)",
                0.0, 20.0, float(st.session_state.config.alternatives.risk_free_rate) * 100, 0.5,
                help="Baseline risk-free rate (e.g., US Treasury)"
            )
            st.session_state.config.alternatives.risk_free_rate = risk_free_pct / 100.0

            st.markdown("#### Circulating Denominator (Display Only)")
            horizon_months = int(st.session_state.config.simulation.time_horizon_months)
            baseline_minting = st.number_input(
                "Baseline minting (FIL/year, display only)",
                min_value=0.0,
                value=float(st.session_state.baseline_minting_fil_per_year),
                step=1_000_000.0,
                help="Set to 0 by default; only affects share-of-circulating display."
            )
            st.session_state.baseline_minting_fil_per_year = baseline_minting

            adjustment = compute_circulating_adjustment(
                DISPLAY_START_DATE,
                horizon_months,
                baseline_minting_fil_per_year=baseline_minting
            )
            st.caption(
                f"Adds +{adjustment['total']:,.0f} FIL by horizon "
                f"(simple minting +{adjustment['simple_minting']:,.0f}, "
                f"PL/FF vesting +{adjustment['vesting']:,.0f}, "
                f"baseline +{adjustment['baseline']:,.0f})."
            )

        st.markdown("---")

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("### Reserve Emissions")
            # Use percentage for better legibility
            emission_rate_pct = st.slider(
                "Mining Reserve Emission Rate (%)",
                0.0, 20.0, float(st.session_state.config.yield_source.reserve_annual_rate) * 100, 0.5,
                help="Annual emission rate from 300M mining reserve"
            )
            st.session_state.config.yield_source.reserve_annual_rate = emission_rate_pct / 100.0
            # Show the FIL amount this represents
            reserve_amount = st.session_state.config.initial_supply.reserve
            annual_emission_fil = reserve_amount * (emission_rate_pct / 100.0)
            st.caption(f"≈ {annual_emission_fil:,.0f} FIL/year from {reserve_amount:,.0f} FIL reserve")

            st.markdown("### Reward Curve")
            # Curve preset dropdown
            curve_preset = st.selectbox(
                "Curve Preset",
                ["Default (k=1.5)", "Linear (k=1.0)", "Convex (k=2.0)"],
                help="How much extra reward do longer locks get?"
            )
            if curve_preset == "Linear (k=1.0)":
                st.session_state.config.reward_curve.k = 1.0
            elif curve_preset == "Convex (k=2.0)":
                st.session_state.config.reward_curve.k = 2.0
            else:
                st.session_state.config.reward_curve.k = 1.5

            st.session_state.config.reward_curve.k = st.slider(
                "Duration Exponent (k)",
                0.1, 3.0, float(st.session_state.config.reward_curve.k), 0.05,
                help="Higher k rewards longer locks more (k=1 linear, k>1 convex)"
            )
            min_months = st.slider(
                "Min Duration (months)",
                3, 36,
                int(round(st.session_state.config.reward_curve.min_duration_years * 12)),
                3
            )
            max_months = st.slider(
                "Max Duration (months)",
                12, 120,
                int(round(st.session_state.config.reward_curve.max_duration_years * 12)),
                6
            )
            if max_months <= min_months:
                max_months = min_months + 3
            st.session_state.config.reward_curve.min_duration_years = min_months / 12.0
            st.session_state.config.reward_curve.max_duration_years = max_months / 12.0

            st.markdown("### Simulation")
            st.session_state.config.simulation.time_horizon_months = int(st.slider(
                "Time Horizon (months)",
                6, 120, int(st.session_state.config.simulation.time_horizon_months), 6
            ))
            timestep_options = [7, 14, 30, 90]
            current_timestep = int(st.session_state.config.simulation.timestep_days or 30)
            if current_timestep not in timestep_options:
                current_timestep = 30
            st.session_state.config.simulation.timestep_days = int(st.selectbox(
                "Timestep",
                timestep_options,
                format_func=lambda x: f"{x} days",
                index=timestep_options.index(current_timestep)
            ))

        with st.expander("Initial Supply Configuration", expanded=False):
            MAX_TOTAL_SUPPLY = 2_000_000_000  # 2B FIL hard cap

            total_supply = st.number_input(
                "Total Supply (FIL)",
                min_value=0.0,
                max_value=float(MAX_TOTAL_SUPPLY),
                value=min(float(st.session_state.config.initial_supply.total), float(MAX_TOTAL_SUPPLY)),
                step=1_000_000.0,
                help="Maximum: 2,000,000,000 FIL"
            )
            st.session_state.config.initial_supply.total = min(total_supply, MAX_TOTAL_SUPPLY)
            st.caption(f"Current: {st.session_state.config.initial_supply.total:,.0f} FIL")

            circulating = st.number_input(
                "Circulating (FIL)",
                min_value=0.0,
                max_value=float(MAX_TOTAL_SUPPLY),
                value=float(st.session_state.config.initial_supply.circulating),
                step=1_000_000.0
            )
            st.session_state.config.initial_supply.circulating = min(circulating, total_supply)
            st.caption(f"Current: {st.session_state.config.initial_supply.circulating:,.0f} FIL")

            reserve = st.number_input(
                "Mining Reserve (FIL)",
                min_value=0.0,
                max_value=float(MAX_TOTAL_SUPPLY),
                value=float(st.session_state.config.initial_supply.reserve),
                step=1_000_000.0,
                help="FIL held in the mining reserve for veFIL emissions"
            )
            st.session_state.config.initial_supply.reserve = min(reserve, total_supply)
            st.caption(f"Current: {st.session_state.config.initial_supply.reserve:,.0f} FIL")

            lending_pool = st.number_input(
                "Lending Pool (FIL)",
                min_value=0.0,
                max_value=float(MAX_TOTAL_SUPPLY),
                value=float(st.session_state.config.initial_supply.lending_pool),
                step=1_000_000.0
            )
            st.session_state.config.initial_supply.lending_pool = min(lending_pool, total_supply)
            st.caption(f"Current: {st.session_state.config.initial_supply.lending_pool:,.0f} FIL")

            sp_collateral = st.number_input(
                "SP Collateral (FIL)",
                min_value=0.0,
                max_value=float(MAX_TOTAL_SUPPLY),
                value=float(st.session_state.config.initial_supply.sp_collateral),
                step=1_000_000.0
            )
            st.session_state.config.initial_supply.sp_collateral = min(sp_collateral, total_supply)
            st.caption(f"Current: {st.session_state.config.initial_supply.sp_collateral:,.0f} FIL")

            locked_vefil = st.number_input(
                "Initial Locked veFIL (FIL)",
                min_value=0.0,
                max_value=float(MAX_TOTAL_SUPPLY),
                value=float(st.session_state.config.initial_supply.locked_vefil),
                step=1_000_000.0
            )
            st.session_state.config.initial_supply.locked_vefil = min(locked_vefil, total_supply)
            st.caption(f"Current: {st.session_state.config.initial_supply.locked_vefil:,.0f} FIL")

            initial_sum = (
                st.session_state.config.initial_supply.circulating +
                st.session_state.config.initial_supply.reserve +
                st.session_state.config.initial_supply.lending_pool +
                st.session_state.config.initial_supply.sp_collateral +
                st.session_state.config.initial_supply.locked_vefil
            )
            if abs(initial_sum - st.session_state.config.initial_supply.total) > 1e-6:
                st.warning(
                    f"Initial components sum to {initial_sum:,.0f} FIL, "
                    f"which differs from total supply {st.session_state.config.initial_supply.total:,.0f} FIL."
                )
            if st.session_state.config.initial_supply.total > MAX_TOTAL_SUPPLY:
                st.error(f"Total supply cannot exceed {MAX_TOTAL_SUPPLY:,} FIL (Filecoin hard cap).")


def render_why_this_matters(result):
    """Render the 'Why this matters' micro-panel explaining the causal chain."""
    metrics_latest = result.metrics_over_time[-1] if result.metrics_over_time else {}
    effective_inflation = metrics_latest.get('effective_inflation', 0.0) * 100
    net_locks = metrics_latest.get('new_locks', 0.0) - metrics_latest.get('unlocks', 0.0)
    horizon_months = int(st.session_state.config.simulation.time_horizon_months)
    horizon_years = horizon_months / 12.0

    if effective_inflation < 0:
        regime = "deflationary"
        regime_explanation = "Locks are absorbing more FIL than is being emitted, reducing circulating supply."
        dynamic_explanation = "When more FIL is locked than emitted, the circulating supply contracts"
        outcome_color = "var(--green)"
    else:
        regime = "inflationary"
        regime_explanation = "Emissions exceed lock absorption, increasing circulating supply over time."
        dynamic_explanation = "When more FIL is emitted than locked, the circulating supply expands"
        outcome_color = "var(--red)"

    source_desc = "drawn from the 300M mining reserve"

    st.markdown(
        f"""
        <div class="why-panel">
            <div class="why-panel-title">
                WHY THIS MATTERS
            </div>
            <div class="why-panel-content">
                <p>
                    <strong>Core Dynamic:</strong> veFIL incentivizes token holders to lock FIL in exchange for yield.
                    This yield is {source_desc}. {dynamic_explanation}—creating
                    <span style="color: {outcome_color}; font-weight: 600; text-transform: uppercase;">{regime}</span> pressure.
                </p>
                <p style="margin-top: 0.5rem;">
                    <strong>End-of-horizon state (t={horizon_years:.1f}y):</strong> {regime_explanation}
                </p>
                <div class="causal-chain">
                    <span class="causal-step">Yield Rate</span>
                    <span class="causal-arrow">→</span>
                    <span class="causal-step">Lock Demand</span>
                    <span class="causal-arrow">→</span>
                    <span class="causal-step">Net Locks ({net_locks:+,.0f})</span>
                    <span class="causal-arrow">→</span>
                    <span class="causal-step">Eff. Inflation ({effective_inflation:+.2f}%)</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def _milestone_months(horizon_months: int) -> list:
    """Return milestone months to display (3, 6, 12, then yearly)."""
    yearly = list(range(24, horizon_months + 1, 12))
    return [m for m in [3, 6, 12, *yearly] if m <= horizon_months]


def _closest_state_idx(states, target_days: float) -> int:
    """Find the index of the state closest to the target time in days."""
    return min(range(len(states)), key=lambda i: abs(states[i].t - target_days))


def compute_milestone_summary(result, dt_days: float, horizon_months: int) -> pd.DataFrame:
    """Compute milestone summary table with key metrics at each milestone."""
    rows = []
    for months in _milestone_months(horizon_months):
        idx = _closest_state_idx(result.states, months * 30.0)
        state = result.states[idx]
        # Use metrics at same index as state (metrics are computed at end of each step)
        met = result.metrics_over_time[idx] if idx < len(result.metrics_over_time) else result.metrics_over_time[-1]

        # Reserve emissions (inflationary component)
        reserve_emission = float(met.get("emission", 0.0) or 0.0)
        annual_reserve_emission = reserve_emission * (365.25 / dt_days) if dt_days > 0 else 0.0

        # Total rewards (reserve + fee) - what lockers actually receive
        total_reward = float(met.get("total_reward_budget", reserve_emission) or reserve_emission)
        annual_total_reward = total_reward * (365.25 / dt_days) if dt_days > 0 else 0.0

        # Use effective_weighted_locked from metrics for APY calculation
        eff_locked = float(met.get("effective_weighted_locked", 0.0) or 0.0)
        if eff_locked <= 0 and state.locked_vefil > 0:
            eff_locked = state.locked_vefil * 0.5  # conservative fallback

        # APY: annual total reward / effective weighted locked
        apy = (annual_total_reward / eff_locked) if eff_locked > 0 else float("nan")

        net_locks = float(met.get("new_locks", 0.0) or 0.0) - float(met.get("unlocks", 0.0) or 0.0)

        # Lock-to-emission ratio: locked FIL / annual reserve emissions
        # This measures how much lock we get per unit of inflation
        lock_emission_ratio = state.locked_vefil / annual_reserve_emission if annual_reserve_emission > 0 else float("inf")

        # Reserve runway: years until reserve depleted at current rate
        runway_years = state.reserve / annual_reserve_emission if annual_reserve_emission > 0 else float("inf")

        rows.append({
            "Milestone": f"{months} months" if months < 24 else f"Year {months // 12}",
            "Locked (M FIL)": state.locked_vefil / 1e6,
            "Emissions (M FIL/yr)": annual_reserve_emission / 1e6,
            "APY (%)": apy * 100.0,
            "Lock:Emission": lock_emission_ratio,
            "Inflation (%)": met.get("effective_inflation", 0.0) * 100.0,
            "Reserve (M FIL)": state.reserve / 1e6,
            "Runway (yrs)": runway_years,
        })

    return pd.DataFrame(rows)


def render_fip_summary(result):
    """Render a clean, user-friendly summary of key model assumptions."""
    cfg = st.session_state.config

    st.markdown("### Model Assumptions")

    # Calculate max lockable FIL
    max_lockable = cfg.initial_supply.circulating * cfg.simulation.addressable_cap / 1e6

    # Build clean summary
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Initial State**")
        st.markdown(f"""
- Circulating supply: **{cfg.initial_supply.circulating/1e6:.0f}M FIL**
- Mining reserve: **{cfg.initial_supply.reserve/1e6:.0f}M FIL**
- Max lockable: **{max_lockable:.0f}M FIL** ({cfg.simulation.addressable_cap*100:.0f}% of circulating)
        """)

        st.markdown("**Emissions Policy**")
        if cfg.emissions_policy.enabled:
            st.markdown(f"""
- Source: **{cfg.yield_source.type}** (from mining reserve)
- Rate range: **{cfg.emissions_policy.reserve_rate_min*100:.0f}–{cfg.emissions_policy.reserve_rate_max*100:.0f}%/yr** (dynamic)
- APY floor: **{cfg.alternatives.ifil_apy*100:.0f}%** + {cfg.emissions_policy.apy_floor_buffer*100:.0f}% buffer
            """)
        else:
            st.markdown(f"""
- Source: **{cfg.yield_source.type}**
- Rate: **{cfg.yield_source.reserve_annual_rate*100:.1f}%/yr** (fixed)
            """)

    with col2:
        st.markdown("**Lock Parameters**")
        st.markdown(f"""
- Duration range: **{cfg.reward_curve.min_duration_years*12:.0f}–{cfg.reward_curve.max_duration_years*12:.0f} months**
- Reward curve: **k={cfg.reward_curve.k:.1f}** (convex, favors longer locks)
- Relock rate: **{cfg.simulation.relock_fraction_unlocked*100:.0f}%** of unlocks
        """)

        st.markdown("**Competing Yields**")
        st.markdown(f"""
- iFIL/GLIF: **{cfg.alternatives.ifil_apy*100:.0f}%** APY
- DeFi lending: **{cfg.alternatives.defi_apy*100:.0f}%** APY
- Risk-free: **{cfg.alternatives.risk_free_rate*100:.1f}%** APY
        """)


def render_milestone_summary(result):
    """Render the milestone summary table with key metrics at each milestone."""
    if not result or not result.states:
        return

    dt_days = float(st.session_state.config.simulation.timestep_days or 30.0)
    horizon_months = int(st.session_state.config.simulation.time_horizon_months)

    df = compute_milestone_summary(result, dt_days=dt_days, horizon_months=horizon_months)

    st.markdown("### Milestone Summary")
    st.caption(
        "Key metrics at each milestone. **Emissions** = annual FIL emitted from reserve. "
        "**APY** = emissions / effective weighted locked. "
        "**Lock:Emission** = locked FIL / emissions (higher = more lock per unit inflation)."
    )

    # Format DataFrame for display
    display_df = df.copy()
    display_df["Locked (M FIL)"] = display_df["Locked (M FIL)"].apply(lambda x: f"{x:.1f}")
    display_df["Emissions (M FIL/yr)"] = display_df["Emissions (M FIL/yr)"].apply(lambda x: f"{x:.1f}")
    display_df["APY (%)"] = display_df["APY (%)"].apply(lambda x: f"{x:.1f}%")
    display_df["Lock:Emission"] = display_df["Lock:Emission"].apply(
        lambda x: "∞" if x == float("inf") else f"{x:.0f}:1"
    )
    display_df["Inflation (%)"] = display_df["Inflation (%)"].apply(lambda x: f"{x:.2f}%")
    display_df["Reserve (M FIL)"] = display_df["Reserve (M FIL)"].apply(lambda x: f"{x:.1f}")
    display_df["Runway (yrs)"] = display_df["Runway (yrs)"].apply(
        lambda x: "∞" if x == float("inf") else f"{x:.0f}"
    )

    st.dataframe(
        display_df,
        hide_index=True,
        width='stretch',
    )


def render_regime_analysis(result):
    """Render time-frame regime analysis with guardrails.

    This component shows:
    - Regime classification across multiple time windows (3mo, 6mo, 12mo, 24mo, full)
    - Near-term lock guardrails (<1M warnings)
    - Lock progress tracking

    Regime is computed from aggregated flows, not single-step metrics.
    Denominator: circulating at window start (stable reference point).
    """
    if not result or not result.states:
        return

    dt_days = float(st.session_state.config.simulation.timestep_days or 30.0)
    total_supply = result.final_metrics.get('total_supply', 1) or 1
    circulating = result.states[-1].circulating if result.states else 0
    horizon_months = int(st.session_state.config.simulation.time_horizon_months)
    adjustment = compute_circulating_adjustment(
        DISPLAY_START_DATE,
        horizon_months,
        baseline_minting_fil_per_year=float(st.session_state.baseline_minting_fil_per_year)
    )
    circulating_adjustment = adjustment["total"]
    circulating_display = max(0.0, circulating + circulating_adjustment)

    analysis = analyze_regime(
        metrics_over_time=result.metrics_over_time,
        states=result.states,
        total_supply=total_supply,
        circulating=circulating_display,
        dt_days=dt_days
    )

    st.markdown("## Regime Analysis")
    st.caption("Regime can flip between deflationary and inflationary across different time windows. "
               "Analysis uses aggregated flows with circulating supply at window start as denominator.")

    # Guardrails - critical warnings first
    guardrails = analysis.guardrails
    if guardrails.warning_3m_below_1m:
        st.markdown(
            f"""
            <div class="guardrail-critical">
                <div class="guardrail-critical-title">Near-Term Lock Warning</div>
                <div class="guardrail-critical-text">
                    Only <strong>{guardrails.locked_at_3_months:,.0f} FIL</strong> locked at 3 months.
                    Target is &gt;1M FIL to establish meaningful protocol adoption.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    if guardrails.warning_6m_below_1m and not guardrails.warning_3m_below_1m:
        st.warning(f"Near-term lock warning: Only {guardrails.locked_at_6_months:,.0f} FIL locked at 6 months (target: >1M)")

    # Lock progress summary (values from milestone summary table are more complete)
    horizon_months = int(st.session_state.config.simulation.time_horizon_months)
    horizon_str = f"{horizon_months // 12} years" if horizon_months >= 24 else f"{horizon_months} months"
    st.caption(f"Lock metrics at end of {horizon_str} horizon. See Milestone Summary table above for full trajectory.")

    # Window regime table
    st.markdown("### Regime by Time Window")

    for w in analysis.windows:
        label, color = format_regime_for_display(w.regime)
        inflation_str = format_inflation_for_display(w.effective_inflation_annualized)

        regime_class = "deflationary" if w.regime == RegimeType.DEFLATIONARY else (
            "inflationary" if w.regime == RegimeType.INFLATIONARY else "neutral"
        )

        st.markdown(f"""
        <div class="regime-row">
            <div class="regime-window">{w.window_name}</div>
            <div class="regime-label {regime_class}">{label}</div>
            <div class="regime-inflation">{inflation_str}</div>
            <div class="regime-details">
                Emissions: {w.emission_sum:,.0f} FIL | Locks: {w.new_locks_sum:,.0f} FIL | Unlocks: {w.unlocks_sum:,.0f} FIL | Net: {w.net_lock_change:+,.0f} FIL
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Explanation
    st.caption(
        "Effective inflation = (emissions - net_lock_change) / circulating_at_start × (365.25 / window_days). "
        "Deflationary: <-0.5%, Neutral: -0.5% to +0.5%, Inflationary: >+0.5%."
    )

    # Credibility analysis - sensitivity and break conditions
    with st.expander("Sensitivity & Break Conditions", expanded=False):
        credibility = analyze_credibility(analysis.windows, primary_window_name="12 months")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Sensitivity Analysis")
            st.caption("How regime changes under ±10% parameter shifts")

            if credibility.sensitivity_results:
                for sr in credibility.sensitivity_results:
                    flip_indicator = "⚠️ FLIPS" if sr.regime_flipped else "stable"
                    delta_sign = "+" if sr.inflation_delta >= 0 else ""
                    st.markdown(
                        f"**{sr.parameter_name}** +10%: {delta_sign}{sr.inflation_delta*100:.2f}pp → {flip_indicator}"
                    )
            else:
                st.caption("No sensitivity data available")

            # Stability indicator
            stability_colors = {"stable": "green", "marginal": "amber", "fragile": "red"}
            stability_color = stability_colors.get(credibility.regime_stability, "text-secondary")
            st.markdown(f"**Regime Stability:** <span style='color: var(--{stability_color})'>{credibility.regime_stability.upper()}</span>", unsafe_allow_html=True)

        with col2:
            st.markdown("#### Break Conditions")
            st.caption("What would flip the current regime?")

            if credibility.break_conditions:
                for bc in credibility.break_conditions:
                    st.markdown(f"• {bc.description}")
            else:
                st.caption("No break conditions computed")

        st.caption("These are computed from model outputs, not hardcoded thresholds. "
                   "Results depend on the current simulation assumptions.")


def render_dashboard_overview():
    """Render dashboard overview with key metrics."""
    if st.session_state.simulation_result is None:
        st.warning("Run a simulation to view impact metrics.")
        return

    result = st.session_state.simulation_result
    cfg = st.session_state.config
    horizon_months = int(cfg.simulation.time_horizon_months)
    time_horizon_str = f"{horizon_months} months" if horizon_months < 24 else f"{horizon_months // 12} years"

    st.markdown("## Simulation Results")
    st.caption(f"Time horizon: {time_horizon_str}")

    # The table is the hero - show it first
    render_milestone_summary(result)

    # Simplified assumptions summary below the table
    render_fip_summary(result)

    render_why_this_matters(result)

    # Regime analysis component
    render_regime_analysis(result)

    st.markdown("---")


def render_locked_impact():
    """Render the Locked Impact view with overlay chart."""
    if st.session_state.simulation_result is None:
        st.warning("Run a simulation to view the locked impact analysis.")
        return

    result = st.session_state.simulation_result

    st.markdown("### Locked Impact Analysis")
    st.caption("Visualizes the relationship between locked supply growth and effective inflation. "
               "Highlighted regions show spans where effective inflation is below 0%.")

    fig = create_locked_impact_chart(result.states, result.metrics_over_time)
    render_chart_with_download(fig, "locked_impact_chart", "vefil_locked_impact")

    render_chart_disclosure(
        title="Locked Impact",
        assumptions=[
            "Lock demand is driven by APY relative to alternatives",
            "Unlocks occur when lock positions mature (no early exit)",
            "Effective inflation = (emission_sold - net_locks) / circulating_supply (net_locks exclude relocks)",
            "Deflation boundary at 0% effective inflation"
        ],
        equations=[
            "effective_inflation = (emission_sold - new_locks + unlocks) / circulating × (365.25 / dt) (new_locks exclude relocks)",
            "lock_weight(d) = (d / max_duration)^k"
        ],
        time_unit="years"
    )

    # Summary stats (chart tells the story - these are supplementary)
    deflation_steps = sum(1 for m in result.metrics_over_time if m.get('effective_inflation', 0) < 0)
    deflation_pct = (deflation_steps / len(result.metrics_over_time) * 100) if result.metrics_over_time else 0.0

    st.caption(f"Deflation coverage: **{deflation_pct:.0f}%** of timesteps have negative effective inflation.")


def render_policy_designer():
    """Policy Designer tab with polished UI."""
    st.markdown("## Policy Designer")
    st.markdown("Design and preview reward curve parameters")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Reward Curve Parameters")

        k = st.slider(
            "Duration Exponent (k)",
            0.1, 3.0, st.session_state.config.reward_curve.k, 0.1,
            help="Controls how steeply rewards scale with lock duration"
        )

        min_months_policy = st.slider(
            "Minimum Duration (months)",
            3, 36,
            int(round(st.session_state.config.reward_curve.min_duration_years * 12)),
            3,
            key="policy_min_months"
        )

        max_months_policy = st.slider(
            "Maximum Duration (months)",
            12, 120,
            int(round(st.session_state.config.reward_curve.max_duration_years * 12)),
            6,
            key="policy_max_months"
        )

        if max_months_policy <= min_months_policy:
            max_months_policy = min_months_policy + 3

        min_dur = min_months_policy / 12.0
        max_dur = max_months_policy / 12.0

        if st.button("Apply to Config"):
            st.session_state.config.reward_curve.k = k
            st.session_state.config.reward_curve.min_duration_years = min_dur
            st.session_state.config.reward_curve.max_duration_years = max_dur
            st.success("Configuration updated")

    with col2:
        st.markdown("### Reward Curve Preview")

        durations = np.linspace(min_dur, max_dur, 50)
        weights = [(d / max_dur) ** k for d in durations]
        base_apy = st.session_state.config.yield_source.reserve_annual_rate
        avg_weight = sum(weights) / len(weights) if weights else 1
        apys = [base_apy * w / avg_weight for w in weights]

        fig = create_apy_curve_chart(durations.tolist(), apys)
        st.plotly_chart(fig, width="stretch")

        st.caption(f"At k={k:.1f}: 1-year lock gets {weights[len(weights)//5]*100:.0f}% weight, "
                   f"max duration gets 100% weight")


def render_supply_flows():
    """Supply & Flows tab."""
    st.markdown("## Supply & Flows")

    if st.session_state.simulation_result is None:
        st.warning("Run a simulation to view supply and flow metrics.")
        return

    result = st.session_state.simulation_result
    dt_days = float(st.session_state.config.simulation.timestep_days or 30.0)

    tab1, tab2, tab3 = st.tabs(["Supply Over Time", "Inflation Metrics", "Reserve Runway"])

    with tab1:
        st.markdown("### Supply Over Time")

        fig = create_supply_chart(result.states)
        render_chart_with_download(fig, "supply_chart", "vefil_supply")

        render_chart_disclosure(
            title="Supply Chart",
            assumptions=[
                "Conservation: total = circulating + locked + reserve + lending + collateral",
                "No external minting/burning during simulation",
                "Unlocks return FIL to circulating supply"
            ],
            time_unit="years"
        )

    with tab2:
        st.markdown("### Inflation Metrics")

        fig = create_inflation_chart(result.metrics_over_time)
        render_chart_with_download(fig, "inflation_chart", "vefil_inflation")

        render_chart_disclosure(
            title="Inflation Chart",
            assumptions=[
                "Net inflation = change in circulating / previous circulating",
                "Gross emission = emission / previous reserve",
                "Effective inflation accounts for lock absorption"
            ],
            equations=[
                "net_inflation = Δcirculating / circulating_prev × (365.25 / dt)",
                "effective_inflation = (emission_sold - net_locks) / circulating × (365.25 / dt) (net_locks exclude relocks)"
            ],
            time_unit="years"
        )

    with tab3:
        st.markdown("### Reserve Runway Analysis")

        emission_history = [m.get('emission', 0) for m in result.metrics_over_time]
        fig = create_reserve_runway_chart(result.states, emission_history, dt_days=dt_days)
        render_chart_with_download(fig, "runway_chart", "vefil_runway")

        render_chart_disclosure(
            title="Runway Chart",
            assumptions=[
                "Runway = current_reserve / annual_emission_rate",
                "Assumes emission rate stays constant",
                "Critical zone: <5 years, Warning: 5-10 years"
            ],
            time_unit="years"
        )


def render_capital_dynamics():
    """Capital Dynamics tab."""
    st.markdown("## Capital Dynamics")

    if st.session_state.simulation_result is None:
        st.warning("Run a simulation to view capital dynamics.")
        return

    result = st.session_state.simulation_result

    st.markdown("### Capital Flow Decomposition")
    st.caption("Where locked FIL comes from (cumulative over simulation horizon)")

    total_locks = sum(m.get('new_locks', 0) for m in result.metrics_over_time)
    net_new_frac = st.session_state.config.capital_flow.net_new_fraction
    recycled_frac = st.session_state.config.capital_flow.recycled_fraction
    cannibalized_frac = st.session_state.config.capital_flow.cannibalized_fraction

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Net New Capital**")
        st.markdown(f"{total_locks * net_new_frac / 1e6:.0f}M FIL ({net_new_frac*100:.0f}%)")
    with col2:
        st.markdown(f"**Recycled from Idle**")
        st.markdown(f"{total_locks * recycled_frac / 1e6:.0f}M FIL ({recycled_frac*100:.0f}%)")
    with col3:
        st.markdown(f"**From Lending Markets**")
        st.markdown(f"{total_locks * cannibalized_frac / 1e6:.0f}M FIL ({cannibalized_frac*100:.0f}%)")

    st.markdown("---")

    st.markdown("### Capital Flows Over Time")

    fig = create_capital_flow_chart(result.metrics_over_time)
    render_chart_with_download(fig, "capital_flow_chart", "vefil_capital_flows")

    render_chart_disclosure(
        title="Capital Flows",
        assumptions=[
            "New locks split into net-new (market buying), recycled (idle holdings), cannibalized (from lending)",
            "Unlocks return principal to circulating",
            "Net flow = new_locks - unlocks (new_locks exclude relocks)"
        ],
        time_unit="years"
    )


def render_participation():
    """Participation & Opportunity Costs tab."""
    st.markdown("## Participation & Opportunity Costs")

    st.markdown("### Alternative Yield Comparison")
    st.caption("Competing yield options that veFIL must beat to attract capital")

    alternatives = AlternativeYields(
        ifil_apy=st.session_state.config.alternatives.ifil_apy,
        glif_apy=st.session_state.config.alternatives.glif_apy,
        defi_apy=st.session_state.config.alternatives.defi_apy,
        risk_free_rate=st.session_state.config.alternatives.risk_free_rate
    )

    best_alt_name, best_alt_apy = alternatives.get_best_alternative()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"**iFIL/GLIF**")
        st.markdown(f"{alternatives.ifil_apy*100:.0f}% APY")
    with col2:
        st.markdown(f"**DeFi Lending**")
        st.markdown(f"{alternatives.defi_apy*100:.0f}% APY")
    with col3:
        st.markdown(f"**Risk-Free**")
        st.markdown(f"{alternatives.risk_free_rate*100:.1f}% APY")
    with col4:
        st.markdown(f"**Best Alternative**")
        st.markdown(f"{best_alt_name}: {best_alt_apy*100:.0f}%")

    st.markdown("---")

    st.markdown("### Time-Preference Indifference Curve")
    calc = OpportunityCostCalculator(alternatives, volatility=st.session_state.config.market.volatility)

    durations = np.linspace(0.25, 5.0, 20)
    required_apys = [calc.compute_required_apy(d) for d in durations]

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=durations.tolist(),
        y=[r * 100 for r in required_apys],
        mode='lines+markers',
        name='Required APY',
        line=dict(color='#00d4ff', width=2)
    ))
    fig.add_hline(y=best_alt_apy * 100, line_dash="dash", line_color="#ffab00",
                  annotation_text=f"Best Alt: {best_alt_apy*100:.1f}%")
    fig.update_layout(
        title="Required APY vs Lock Duration",
        xaxis_title="Lock Duration (years)",
        yaxis_title="Required APY (%)",
        template="plotly_dark",
        height=350,
        plot_bgcolor="rgba(8, 9, 10, 1)",
        paper_bgcolor="rgba(8, 9, 10, 1)"
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Cohort Analysis")

    cohort_data = []
    for cohort_name, cohort in st.session_state.config.cohorts:
        cohort_data.append({
            "Cohort": cohort_name.replace("_", " ").title(),
            "Size %": f"{cohort.size_fraction*100:.0f}%",
            "Required Premium": f"+{cohort.required_premium*100:.0f}%",
            "Avg Lock Size": f"{cohort.avg_lock_size:,.0f} FIL",
            "Avg Duration": f"{cohort.avg_duration_years:.1f} years"
        })

    st.dataframe(pd.DataFrame(cohort_data), hide_index=True, width="stretch")


def render_lending_impact():
    """Lending Market Impact tab."""
    st.markdown("## Lending Market Impact")

    st.markdown("### Cannibalization Scenarios")

    lending_config = st.session_state.config.lending
    lending_model = LendingMarketModel(
        base_rate=lending_config.base_rate,
        utilization_elasticity=lending_config.utilization_elasticity,
        sp_borrow_demand=lending_config.sp_borrow_demand
    )

    initial_pool = st.session_state.config.initial_supply.lending_pool
    scenarios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    scenario_data = []

    for cannib_rate in scenarios:
        result = lending_model.simulate_cannibalization_scenario(
            initial_pool=initial_pool,
            cannibalization_fraction=cannib_rate
        )
        # Determine stress level based on rate increase
        rate_increase_pct = result['rate_increase_percent']
        if rate_increase_pct < 2:
            stress_level = "Low"
        elif rate_increase_pct < 5:
            stress_level = "Medium"
        else:
            stress_level = "High"

        scenario_data.append({
            "Cannibalization": f"{cannib_rate*100:.0f}%",
            "New Rate": f"{result['final_rate']*100:.1f}%",
            "Rate Change": f"+{result['rate_increase_percent']:.1f}pp",
            "SP Stress Level": stress_level
        })

    st.dataframe(pd.DataFrame(scenario_data), hide_index=True, width="stretch")

    st.caption("Higher cannibalization from lending pools increases borrowing rates for SPs, "
               "potentially creating adverse feedback loops.")


def render_adversarial():
    """Adversarial & Stress Scenarios tab."""
    st.markdown("## Adversarial & Stress Scenarios")

    tab1, tab2 = st.tabs(["Delta-Neutral Hedging", "Crisis Behavior"])

    with tab1:
        st.markdown("### Delta-Neutral Strategy Analysis")

        adversarial_config = st.session_state.config.adversarial
        hedger = DeltaNeutralAnalyzer(
            funding_rate_daily=adversarial_config.hedging_funding_rate,
            borrow_rate_daily=adversarial_config.hedging_borrow_rate
        )

        col1, col2 = st.columns(2)
        with col1:
            lock_amount = st.number_input(
                "Lock Amount (FIL)",
                min_value=1000.0,
                max_value=100_000_000.0,
                value=100_000.0,
                step=10_000.0
            )
            st.caption(f"Current: {lock_amount:,.0f} FIL")
            lock_duration = st.slider("Lock Duration (years)", 0.5, 5.0, 2.0, 0.5)
            vefil_apy_pct = st.slider("veFIL APY (%)", 5.0, 30.0, 15.0, 1.0)
            vefil_apy = vefil_apy_pct / 100.0

        with col2:
            result = hedger.compute_delta_neutral_profit(vefil_apy, lock_amount, lock_duration)

            st.metric("Net APY", f"{result['net_apy']*100:.2f}%")
            st.metric("Annual Hedging Cost", f"{result['annual_hedging_cost']:,.0f} FIL/yr")

            if result['is_profitable']:
                st.success("Strategy is profitable")
            else:
                st.error("Strategy is unprofitable")

    with tab2:
        st.markdown("### Crisis Behavior Prediction")

        crisis_model = CrisisBehaviorModel()

        value_ratio_pct = st.slider("Value Retention (%)", 0.0, 100.0, 50.0, 5.0,
                                help="Current value as percentage of initial value")
        value_ratio = value_ratio_pct / 100.0
        time_in_position = st.slider("Time in Position (years)", 0.0, 5.0, 1.0, 0.25)
        total_lock_duration = st.slider("Total Lock Duration (years)", 1.0, 5.0, 4.0, 0.5)

        # Create CrisisState for prediction
        initial_value = 100000  # Representative value
        crisis_state = CrisisState(
            price_drop_fraction=1.0 - value_ratio,
            time_into_lock_years=time_in_position,
            total_lock_duration_years=total_lock_duration,
            initial_lock_value=initial_value,
            current_lock_value=initial_value * value_ratio
        )

        prediction = crisis_model.predict_behavior(crisis_state)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Behavior", prediction.behavior_type.replace("_", " ").title())
            st.metric("Probability", f"{prediction.probability*100:.0f}%")
        with col2:
            st.markdown("**Expected Actions:**")
            for action in prediction.expected_actions:
                st.markdown(f"- {action}")


def render_externalities():
    """Real-World Externalities tab."""
    st.markdown("## Real-World Externalities")

    tab1, tab2, tab3 = st.tabs(["Storage Pricing", "Hardware Mismatch", "Network Growth"])

    with tab1:
        st.markdown("### Storage Pricing Feedback")

        ext_config = st.session_state.config.externalities
        storage_model = StoragePricingModel(
            s3_price_per_gb_month=ext_config.s3_price_per_gb_month
        )

        lock_percentages = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        circulating = st.session_state.config.initial_supply.circulating
        # Assume representative values for analysis
        network_qap = 10000.0  # TiB
        fil_price = 5.0  # USD

        pricing_data = []
        for pct in lock_percentages:
            lock_amount = circulating * pct
            impact = storage_model.analyze_supply_shock_impact(
                initial_circulating=circulating,
                locked_amount=lock_amount,
                network_qap=network_qap,
                fil_price=fil_price
            )
            # Determine competitiveness status
            if impact['competitiveness_after']['is_competitive']:
                comp_status = "Competitive"
            else:
                comp_status = "Uncompetitive"

            pricing_data.append({
                "Locked %": f"{pct*100:.0f}%",
                "Locked Amount": f"{lock_amount/1e6:.0f}M FIL",
                "Pledge Increase": f"+{impact['pledge_change_percent']:.1f}%",
                "Storage Price Impact": f"+{impact['storage_price_change_percent']:.1f}%",
                "Competitiveness": comp_status
            })

        st.dataframe(pd.DataFrame(pricing_data), hide_index=True, width="stretch")

    with tab2:
        st.markdown("### Hardware Depreciation Mismatch")

        ext_config = st.session_state.config.externalities
        hardware_analyzer = HardwareMismatchAnalyzer(
            hardware_depreciation_years=ext_config.hardware_depreciation_years
        )

        lock_duration = st.slider("Lock Duration (years)", 1.0, 5.0, 3.0, 0.5, key="hw_lock_dur")
        hardware_age = st.slider("Hardware Age at Lock (years)", 0.0, 3.0, 1.0, 0.5)

        analysis = hardware_analyzer.compute_mismatch_risk(lock_duration, hardware_age)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mismatch Years", f"{analysis['mismatch_years']:.1f}")
            st.metric("Risk Level", analysis['risk_level'].title())
        with col2:
            zombie_risk = "Yes" if analysis['is_zombie_risk'] else "No"
            st.metric("Zombie SP Risk", zombie_risk)
            st.metric("Recommended Max Lock", f"{analysis['recommended_max_lock']:.1f} years")

    with tab3:
        st.markdown("### Network Growth Dependency")

        network_config = st.session_state.config.network
        ext_config = st.session_state.config.externalities
        growth_model = NetworkGrowthModel(
            base_transaction_volume=network_config.transaction_volume_base,
            fee_per_transaction=network_config.fee_per_transaction,
            growth_rate=ext_config.network_growth_rate
        )

        # Estimate required annual emission from yield source config
        yield_config = st.session_state.config.yield_source
        reserve = st.session_state.config.initial_supply.reserve
        required_emission = reserve * yield_config.reserve_annual_rate

        growth_rates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

        growth_data = []
        for rate in growth_rates:
            analysis = growth_model.analyze_fee_yield_sustainability(
                required_annual_emission=required_emission,
                growth_rate=rate,
                years=5
            )
            gap = analysis['projected_revenue'] - required_emission
            growth_data.append({
                "Growth Rate": f"{rate*100:.0f}%",
                "Year 5 Fee Revenue": f"{analysis['projected_revenue']/1e6:.1f}M FIL",
                "Covers Emissions": "Yes" if analysis['projected_sustainable'] else "No",
                "Gap/Surplus": f"{gap/1e6:+.1f}M FIL"
            })

        st.dataframe(pd.DataFrame(growth_data), hide_index=True, width="stretch")


def render_compare():
    """Scenario Comparison tab."""
    st.markdown("## Scenario Comparison")

    runner = ScenarioRunner(st.session_state.config)
    scenarios = runner.get_available_scenarios()

    categories = sorted({s.category for s in scenarios.values()})
    selected_categories = st.multiselect(
        "Scenario categories",
        categories,
        default=["calibration", "market"],
        help="Select which scenario categories to display"
    )

    filtered = {
        key: sc for key, sc in scenarios.items()
        if sc.category in selected_categories
    }

    scenario_options = {key: f"{sc.name} — {sc.description}" for key, sc in filtered.items()}
    default_selection = [key for key in scenario_options.keys() if key == "reserve_only"]
    selected = st.multiselect(
        "Scenarios to compare",
        list(scenario_options.keys()),
        format_func=lambda k: scenario_options[k],
        default=default_selection
    )

    include_base = st.checkbox("Include base case (current config)", value=True)

    if selected and st.button("Run Comparison"):
        with st.spinner("Running scenarios..."):
            comparison = runner.compare_scenarios(selected, include_base=include_base)

        # Build comparison table
        rows = []
        for name, summary in comparison.summary.items():
            scenario = comparison.scenarios.get(name)
            display_name = scenario.name if scenario else name
            rows.append({
                "Scenario": display_name,
                "Locked (M)": summary["final_locked"] / 1e6,
                "Locked %": summary["final_locked_share"] * 100,
                "Reserve (M)": summary["final_reserve"] / 1e6,
                "Runway (yr)": summary["reserve_runway_years"],
                "Eff. Infl. (%)": summary["effective_inflation"] * 100,
                "Conservation Errors": summary["conservation_errors"],
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, width="stretch")

        if any(scenarios[name].category == "yield_source" and name != "reserve_only" for name in selected):
            st.info("Non-reserve funding scenarios are shown for background context only.")
    else:
        st.caption("Select one or more scenarios and run the comparison.")


def render_sensitivity():
    """Sensitivity Analysis tab."""
    st.markdown("## Sensitivity Analysis")

    tab_mc, tab_sweep, tab_tornado = st.tabs(["Monte Carlo", "Parameter Sweep", "Tornado"])

    with tab_mc:
        st.markdown("### Monte Carlo (Uncertainty)")

        runs = st.slider(
            "Monte Carlo runs",
            min_value=10,
            max_value=500,
            value=int(st.session_state.config.simulation.monte_carlo_runs),
            step=10
        )
        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=999999,
            value=int(st.session_state.config.simulation.random_seed),
            step=1
        )

        if st.button("Run Monte Carlo"):
            with st.spinner("Running Monte Carlo..."):
                stats, confidence = _run_monte_carlo_cached(
                    st.session_state.config.to_dict(),
                    num_runs=runs,
                    seed=int(seed)
                )
            st.session_state.assistant_mc_stats = stats
            st.session_state.assistant_mc_confidence = confidence
            st.session_state.assistant_mc_runs = runs
            st.session_state.assistant_mc_hash = st.session_state.config.compute_hash()

        if st.session_state.assistant_mc_stats:
            stats = st.session_state.assistant_mc_stats
            confidence = st.session_state.assistant_mc_confidence or {}

            st.markdown("#### Confidence")
            st.write(f"{confidence.get('confidence_level', 'unknown').upper()}: {confidence.get('message', '')}")
            if confidence.get("flags"):
                for flag in confidence.get("flags", []):
                    st.warning(flag)

            st.markdown("#### Outcome Ranges (P5–P95)")
            summary_rows = []
            for key, label in [
                ("final_locked", "Final Locked"),
                ("final_reserve", "Final Reserve"),
                ("reserve_runway_years", "Reserve Runway (yrs)"),
            ]:
                data = stats.get(key, {})
                summary_rows.append({
                    "Metric": label,
                    "P5": data.get("p5", 0),
                    "P50": data.get("p50", 0),
                    "P95": data.get("p95", 0),
                })
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")
        else:
            st.caption("Run Monte Carlo to view uncertainty ranges.")

    with tab_sweep:
        st.markdown("### One-at-a-Time Parameter Sweep")

        param_options = list(SensitivityAnalyzer.DEFAULT_PARAMETERS.keys())
        param_labels = {
            k: SensitivityAnalyzer.DEFAULT_PARAMETERS[k][1] for k in param_options
        }
        parameter = st.selectbox(
            "Parameter",
            param_options,
            format_func=lambda k: param_labels[k]
        )
        num_points = st.slider("Sweep points", 5, 21, 11, 2)
        seed = int(st.session_state.config.simulation.random_seed)

        if st.button("Run Sweep"):
            with st.spinner("Running sweep..."):
                sweep = _run_sensitivity_sweep_cached(
                    st.session_state.config.to_dict(),
                    parameter_name=parameter,
                    num_points=num_points,
                    seed=seed
                )

            df = pd.DataFrame({
                "Value": sweep.sweep_values,
                "Final Locked": sweep.metric_values["final_locked"],
                "Final Reserve": sweep.metric_values["final_reserve"],
                "Runway (yrs)": sweep.metric_values["reserve_runway_years"],
                "Eff. Infl. Final": sweep.metric_values["effective_inflation_final"],
            })
            st.dataframe(df, hide_index=True, width="stretch")

    with tab_tornado:
        st.markdown("### Tornado (Parameter Importance)")
        metric = st.selectbox(
            "Target Metric",
            ["final_locked", "final_reserve", "reserve_runway_years", "effective_inflation_final"],
            format_func=lambda m: {
                "final_locked": "Final Locked",
                "final_reserve": "Final Reserve",
                "reserve_runway_years": "Reserve Runway (yrs)",
                "effective_inflation_final": "Effective Inflation (Final)"
            }[m]
        )

        if st.button("Run Tornado"):
            with st.spinner("Running tornado analysis..."):
                entries = _run_tornado_cached(
                    st.session_state.config.to_dict(),
                    metric=metric,
                    seed=int(st.session_state.config.simulation.random_seed)
                )

            rows = []
            for entry in entries:
                rows.append({
                    "Parameter": entry.parameter_label,
                    "Base": entry.base_value,
                    "Low": entry.low_value,
                    "High": entry.high_value,
                    "Metric@Low": entry.metric_at_low,
                    "Metric@High": entry.metric_at_high,
                    "Impact Range": entry.impact_range,
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render_assistant():
    """AI Assistant tab."""
    st.markdown("## AI Assistant")

    st.caption("Ask questions about the simulation results, get explanations of tokenomics concepts, "
               "or request specific analyses.")

    if st.session_state.assistant_mc_hash and st.session_state.assistant_mc_hash != st.session_state.config.compute_hash():
        st.warning("Monte Carlo stats were generated from a different config. Re-run Monte Carlo for fresh uncertainty context.")

    # Display chat history
    for message in st.session_state.assistant_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about the simulation..."):
        st.session_state.assistant_messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                api_key = None
                try:
                    api_key = st.secrets.get("OPENAI_API_KEY")
                except Exception:
                    api_key = None
                response_text, error = generate_response(
                    user_question=prompt,
                    result=st.session_state.simulation_result,
                    config=st.session_state.config,
                    validation_warnings=st.session_state.validation_warnings,
                    monte_carlo_confidence=st.session_state.assistant_mc_confidence,
                    monte_carlo_stats=st.session_state.assistant_mc_stats,
                    show_workings=st.session_state.assistant_show_workings,
                    model=st.session_state.assistant_model,
                    api_key=api_key,
                )
                if error:
                    if "OpenAI SDK not installed" in error:
                        st.error("OpenAI SDK not installed. Run: `pip install openai`")
                    elif "Missing OPENAI_API_KEY" in error:
                        st.error("Missing OPENAI_API_KEY. Set it via `.streamlit/secrets.toml` or your shell env.")
                    else:
                        st.error(error)
                else:
                    st.markdown(response_text or "")
                    st.session_state.assistant_messages.append(
                        {"role": "assistant", "content": response_text or ""}
                    )


def render_export():
    """Export panel in sidebar."""
    st.markdown("### Export")

    if st.session_state.simulation_result is None:
        st.caption("Run simulation to enable exports")
        return

    output_dir = st.text_input("Export folder", value="exports")
    csv_name = st.text_input("CSV filename", value="vefil_simulation.csv")
    json_name = st.text_input("JSON filename", value="vefil_simulation.json")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Write CSV"):
            try:
                export_path = Path(output_dir)
                export_path.mkdir(parents=True, exist_ok=True)
                csv_path = export_path / csv_name
                export_csv(st.session_state.simulation_result, str(csv_path))
                st.success(f"Saved CSV to {csv_path}")
            except Exception as exc:
                st.error(f"CSV export failed: {exc}")
    with col2:
        if st.button("Write JSON"):
            try:
                export_path = Path(output_dir)
                export_path.mkdir(parents=True, exist_ok=True)
                json_path = export_path / json_name
                export_json(st.session_state.simulation_result, str(json_path))
                st.success(f"Saved JSON to {json_path}")
            except Exception as exc:
                st.error(f"JSON export failed: {exc}")


def main():
    """Main application entry point."""
    render_header()
    render_sidebar()

    # Main content tabs
    main_tabs = st.tabs([
        "Overview",
        "Locked Impact",
        "Advanced"
    ])

    with main_tabs[0]:
        render_dashboard_overview()

    with main_tabs[1]:
        render_locked_impact()

    with main_tabs[2]:
        advanced_tabs = st.tabs([
            "Policy",
            "Supply",
            "Capital",
            "Participation",
            "Lending",
            "Adversarial",
            "Externalities",
            "Compare",
            "Sensitivity",
            "Assistant"
        ])

        with advanced_tabs[0]:
            render_policy_designer()
        with advanced_tabs[1]:
            render_supply_flows()
        with advanced_tabs[2]:
            render_capital_dynamics()
        with advanced_tabs[3]:
            render_participation()
        with advanced_tabs[4]:
            render_lending_impact()
        with advanced_tabs[5]:
            render_adversarial()
        with advanced_tabs[6]:
            render_externalities()
        with advanced_tabs[7]:
            render_compare()
        with advanced_tabs[8]:
            render_sensitivity()
        with advanced_tabs[9]:
            render_assistant()


if __name__ == "__main__":
    main()
