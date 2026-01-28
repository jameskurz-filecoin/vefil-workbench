# Architecture Documentation

This document provides a comprehensive overview of the veFIL Tokenomics Workbench codebase, designed to help maintainers understand, modify, and extend the system.

## Table of Contents

1. [System Overview](#system-overview)
2. [Module Reference](#module-reference)
3. [Data Flow](#data-flow)
4. [Key Algorithms](#key-algorithms)
5. [Configuration System](#configuration-system)
6. [Extension Guide](#extension-guide)

---

## System Overview

### Design Principles

1. **Conservation Law**: Every timestep validates that all FIL is accounted for (tolerance: 1e-6)
2. **Determinism**: Same configuration + seed produces identical results
3. **Modularity**: Each engine is independent and testable
4. **Economic Grounding**: Lock demand driven by rational opportunity cost comparison
5. **Comprehensive Coverage**: Addresses supply dynamics, behavioral modeling, risk scenarios, and externalities

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit UI Layer                          │
│  (streamlit_app.py - 833 lines)                                │
│  - Parameter controls, visualization, export                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Simulation Orchestrator                      │
│  (simulation/runner.py, monte_carlo.py)                        │
│  - Discrete-time stepping, engine coordination                  │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   Engines     │    │   Behavior    │    │  Adversarial  │
│  (engine/)    │    │  (behavior/)  │    │ (adversarial/)│
│               │    │               │    │               │
│ - accounting  │    │ - adoption    │    │ - hedging     │
│ - rewards     │    │ - cohorts     │    │ - crisis      │
│ - capital_flow│    │ - opp_cost    │    │ - wrappers    │
│ - lending     │    │               │    │               │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Externalities                              │
│  (externalities/)                                               │
│  - storage_pricing, hardware, network_growth                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Configuration Layer                          │
│  (config/schema.py, loader.py, defaults.yaml)                  │
│  - Pydantic validation, YAML loading, 100+ parameters           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Reference

### `config/` - Configuration Management

#### `schema.py` - Pydantic Models

Defines all configuration parameters with validation:

```python
class InitialSupply(BaseModel):
    total: float = 2_000_000_000        # Total FIL supply
    circulating: float = 600_000_000     # Circulating supply
    reserve: float = 300_000_000         # Mining reserve
    lending_pool: float = 50_000_000     # In lending markets
    sp_collateral: float = 100_000_000   # SP collateral
    locked_vefil: float = 0              # Initially locked

class YieldSource(BaseModel):
    type: Literal["reserve"] = "reserve"  # Reserve emissions only
    reserve_annual_rate: float = 0.02     # 2% annual (policy adjusts dynamically)

class RewardCurve(BaseModel):
    k: float = 1.5                       # Duration exponent
    max_duration_years: float = 5.0
    min_duration_years: float = 0.25

# ... 10+ more model classes
```

**Key Validation**: `CapitalFlow` ensures fractions sum to ≤1.0.

#### `loader.py` - YAML Loading

```python
def load_config(yaml_path: str = None) -> Config:
    """Load config from YAML, defaulting to defaults.yaml."""
```

#### `defaults.yaml` - Default Values

100+ parameters organized by category. Edit this file to change defaults.

---

### `engine/` - Core Simulation Engines

#### `accounting.py` - Supply Tracking

**Purpose**: Track all FIL flows with conservation validation.

```python
@dataclass
class SystemState:
    total_supply: float
    circulating: float
    locked_vefil: float
    reserve: float
    lending_pool: float
    sp_collateral: float

class AccountingEngine:
    def apply_transition(self, state: SystemState,
                         emission: float, locks: float,
                         unlocks: float) -> SystemState:
        """Apply state transition with conservation check."""
        # Validates: sum of all buckets = total_supply (±1e-6)
```

**Critical**: The `validate_conservation()` method ensures no FIL is created or destroyed.

#### `rewards.py` - Yield Mechanics

**Purpose**: Calculate rewards based on yield source and lock weights.

```python
class YieldSourceEngine:
    def compute_reward(self, amount: float, duration: float,
                       total_weighted_locks: float) -> float:
        """
        Reward = emission × (amount × weight) / total_weighted_locks

        Weight function: w(d) = (d / d_max)^k
        - k < 1: Concave (favors short locks)
        - k = 1: Linear
        - k > 1: Convex (favors long locks)
        """
```

**Yield Source**:
- `reserve`: Emissions from the 300M mining reserve (dynamic rate adjustment via emissions policy)

#### `capital_flow.py` - Capital Decomposition

**Purpose**: Analyze where locked capital comes from.

```python
class CapitalFlowModel:
    def decompose_capital(self, total_locks: float) -> CapitalSources:
        """
        Decompose into:
        - net_new: Purchased on open market (40% default)
        - recycled: From idle holdings (40% default)
        - cannibalized: From lending/staking (20% default)
        """

    def compute_slippage(self, amount: float,
                         order_book: OrderBookDepth) -> float:
        """Buy-side slippage based on order book depth."""
```

**Market Multipliers**:
- Low liquidity: 4.0x slippage
- Medium: 1.75x
- High: 1.0x

#### `lending.py` - Lending Market Dynamics

**Purpose**: Model cannibalization from lending pools.

```python
class LendingMarketModel:
    def simulate_cannibalization_scenario(self, pool: float,
                                          fraction: float) -> dict:
        """
        Models rate increase when capital leaves lending.
        Rate = base_rate × (1 + elasticity × utilization_change)
        """
```

---

### `behavior/` - Behavioral Models

#### `adoption.py` - Lock Demand

**Purpose**: Model participation based on rational utility comparison.

```python
class AdoptionModel:
    def compute_participation(self, vefil_apy: float,
                              cohort: CohortConfig) -> float:
        """
        Utility = vefil_apy - (best_alternative + required_premium)

        Participation = 1 / (1 + exp(-elasticity × utility))

        Logistic curve creates realistic S-shaped adoption.
        """
```

#### `cohorts.py` - User Segments

Four user segments with distinct behaviors:

| Cohort | Size | Required Premium | Avg Lock | Duration |
|--------|------|------------------|----------|----------|
| Retail | 50% | 8% | 1,000 FIL | 2 years |
| Institutional | 20% | 20% | 50,000 FIL | 4 years |
| Storage Providers | 20% | 12% | 20,000 FIL | 3 years |
| Treasuries | 10% | 6% | 100,000 FIL | 5 years |

#### `opportunity_cost.py` - Alternative Yields

**Purpose**: Compare veFIL against competing opportunities.

```python
class AlternativeYields:
    ifil_apy: float = 0.10      # iFIL lending
    glif_apy: float = 0.07      # GLIF liquid staking
    defi_apy: float = 0.12      # DeFi lending
    risk_free_rate: float = 0.04  # Stablecoin baseline

class OpportunityCostCalculator:
    def compute_required_apy(self, duration: float) -> float:
        """
        Required APY = best_alternative + time_premium × duration

        Accounts for illiquidity cost of long locks.
        """
```

---

### `adversarial/` - Risk Scenarios

#### `hedging.py` - Delta-Neutral Analysis

**Purpose**: Detect if yield can be extracted risk-free.

```python
class DeltaNeutralAnalyzer:
    def compute_delta_neutral_profit(self, vefil_apy: float,
                                     amount: float,
                                     duration: float) -> dict:
        """
        Strategy: Long veFIL + Short perpetuals

        Profit = veFIL yield - (funding_rate + borrow_rate) × duration

        If positive: "Skin in the game" alignment is eroded.
        """
```

**Risk Indicator**: `skin_in_game_eroded = True` if delta-neutral is profitable.

#### `crisis.py` - Crisis Behavior

**Purpose**: Predict behavior during extreme price drops.

```python
class CrisisBehaviorModel:
    def simulate_crisis_scenario(self, lock_amount: float,
                                 duration: float, time_into_lock: float,
                                 price_drop: float) -> dict:
        """
        Three behaviors:
        - diamond_hands: Hold through crash
        - desperate_extraction: Early unlock (penalty)
        - abandon: Exit entirely

        Probability based on sunk cost, time remaining, utility shift.
        """
```

#### `wrappers.py` - Liquid Wrapper Risks

**Purpose**: Analyze concentration risks from liquid veFIL wrappers.

```python
class LiquidWrapperAnalyzer:
    def analyze_concentration_risk(self, wrapper_share: float) -> dict:
        """
        Risks increase as single wrapper dominates:
        - Governance capture
        - Systemic counterparty risk
        - Oracle manipulation
        """
```

---

### `externalities/` - Real-World Effects

#### `storage_pricing.py` - Supply Shock Impact

**Purpose**: Model how veFIL affects storage costs.

```python
class StoragePricingModel:
    def analyze_supply_shock_impact(self, initial_circ: float,
                                    locked: float, network_qap: float,
                                    fil_price: float) -> dict:
        """
        Lock-induced supply reduction → pledge cost increase
        → storage price increase → competitiveness vs S3

        Checks: fil_price < s3_price ($0.023/GB/month)
        """
```

#### `hardware.py` - Depreciation Mismatch

**Purpose**: Detect 5-year lock vs 3-year hardware cycle risk.

```python
class HardwareMismatchAnalyzer:
    def compute_mismatch_risk(self, lock_duration: float,
                              hardware_age: float) -> dict:
        """
        Zombie SP Risk: Hardware obsolete before lock expires.

        mismatch_years = lock_duration - (hw_cycle - hw_age)

        Risk levels: none, low, medium, high
        """
```

#### `network_growth.py` - Network Growth Analysis

**Purpose**: Analyze network growth trends and their impact on veFIL economics.

```python
class NetworkGrowthModel:
    def analyze_growth_trajectory(self) -> dict:
        """
        Model network transaction volume growth over time.
        Used for externalities analysis (not for emissions funding).
        """
```

---

### `simulation/` - Orchestration

#### `runner.py` - Main Executor

```python
class SimulationRunner:
    def run(self) -> SimulationResult:
        """
        Main simulation loop:

        for month in range(time_horizon):
            1. Compute emissions from yield source
            2. Model adoption/participation
            3. Process locks and unlocks
            4. Apply state transition (with conservation check)
            5. Record metrics

        Returns: states[], metrics_over_time[], final_metrics{}
        """
```

**Key Methods**:
- `_compute_emissions()`: Get FIL to distribute
- `_model_participation()`: Behavioral adoption
- `_apply_state_transition()`: Conservation-validated update

#### `monte_carlo.py` - Parameter Sweeping

```python
class MonteCarloRunner:
    def run(self, n_runs: int = 100) -> MonteCarloResult:
        """
        Samples 8+ parameters from distributions:
        - network_growth: ±5%
        - alternative_yields: ±2-3%
        - capital_flow: ±10%
        - volatility: ±10%

        Returns: confidence intervals, sensitivity analysis
        """
```

---

### `reporting/` - Output Generation

#### `charts.py` - Plotly Visualizations

```python
def create_supply_chart(states: List[SystemState]) -> go.Figure:
    """Stacked area chart of supply components."""

def create_inflation_chart(metrics: List[dict]) -> go.Figure:
    """Line chart of net/gross/effective inflation."""

def create_capital_flow_chart(metrics: List[dict]) -> go.Figure:
    """Sankey or stacked bar of capital sources."""

def create_reserve_runway_chart(states, emissions) -> go.Figure:
    """Reserve depletion projection."""

def create_apy_curve_chart(durations, values) -> go.Figure:
    """APY vs lock duration curve."""
```

#### `export.py` - Data Export

```python
def export_csv(result: SimulationResult, path: str):
    """Export time series to CSV for spreadsheet analysis."""

def export_json(result: SimulationResult, path: str):
    """Export full state including config hash for reproducibility."""

def export_html_report(result: SimulationResult, path: str):
    """Generate standalone HTML report with embedded charts."""
```

---

## Data Flow

### Simulation Lifecycle

```
1. INITIALIZATION
   Config → load_config() → Config object
   Config → AccountingEngine → Initial SystemState

2. EACH TIMESTEP (monthly)
   SystemState + Config → YieldSourceEngine → emission amount
   emission + alternatives → AdoptionModel → participation rates
   participation → CapitalFlowModel → lock amounts by source
   state + locks + unlocks → AccountingEngine → new SystemState
   new state → validate_conservation() → pass/fail

3. POST-PROCESSING
   states[] → reporting/charts.py → Plotly figures
   states[] + metrics[] → reporting/export.py → CSV/JSON/HTML

4. MONTE CARLO (optional)
   Config → MonteCarloRunner → 100 runs with sampled parameters
   runs[] → confidence intervals, sensitivity analysis
```

### State Transitions

```
SystemState(t) ──────────────────────────────────────→ SystemState(t+1)
    │                                                       ▲
    │ emission = yield_source.compute_emission()            │
    │ locks = adoption.compute_locks()                      │
    │ unlocks = scheduled_unlocks[t]                        │
    │                                                       │
    └──→ accounting.apply_transition(emission, locks, unlocks)
```

---

## Key Algorithms

### 1. Lock Weight Function

```
w(d) = (d / d_max)^k

where:
  d = lock duration (years)
  d_max = maximum duration (5 years default)
  k = exponent (1.5 default)

k < 1: Concave → favors short locks
k = 1: Linear → proportional
k > 1: Convex → favors long locks (default behavior)
```

### 2. Reward Distribution

```
reward_i = emission × (amount_i × w(d_i)) / Σ(amount_j × w(d_j))

Pro-rata based on weighted stake.
```

### 3. Adoption Logistic Curve

```
utility = vefil_apy - (best_alternative + required_premium)
participation = 1 / (1 + exp(-elasticity × utility))

Creates realistic S-curve adoption dynamics.
```

### 4. Conservation Validation

```
assert |total - (circulating + locked + reserve + lending + collateral)| < 1e-6

Runs every timestep. Failure = bug in accounting logic.
```

---

## Configuration System

### Parameter Hierarchy

```yaml
initial_supply:     # Starting state (6 params)
yield_source:       # Reward mechanics (4 params)
reward_curve:       # Lock incentives (3 params)
market:             # Market conditions (4 params)
capital_flow:       # Capital sources (4 params)
alternatives:       # Competing yields (4 params)
cohorts:            # User segments (4 × 4 params)
lending:            # Lending market (3 params)
adversarial:        # Risk scenarios (4 params)
externalities:      # Real-world effects (4 params)
simulation:         # Run parameters (4 params)
network:            # Network params (2 params)
```

### Config Hash

```python
config.compute_hash()  # Returns deterministic hash of all parameters
```

Used for reproducibility - same hash = same configuration.

---

## Extension Guide

### Adding a New Engine

1. **Create module** in appropriate directory:
   ```python
   # src/vefil/engine/new_engine.py
   class NewEngine:
       def __init__(self, param1: float, param2: float):
           self.param1 = param1
           self.param2 = param2

       def compute(self, state: SystemState) -> float:
           """Document what this computes."""
           return result
   ```

2. **Add config schema**:
   ```python
   # config/schema.py
   class NewEngineConfig(BaseModel):
       param1: float = Field(default=1.0, ge=0)
       param2: float = Field(default=0.5, ge=0, le=1)
   ```

3. **Add defaults**:
   ```yaml
   # config/defaults.yaml
   new_engine:
     param1: 1.0
     param2: 0.5
   ```

4. **Integrate into runner**:
   ```python
   # simulation/runner.py
   from vefil.engine.new_engine import NewEngine

   class SimulationRunner:
       def __init__(self, config: Config):
           self.new_engine = NewEngine(
               config.new_engine.param1,
               config.new_engine.param2
           )
   ```

5. **Add UI controls**:
   ```python
   # streamlit_app.py
   st.slider("Param 1", 0.0, 2.0, config.new_engine.param1)
   ```

### Adding a New Visualization

1. **Create chart function**:
   ```python
   # reporting/charts.py
   def create_new_chart(data: List[dict]) -> go.Figure:
       fig = go.Figure()
       fig.add_trace(go.Scatter(...))
       fig.update_layout(title="New Chart")
       return fig
   ```

2. **Add to UI**:
   ```python
   # streamlit_app.py
   from vefil.reporting.charts import create_new_chart

   fig = create_new_chart(result.metrics_over_time)
   st.plotly_chart(fig)
   ```

---

## Testing Strategy

### Unit Tests

Each engine should have corresponding tests:

```python
# tests/test_accounting.py
def test_conservation():
    engine = AccountingEngine()
    state = engine.apply_transition(initial, emission=100, locks=50, unlocks=30)
    assert engine.validate_conservation(state)

def test_emission_bounds():
    # Emission should never exceed reserve
    pass
```

### Integration Tests

```python
# tests/test_simulation.py
def test_full_run():
    config = load_config()
    runner = SimulationRunner(config)
    result = runner.run()

    assert len(result.states) == config.simulation.time_horizon_months
    assert result.final_metrics['final_reserve'] >= 0
```

### Property-Based Tests

```python
# tests/test_properties.py
from hypothesis import given, strategies as st

@given(st.floats(0.1, 5.0))
def test_weight_monotonic(duration):
    """Longer locks should always have higher weight."""
    w1 = compute_weight(duration, k=1.5)
    w2 = compute_weight(duration + 0.1, k=1.5)
    assert w2 >= w1
```

---

## Performance Considerations

### Memory

- States are stored as dataclasses (low overhead)
- Monte Carlo runs are independent (parallelizable)
- Charts generated on-demand (not pre-computed)

### CPU

- Single simulation: <1 second for 60-month horizon
- Monte Carlo (100 runs): ~30 seconds
- Bottleneck: NumPy array operations in rewards calculation

### Streamlit Caching

```python
@st.cache_data
def run_cached_simulation(config_hash: str, config: Config):
    """Cache simulation results by config hash."""
    return SimulationRunner(config).run()
```

---

## Glossary

| Term | Definition |
|------|------------|
| **veFIL** | Vote-escrowed FIL - locked tokens with time-weighted voting power |
| **Conservation** | Total FIL = sum of all buckets (invariant) |
| **Cohort** | User segment with distinct behavior parameters |
| **Cannibalization** | Capital moving from lending to veFIL |
| **Delta-neutral** | Position hedged against price movements |
| **Zombie SP** | Storage provider with obsolete hardware but locked collateral |
| **Runway** | Years until reserve exhaustion at current emission rate |

