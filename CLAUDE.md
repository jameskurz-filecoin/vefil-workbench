# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Evaluation Benchmarks

**These represent the FIP authors' view of healthy program traction. The base case produces outcomes in these ranges under moderate assumptions.**

### Adoption Benchmarks
| Metric | Range | Rationale |
|--------|-------|-----------|
| **Locked @ 3 months** | 30-40M FIL | Reasonable early adoption ramp |
| **Locked @ 6 months** | 60-80M FIL | Continued growth trajectory |
| **Locked @ 12 months** | 100-150M FIL | Meaningful protocol adoption |
| **APY floor** | 8-9%+ | Must stay competitive with alternatives (iFIL ~9%) |
| **Inflation** | Negative to neutral | Deflationary or neutral throughout simulation |
| **Reserve runway** | ~20 years | Sustainable long-term operation |

### Model Philosophy
1. **Pure reserve emissions only** - Yield comes from mining reserve, not fee rewards or hybrid models
2. **APY must remain competitive** - Must exceed alternative yields to attract capital
3. **Emissions policy enabled** - Dynamically adjusts reserve rate to maintain APY floor
4. **Early APY will be higher** - Fewer lockers = bigger share of rewards (early adopter premium)

### On Assumptions
Token-locking mechanisms have limited empirical precedent. Parameters like required yield premium and participation elasticity are informed judgments, not measured values. This tool exists precisely because reasonable people may disagree on inputs. Use the scenario presets and sidebar controls to test alternative assumptions.

---

## Filecoin Tokenomics Context

**This section provides background for understanding the veFIL model.**

### Filecoin Supply Structure (Jan 2026 baseline)
| Bucket | Amount | Notes |
|--------|--------|-------|
| Total Max Supply | 2B FIL | Hard cap, never changes |
| Circulating | ~818M FIL | Liquid, tradeable |
| Mining Reserve | 300M FIL | **Source of veFIL emissions** (proposed in FIP) |
| Other Allocations | ~733M FIL | SAFT, team, foundation vesting |
| SP Collateral | ~100M FIL | Locked as storage provider pledge |
| Lending Markets | ~50M FIL | iFIL, GLIF, DeFi lending pools |

### The 300M Mining Reserve
- The mining reserve is ~300M FIL of unminted supply originally allocated for future miner rewards
- The veFIL FIP proposes using this reserve to fund lock rewards (instead of transaction fees or new inflation)
- This is **pure reserve emissions**: rewards come from existing unminted supply, not new inflation
- At 2% annual emissions, this provides ~20 years of sustainable rewards

### Why Reserve-Only (Not Fee-Based)
1. **No fee model in this FIP** - The proposal specifically uses the mining reserve, not network fees
2. **Deflationary by design** - Reserve emissions don't add to circulating supply; they transfer from unminted → locked
3. **Predictable economics** - Reserve-based rewards are more stable than variable fee revenue
4. **Sustainable runway** - 300M reserve at conservative rates = multi-decade sustainability

### Alternative Yields (Opportunity Cost)
veFIL competes with:
- **iFIL/GLIF**: ~9% liquid staking yield
- **DeFi lending**: ~7% variable
- **Risk-free rate**: ~2.5% stablecoin baseline

veFIL must offer APY above these alternatives (with appropriate premium for lock duration) to attract capital.

---

## Project Overview

veFIL Tokenomics Workbench - a Python/Streamlit simulation platform for modeling vote-escrowed FIL (veFIL) token-locking mechanisms for Filecoin. Simulates supply dynamics, capital flows, behavioral adoption, lending market effects, adversarial scenarios, and real-world externalities.

## Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -e .          # Install in development mode
pip install -e ".[dev]"   # Include dev dependencies (pytest, black, ruff)

# Run app
streamlit run streamlit_app.py

# Tests
pytest -q                 # Quick test run
pytest tests/             # Verbose test output

# Lint/format
black src/ streamlit_app.py --line-length=100
ruff check src/ streamlit_app.py
```

## Streamlit API Notes

**Deprecated parameters (as of Jan 2026):**
- `use_container_width=True` → use `width='stretch'`
- `use_container_width=False` → use `width='content'`

## Architecture

```
Streamlit UI (streamlit_app.py)
         ↓
Simulation Orchestrator (simulation/runner.py, monte_carlo.py)
         ↓
┌────────────────────────────────────────────────┐
│ Engines (engine/)     │ Behavior (behavior/)   │
│ - accounting.py       │ - adoption.py          │
│ - rewards.py          │ - cohorts.py           │
│ - capital_flow.py     │ - opportunity_cost.py  │
│ - lending.py          │                        │
├────────────────────────────────────────────────┤
│ Adversarial (adversarial/) │ Externalities     │
│ - hedging.py, crisis.py    │ - storage_pricing │
│ - wrappers.py              │ - hardware.py     │
└────────────────────────────────────────────────┘
         ↓
Configuration (config/schema.py, defaults.yaml)
```

## Critical Invariants

**Conservation Law**: Every timestep validates all FIL is accounted for (tolerance: 1e-6):
```
total_supply (2B FIL) = circulating + locked_vefil + lending_pool + sp_collateral + reserve + other_allocations + burned_cumulative
```

**Flow Accounting**: The `Flows` dataclass tracks all inter-bucket movements:
- `emission`: Reserve → (circulating or locked via reward_relocks)
- `emission_to_circulating`: Portion of emission sold into circulation
- `reward_relocks`: Rewards that go directly from reserve to locked (bypass circulating)
- `net_locks`: New FIL locked (from circulating, lending, and reward_relocks)
- `lending_cannibalized`: FIL moved from lending pool to locked

**Max-Supply Ledger**: `total_supply` is max supply (2B FIL constant); `reserve` is unissued/unminted FIL.

**Determinism**: Same config + seed = identical results.

## Key Files

- `src/vefil/engine/accounting.py` - Supply tracking, conservation validation
- `src/vefil/simulation/runner.py` - Main discrete-time simulation loop
- `src/vefil/config/defaults.yaml` - Default parameters (tuned to meet objectives above)
- `src/vefil/config/schema.py` - Pydantic models with validation
- `src/vefil/behavior/adoption.py` - Lock demand modeling (logistic adoption curves)

## Adding Features

1. Add engine/model in `src/vefil/` subdirectory
2. Update `Config` schema in `config/schema.py`
3. Add defaults to `config/defaults.yaml`
4. Integrate into `simulation/runner.py`
5. Add UI controls in `streamlit_app.py`
6. **Verify the design objectives are still met after changes**

## Economic Model Notes

**Yield Curve**: `w(d) = (min(d, d_max) / d_max)^k` where k=1.5 (convex, favors long locks)

**Adoption**: Four cohorts (Retail, Institutional, SPs, Treasuries) with distinct required premiums. Lock demand bounded by addressable supply per cohort.

**APY/Demand Circularity**: APY depends on locked amount, which depends on APY. The solver uses fixed-point/partial adjustment within each timestep.

**Emissions Stabilizer Policy** (`emissions_policy` in config): Dynamic feedback control that adjusts `reserve_rate` to maintain APY above the floor. Increases reserve rate as more FIL locks to prevent APY from falling below competing alternatives.

## Current Calibration (v1.0)

The default configuration uses **pure reserve emissions** with dynamic rate adjustment.

### Scenario Comparison

| Scenario | 3 months | 6 months | 12 months | Year 2 | Year 5 |
|----------|----------|----------|-----------|--------|--------|
| **Base Case** | 30M | 79M | 112M | 125M | 144M |
| **Conservative** | 21M | 58M | 82M | 92M | 109M |
| **Optimistic** | 44M | 108M | 156M | 173M | 194M |

### Scenario Drivers

**Base Case**: Default adoption parameters, balanced assumptions.

**Conservative** (lower adoption):
- Higher required premiums (harder to convince users to lock)
- Smaller addressable market (12% vs 17% of circulating)
- Lower relock rates (52% vs 70%)
- Slower adoption response

**Optimistic** (higher adoption):
- Lower required premiums (easier to convince users)
- Larger addressable market (25% of circulating)
- Higher relock rates (85%)
- Faster adoption response

### Inflation Behavior
All scenarios maintain **negative inflation** throughout:
- Early: -20% to -40% (strongly deflationary as reserve → locked)
- Steady-state: ~-0.2% (neutral to slightly deflationary)
- Optimistic has better deflation than Conservative (more locking = more supply absorption)

### Key Configuration
```yaml
yield_source:
  type: "reserve"             # Pure reserve emissions (no fee rewards)
  reserve_annual_rate: 0.02   # 2% starting rate

emissions_policy:
  enabled: true               # Dynamically adjusts rate to maintain APY floor
  reserve_rate_min: 0.02      # Min 2%/yr
  reserve_rate_max: 0.05      # Max 5%/yr
  apy_floor_buffer: 0.01      # Keep APY 1% above alternatives
```

Early APY is higher because fewer lockers share the reward pool - this is mathematically correct and economically sensible (early adopter premium). The emissions policy increases the reserve rate over time as more FIL locks, maintaining stable APY.
