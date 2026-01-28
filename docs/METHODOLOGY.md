# veFIL Tokenomics Workbench - Methodology

This document describes the mathematical models, equations, and data assumptions underlying the veFIL simulation.

## Table of Contents

1. [Model Overview](#model-overview)
2. [Core Equations](#core-equations)
3. [Behavioral Models](#behavioral-models)
4. [Data Sources & Assumptions](#data-sources--assumptions)
5. [Validation & Limitations](#validation--limitations)

---

## Model Overview

The veFIL Tokenomics Workbench is a discrete-time simulation that models the dynamics of a vote-escrowed token system for Filecoin. The model operates on monthly timesteps and tracks:

- **Supply components**: Total, circulating, locked, reserve, lending pool, SP collateral
- **Flow dynamics**: Emissions, locks, unlocks, capital sources
- **Behavioral response**: Participation rates based on yield opportunity costs
- **Risk scenarios**: Hedging arbitrage, crisis behavior, wrapper concentration

### Design Philosophy

1. **Conservation Law**: Total supply is invariant; all flows are zero-sum transfers
2. **Rational Actors**: Users respond to economic incentives via utility maximization
3. **Modularity**: Each subsystem is independent and configurable
4. **Transparency**: All assumptions are explicit and adjustable

---

## Core Equations

### 1. Supply Conservation (Fundamental Constraint)

At every timestep, the following must hold:

```
Total Supply = Circulating + Locked + Reserve + Lending Pool + SP Collateral
```

**Validation tolerance**: |error| < 10⁻⁶ FIL

### 2. Lock Weight Function

The weight assigned to a lock determines its share of rewards:

```
w(d) = (d / d_max)^k
```

Where:
- `d` = lock duration in years
- `d_max` = maximum allowed duration (default: 5 years)
- `k` = curvature exponent (default: 1.5)

**Behavior by k**:
| k value | Curve Shape | Implication |
|---------|-------------|-------------|
| k < 1 | Concave | Favors shorter locks |
| k = 1 | Linear | Proportional to duration |
| k > 1 | Convex | Favors longer locks |

### 3. Reward Distribution

Rewards are distributed pro-rata based on weighted stake:

```
reward_i = E × (a_i × w_i) / Σ(a_j × w_j)
```

Where:
- `E` = total emission for the period
- `a_i` = amount locked by participant i
- `w_i` = weight of participant i's lock

### 4. Emission Sources

#### Reserve-Based Emission
```
emission = reserve × annual_rate × (dt / 365.25)
```

Where `dt` is timestep in days. The `annual_rate` is dynamically adjusted by the emissions policy to maintain APY above the floor.

### 5. Inflation Metrics

#### Net Inflation Rate
```
net_inflation = emission / circulating × (365.25 / dt)
```

#### Effective Inflation
Accounts for tokens removed from circulation via locking:

```
net_lock_change = new_locks - unlocks
effective_inflation = (emission - net_lock_change) / circulating × (365.25 / dt)
```

**Interpretation**:
- `effective_inflation > 0`: Net dilution to holders
- `effective_inflation < 0`: Net accretion (deflation via locking)

### 6. Reserve Runway

```
runway_years = reserve / (annual_emission_rate × reserve)
             = 1 / annual_emission_rate
```

At 5% annual rate: runway ≈ 20 years

---

## Behavioral Models

### 1. Adoption Model (Logistic Response)

Participation rate follows a logistic curve based on utility:

```
utility = veFIL_APY - (best_alternative + required_premium)
participation = 1 / (1 + exp(-ε × utility))
```

Where:
- `ε` = participation elasticity (default: 1.5)
- `required_premium` = cohort-specific minimum excess yield

**APY Estimation**:
```
APY = annual_emission × avg_weight / total_locked
```

### 2. Cohort Heterogeneity

Four user segments with distinct preferences:

| Cohort | Size | Required Premium | Avg Lock | Duration | Risk Tolerance |
|--------|------|------------------|----------|----------|----------------|
| Retail | 50% | 8% | 1,000 FIL | 2 yr | Medium |
| Institutional | 20% | 20% | 50,000 FIL | 4 yr | Low |
| Storage Providers | 20% | 12% | 20,000 FIL | 3 yr | Medium |
| Treasuries | 10% | 6% | 100,000 FIL | 5 yr | Low |

### 3. Capital Flow Decomposition

New locks originate from three sources:

```
total_locks = net_new + recycled + cannibalized
```

**Default fractions**:
- Net-new (market purchases): 40%
- Recycled (idle holdings): 40%
- Cannibalized (from lending): 20%

**Market Impact**:
```
slippage = base_slippage × market_multiplier × (amount / order_book_depth)
```

Market multipliers by liquidity regime:
- Low: 4.0x
- Medium: 1.75x
- High: 1.0x

### 4. Opportunity Cost Model

Best alternative yield determines participation threshold:

```
required_APY = max(iFIL, GLIF, DeFi, risk_free) + cohort_premium + duration_penalty
```

Default alternative yields:
- iFIL: 10%
- GLIF: 7%
- DeFi: 12%
- Risk-free: 4%

---

## Data Sources & Assumptions

### High-Confidence Parameters

| Parameter | Value | Source | Notes |
|-----------|-------|--------|-------|
| Total Supply | 2B FIL | Protocol specification | Fixed supply cap |
| Circulating | 600M FIL | On-chain data | Approximate current state |
| Reserve | 300M FIL | Protocol specification | Mining reserve |
| Max Duration | 5 years | Policy choice | ve-token standard |

### Medium-Confidence Parameters

| Parameter | Value | Range | Source |
|-----------|-------|-------|--------|
| Reward Curve k | 1.5 | 1.0 - 2.5 | Comparable protocols (Curve, Balancer) |
| Reserve Rate | 5% | 3% - 10% | Economic modeling |
| Market Volatility | 60% | 40% - 80% | Historical FIL volatility |

### Low-Confidence Parameters (High Leverage)

| Parameter | Value | Range | Notes |
|-----------|-------|-------|-------|
| Net-New Fraction | 40% | 20% - 60% | **Highly uncertain**; no direct data |
| Participation Elasticity | 1.5 | 0.5 - 3.0 | Behavioral assumption |
| Cohort Premiums | 6% - 20% | Varies | Estimated from DeFi behavior |
| DeFi APY | 12% | 5% - 25% | **Highly variable** |

### External Data Dependencies

1. **Filecoin Network Stats**: Transaction volume, active SPs, pledge requirements
2. **DeFi Yields**: iFIL, GLIF, lending protocol rates
3. **Market Data**: FIL price, volatility, order book depth

**Note**: These are STATIC per simulation run. Real values are dynamic.

---

## Validation & Limitations

### Model Validation

1. **Conservation Check**: Verified every timestep (tolerance: 10⁻⁶)
2. **Boundary Tests**: All parameters constrained to valid ranges via Pydantic
3. **Sanity Checks**: Warnings for negative values, NaN, out-of-range results

### Known Limitations

| Limitation | Description | Impact | Mitigation |
|------------|-------------|--------|------------|
| No Price Dynamics | FIL price is exogenous | Cannot capture reflexive effects | Run scenarios with varied yields |
| Static Alternatives | Competing yields are constant | Misses yield competition | Monte Carlo sampling |
| Simplified Behavior | Rational utility maximization | May miss irrational behavior | Elasticity parameter |
| Discrete Time | Monthly timesteps | Misses intra-month dynamics | Use shorter timesteps |
| No Secondary Markets | No veFIL derivatives modeled | Ignores wrapper liquidity | Concentration limits |

### When to Trust Results

**High confidence**:
- Directional effects (e.g., higher emission → more locks)
- Relative comparisons between scenarios
- Conservation and accounting mechanics

**Lower confidence**:
- Absolute participation numbers
- Precise timing of adoption
- Long-horizon (>5yr) projections

### Recommended Usage

1. **Scenario Comparison**: Compare Base Case, Conservative, and Optimistic adoption scenarios
2. **Sensitivity Analysis**: Identify which parameters matter most
3. **Stress Testing**: Check reserve exhaustion, high competition scenarios
4. **Monte Carlo**: Quantify uncertainty via parameter sampling

---

## Appendix: Equation Summary

### State Transition (per timestep)

```
reserve(t+1)    = reserve(t) - emission
locked(t+1)     = locked(t) + new_locks - unlocks
lending(t+1)    = lending(t) - cannibalized
circulating(t+1) = total - locked(t+1) - reserve(t+1) - lending(t+1) - collateral
```

### Key Metrics

```
locked_share = locked / total
effective_inflation = (emission - net_lock_change) / circulating × annualization
reserve_runway = 1 / annual_emission_rate
```

### Adoption Dynamics

```
utility = APY - (max_alternative + premium)
participation = sigmoid(elasticity × utility)
new_locks = Σ(cohort_size × participation × avg_lock_size)
```

