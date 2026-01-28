"""Pydantic schema for configuration validation."""

import hashlib
import json
from typing import Any, Dict, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class InitialSupply(BaseModel):
    """Initial supply state."""
    total: float = Field(gt=0, description="Total FIL supply")
    circulating: float = Field(ge=0, description="Circulating FIL")
    reserve: float = Field(ge=0, description="Mining reserve balance (300M FIL bucket)")
    other_allocations: float = Field(ge=0, default=0.0, description="Other allocations (SAFT, team, foundation, etc.)")
    lending_pool: float = Field(ge=0, description="FIL in lending markets")
    sp_collateral: float = Field(ge=0, description="FIL as SP collateral")
    locked_vefil: float = Field(ge=0, description="Initial locked veFIL")


class YieldSource(BaseModel):
    """Yield source configuration - reserve emissions only."""
    type: Literal["reserve"] = Field(default="reserve", description="Yield source type (reserve only)")
    reserve_annual_rate: float = Field(ge=0, le=1, description="Annual reserve emission rate")


class RewardCurve(BaseModel):
    """Reward curve parameters."""
    k: float = Field(gt=0, le=3, description="Duration exponent")
    max_duration_years: float = Field(gt=0, description="Maximum lock duration")
    min_duration_years: float = Field(gt=0, description="Minimum lock duration")

    @field_validator('min_duration_years')
    @classmethod
    def validate_min_duration(cls, v, info):
        """Ensure min < max duration."""
        if 'max_duration_years' in info.data and v >= info.data['max_duration_years']:
            raise ValueError("min_duration_years must be less than max_duration_years")
        return v


class Market(BaseModel):
    """Market parameters."""
    liquidity_regime: Literal["low", "medium", "high"] = Field(description="Liquidity regime")
    order_book_depth: float = Field(gt=0, description="Order book depth")
    market_multiplier: float = Field(gt=0, description="Price impact multiplier")
    volatility: float = Field(ge=0, description="Annual volatility")


class CapitalFlow(BaseModel):
    """Capital flow assumptions."""
    net_new_fraction: float = Field(ge=0, le=1, description="Fraction of net-new capital")
    recycled_fraction: float = Field(ge=0, le=1, description="Fraction of recycled capital")
    cannibalized_fraction: float = Field(ge=0, le=1, description="Fraction from lending/staking")
    sell_rate: float = Field(ge=0, le=1, description="Fraction of rewards sold")

    @model_validator(mode='after')
    def validate_fractions(self):
        """Ensure capital flow fractions sum to ≤ 1.0."""
        total = (
            self.net_new_fraction +
            self.recycled_fraction +
            self.cannibalized_fraction
        )
        if total > 1.01:  # Allow small floating point errors
            raise ValueError(
                f"Capital flow fractions should sum to ≤ 1.0, got {total:.3f}. "
                f"Net-new: {self.net_new_fraction:.3f}, "
                f"Recycled: {self.recycled_fraction:.3f}, "
                f"Cannibalized: {self.cannibalized_fraction:.3f}"
            )
        return self


class Alternatives(BaseModel):
    """Opportunity cost alternatives."""
    ifil_apy: float = Field(ge=0, description="iFIL lending APY")
    glif_apy: float = Field(
        ge=0,
        description="GLIF facility APY (iFIL receipt; aligned with iFIL APY)"
    )
    defi_apy: float = Field(ge=0, description="DeFi lending APY")
    risk_free_rate: float = Field(ge=0, description="Risk-free baseline rate")

    @model_validator(mode="after")
    def align_glif_with_ifil(self):
        """GLIF emits iFIL receipts; treat GLIF APY as an alias of iFIL APY."""
        self.glif_apy = self.ifil_apy
        return self


class Cohort(BaseModel):
    """User cohort definition."""
    size_fraction: float = Field(ge=0, le=1, description="Fraction of total users")
    required_premium: float = Field(ge=0, description="Required yield premium")
    avg_lock_size: float = Field(gt=0, description="Average lock size")
    avg_duration_years: float = Field(gt=0, description="Average lock duration")


class Cohorts(BaseModel):
    """User cohort definitions."""
    retail: Cohort
    institutional: Cohort
    storage_providers: Cohort
    treasuries: Cohort


class Lending(BaseModel):
    """Lending market parameters."""
    base_rate: float = Field(ge=0, description="Base lending rate")
    utilization_elasticity: float = Field(gt=0, description="Rate elasticity")
    sp_borrow_demand: float = Field(ge=0, description="SP borrowing demand")


class Adversarial(BaseModel):
    """Adversarial scenario parameters."""
    hedging_funding_rate: float = Field(ge=0, description="Daily funding rate for shorts")
    hedging_borrow_rate: float = Field(ge=0, description="Daily borrow rate")
    wrapper_concentration_limit: float = Field(ge=0, le=1, description="Max wrapper concentration")
    crisis_price_drop: float = Field(ge=0, le=1, description="Crisis price drop fraction")


class Externalities(BaseModel):
    """Externalities parameters."""
    network_growth_rate: float = Field(description="Annual network growth rate")
    hardware_depreciation_years: float = Field(gt=0, description="Hardware depreciation period")
    storage_fee_rate: float = Field(ge=0, description="Fee rate per transaction")
    s3_price_per_gb_month: float = Field(gt=0, description="S3 baseline price")


class Simulation(BaseModel):
    """Simulation parameters."""
    time_horizon_months: int = Field(gt=0, description="Simulation time horizon")
    timestep_days: int = Field(gt=0, description="Timestep in days")
    monte_carlo_runs: int = Field(gt=0, description="Number of Monte Carlo runs")
    random_seed: int = Field(description="Random seed for reproducibility")
    # Bootstrap parameters (previously hidden assumptions)
    bootstrap_apy: float = Field(
        ge=0, le=1, default=0.15,
        description="Target APY for initial adoption when no FIL is locked (cold start)"
    )
    adoption_ramp_fraction: float = Field(
        ge=0.1, le=1.0, default=0.3,
        description="Fraction of simulation horizon for adoption ramp-up (e.g., 0.3 = 30%)"
    )
    participation_elasticity: float = Field(
        gt=0, le=10.0, default=1.5,
        description="Elasticity of participation rate to yield premium (higher = more responsive)"
    )
    addressable_cap: float = Field(
        gt=0, le=1.0, default=0.15,
        description="Fraction of circulating supply addressable per cohort (e.g., 0.15 = 15%)"
    )
    max_participation: float = Field(
        gt=0, le=1.0, default=0.95,
        description="Maximum participation cap for lock adoption"
    )
    adjustment_tau_days: float = Field(
        gt=0, default=60.0,
        description="Time constant for partial adjustment in days"
    )
    max_step_lock_fraction: float = Field(
        gt=0, le=1.0, default=0.02,
        description="Maximum share of circulating that can newly lock in a single timestep"
    )
    relock_fraction_unlocked: float = Field(
        ge=0, le=1.0, default=0.35,
        description="Fraction of unlocked FIL that immediately relocks"
    )
    relock_boost_on_competitive_apy: float = Field(
        ge=0, le=1.0, default=0.20,
        description="Additional relock fraction when APY is competitive or lock penetration is high"
    )
    relock_max_fraction: float = Field(
        ge=0, le=1.0, default=0.85,
        description="Maximum relock fraction cap"
    )
    duration_choice_temperature: float = Field(
        gt=0, default=0.35,
        description="Softmax temperature for duration choice (lower = more max-duration)"
    )
    duration_exploration_fraction: float = Field(
        ge=0, le=1.0, default=0.15,
        description="Uniform exploration share in duration choice to model heterogeneity"
    )

    @field_validator("timestep_days", mode="before")
    @classmethod
    def coerce_timestep_days(cls, v):
        """Ensure timestep_days is stored as an int to avoid serializer warnings."""
        if v is None:
            return v
        return int(v)


class Network(BaseModel):
    """Network parameters."""
    transaction_volume_base: float = Field(gt=0, description="Base daily transaction volume")
    fee_per_transaction: float = Field(ge=0, description="FIL per transaction")


class EmissionsPolicy(BaseModel):
    """Emissions stabilizer policy to maintain lock levels and cap inflation."""
    enabled: bool = Field(default=True, description="Enable emissions stabilizer policy")
    lock_growth_target_annual: float = Field(default=0.00, description="Target annual lock growth rate")
    lock_drawdown_threshold: float = Field(default=0.03, description="Max drawdown from peak before triggering (3%)")
    lock_growth_window_months: int = Field(default=6, ge=1, description="Window for measuring lock growth")

    apy_floor_buffer: float = Field(default=0.005, ge=0, description="Buffer over best_alt + avg_premium for APY floor")
    max_effective_inflation: float = Field(default=0.015, description="Target maximum effective inflation (e.g., 0.015 = 1.5%)")
    growth_apy_boost: float = Field(default=0.02, ge=0, description="Extra APY boost when lock growth is below target")
    cap_start_months: int = Field(default=12, ge=0, description="Start inflation cap after N months")
    lock_target_fil: float = Field(default=100_000_000, ge=0, description="Lock target to activate cap")
    cap_ramp_months: int = Field(default=12, ge=0, description="Months to ramp cap down to final target")
    early_max_effective_inflation: float = Field(default=0.02, description="Allowed inflation before cap activates")

    reserve_rate_min: float = Field(default=0.010, ge=0, le=1, description="Minimum reserve emission rate")
    reserve_rate_max: float = Field(default=0.050, ge=0, le=1, description="Maximum reserve emission rate")

    adjustment_speed: float = Field(default=0.35, ge=0, le=1, description="Damping factor for rate adjustments")


class Config(BaseModel):
    """Complete configuration for veFIL workbench."""
    initial_supply: InitialSupply
    yield_source: YieldSource
    reward_curve: RewardCurve
    market: Market
    capital_flow: CapitalFlow
    alternatives: Alternatives
    cohorts: Cohorts
    lending: Lending
    adversarial: Adversarial
    externalities: Externalities
    simulation: Simulation
    network: Network
    emissions_policy: EmissionsPolicy = Field(default_factory=EmissionsPolicy)

    def compute_hash(self) -> str:
        """Compute config hash for reproducibility."""
        config_dict = self.model_dump()
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump()
