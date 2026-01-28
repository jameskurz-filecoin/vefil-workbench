"""Module B: Yield Source & Reward Mechanics - Model reward generation and distribution.

Key Concepts:
- Reward budget is sourced purely from the 300M mining reserve
- Yield curve: apy(d) = annual_reward_budget * w(d) / effective_weighted_locked
- Weight function: w(d) = (min(d, d_max) / d_max)^k
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LockPosition:
    """Represents a locked position."""
    amount: float  # FIL amount locked
    duration_years: float  # Lock duration in years
    lock_time: float  # Time when locked (in days from start)
    unlock_time: float  # Time when unlockable (in days from start)
    cohort: Optional[str] = None  # Optional cohort identifier


@dataclass
class RewardBudget:
    """Reward budget from reserve emissions."""
    reserve_emission: float = 0.0  # FIL issued from reserve
    total_reward_budget: float = 0.0  # Total available for distribution


@dataclass
class RewardDistribution:
    """Reward distribution result."""
    total_emission: float
    positions: List[LockPosition]
    rewards_per_position: List[float]
    weight_sum: float
    effective_weighted_locked: float  # Actual weighted locked from positions
    gross_apy_by_duration: dict[float, float]


class YieldSourceEngine:
    """Yield source and reward mechanics engine.

    Rewards are funded purely from the 300M mining reserve.
    Reserve emissions reduce `reserve` and increase rewards distributed to lockers.
    """

    def __init__(
        self,
        reserve_annual_rate: float = 0.05,
        reward_curve_k: float = 1.5,
        max_duration_years: float = 5.0,
        **kwargs  # Accept but ignore legacy parameters for backwards compatibility
    ):
        """
        Initialize yield source engine.

        Args:
            reserve_annual_rate: Annual emission rate from reserve
            reward_curve_k: Duration exponent for reward curve
            max_duration_years: Maximum lock duration
        """
        self.reserve_annual_rate = reserve_annual_rate
        self.reward_curve_k = reward_curve_k
        self.max_duration_years = max_duration_years

    def compute_reward_budget(
        self,
        reserve_balance: float,
        dt_days: float = 30.0,
        market_health: float = 1.0,
        **kwargs  # Accept but ignore legacy parameters
    ) -> RewardBudget:
        """
        Compute reward budget from reserve emissions.

        Args:
            reserve_balance: Current reserve balance
            dt_days: Timestep in days
            market_health: Market health indicator for throttling

        Returns:
            RewardBudget with emission amount
        """
        dt_years = dt_days / 365.25
        reserve_emission = reserve_balance * self.reserve_annual_rate * dt_years

        # Throttle emissions during market stress
        if market_health < 0.8:
            reserve_emission *= 0.1

        reserve_emission = max(0.0, reserve_emission)

        return RewardBudget(
            reserve_emission=reserve_emission,
            total_reward_budget=reserve_emission
        )

    def compute_emission(
        self,
        reserve_balance: float,
        dt_days: float = 30.0,
        market_health: float = 1.0,
        **kwargs  # Accept but ignore legacy parameters
    ) -> float:
        """
        Compute emission for a timestep.

        Args:
            reserve_balance: Current reserve balance
            dt_days: Timestep in days
            market_health: Market health indicator for throttling

        Returns:
            Total emission amount
        """
        budget = self.compute_reward_budget(reserve_balance, dt_days, market_health)
        return budget.total_reward_budget

    def compute_lock_weight(self, duration_years: float) -> float:
        """
        Compute lock weight based on duration.

        Formula: w(d) = (d / d_max)^k

        Args:
            duration_years: Lock duration in years

        Returns:
            Lock weight
        """
        if duration_years <= 0:
            return 0.0

        normalized_duration = min(duration_years / self.max_duration_years, 1.0)
        weight = normalized_duration ** self.reward_curve_k
        return weight

    def compute_reward_distribution(
        self,
        positions: List[LockPosition],
        total_emission: float,
        dt_days: float = 30.0
    ) -> RewardDistribution:
        """
        Compute reward distribution across positions.

        Args:
            positions: List of lock positions
            total_emission: Total emission to distribute
            dt_days: Timestep in days (for APY annualization)

        Returns:
            Reward distribution result
        """
        if not positions or total_emission <= 0:
            return RewardDistribution(
                total_emission=total_emission,
                positions=positions or [],
                rewards_per_position=[0.0] * len(positions) if positions else [],
                weight_sum=0.0,
                effective_weighted_locked=0.0,
                gross_apy_by_duration={}
            )

        # Compute weights for each position
        weights = [self.compute_lock_weight(pos.duration_years) for pos in positions]
        weighted_amounts = [pos.amount * w for pos, w in zip(positions, weights)]
        effective_weighted_locked = sum(weighted_amounts)

        # Distribute rewards proportionally
        if effective_weighted_locked > 0:
            rewards = [total_emission * (wa / effective_weighted_locked) for wa in weighted_amounts]
        else:
            rewards = [0.0] * len(positions)

        # Compute gross APY by duration for reference
        gross_apy_by_duration = {}
        duration_groups = {}
        for pos, reward in zip(positions, rewards):
            if pos.duration_years not in duration_groups:
                duration_groups[pos.duration_years] = {'amount': 0.0, 'reward': 0.0}
            duration_groups[pos.duration_years]['amount'] += pos.amount
            duration_groups[pos.duration_years]['reward'] += reward

        for duration, data in duration_groups.items():
            if data['amount'] > 0 and duration > 0:
                # Annualize the reward
                annual_reward = data['reward'] * (365.25 / dt_days)
                gross_apy_by_duration[duration] = annual_reward / data['amount']

        return RewardDistribution(
            total_emission=total_emission,
            positions=positions,
            rewards_per_position=rewards,
            weight_sum=effective_weighted_locked,  # Keep for backward compat
            effective_weighted_locked=effective_weighted_locked,
            gross_apy_by_duration=gross_apy_by_duration
        )

    def compute_effective_weighted_locked(self, positions: List[LockPosition]) -> float:
        """
        Compute effective weighted locked from actual position distribution.

        Formula: effective_weighted_locked = Σ(amount_i * w(duration_i))

        Args:
            positions: List of lock positions

        Returns:
            Effective weighted locked amount
        """
        if not positions:
            return 0.0

        total = 0.0
        for pos in positions:
            weight = self.compute_lock_weight(pos.duration_years)
            total += pos.amount * weight
        return total

    def compute_gross_apy(
        self,
        duration_years: float,
        annual_reward_budget: float,
        effective_weighted_locked: float
    ) -> float:
        """
        Compute gross APY for a given duration using actual weighted locked.

        Formula: apy(d) = annual_reward_budget * w(d) / effective_weighted_locked

        Args:
            duration_years: Lock duration
            annual_reward_budget: Annualized reward budget
            effective_weighted_locked: Actual weighted locked from positions

        Returns:
            Gross APY (annualized)
        """
        if effective_weighted_locked <= 0 or duration_years <= 0:
            return 0.0

        weight = self.compute_lock_weight(duration_years)
        return annual_reward_budget * weight / effective_weighted_locked

    def compute_effective_apy(
        self,
        duration_years: float,
        total_locked: float,
        annual_emission: float,
        slippage_cost: float = 0.0,
        illiquidity_premium: float = 0.0,
        effective_weighted_locked: float = None,
        positions: List[LockPosition] = None
    ) -> float:
        """
        Compute effective APY after frictions.

        Args:
            duration_years: Lock duration
            total_locked: Total FIL locked (for backward compat; prefer positions)
            annual_emission: Annual emission amount
            slippage_cost: Slippage cost (absolute, not percentage)
            illiquidity_premium: Illiquidity premium (as fraction, e.g., 0.05 = 5%)
            effective_weighted_locked: Pre-computed weighted locked (optional)
            positions: Actual positions to compute weighted locked (preferred)

        Returns:
            Effective APY (annualized)
        """
        if duration_years <= 0:
            return 0.0

        # Compute effective weighted locked from positions if provided
        if positions is not None:
            effective_weighted_locked = self.compute_effective_weighted_locked(positions)
        elif effective_weighted_locked is None:
            # Fallback: estimate from total_locked with avg_weight assumption
            if total_locked <= 0:
                return 0.0
            # Use 0.5 as fallback estimate (average of uniform distribution)
            effective_weighted_locked = total_locked * 0.5

        if effective_weighted_locked <= 0:
            return 0.0

        # Compute gross APY
        gross_apy = self.compute_gross_apy(
            duration_years, annual_emission, effective_weighted_locked
        )

        # Adjust for slippage (assuming slippage_cost is per lock amount)
        if duration_years > 0:
            slippage_apy_reduction = slippage_cost / duration_years
        else:
            slippage_apy_reduction = 0.0

        # Effective APY
        effective_apy = gross_apy - slippage_apy_reduction - illiquidity_premium

        return max(0.0, effective_apy)
