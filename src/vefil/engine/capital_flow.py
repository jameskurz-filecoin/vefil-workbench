"""Module C: Capital Flow & Market Impact - Model capital movements and price effects."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class CapitalSources:
    """Decomposition of capital sources."""
    net_new: float  # FIL purchased on open market
    recycled: float  # FIL from idle holdings
    cannibalized: float  # FIL from lending/staking


@dataclass
class OrderBookDepth:
    """Order book depth profile."""
    depth: float  # Total depth in FIL equivalent
    # For simplicity, we assume linear depth profile
    # In reality, would be a function depth(price_offset)


class CapitalFlowModel:
    """Model for capital flow decomposition and market impact."""

    def __init__(
        self,
        net_new_fraction: float = 0.4,
        recycled_fraction: float = 0.4,
        cannibalized_fraction: float = 0.2,
        liquidity_regime: Literal["low", "medium", "high"] = "medium",
        order_book_depth: float = 10000000.0  # $10M default
    ):
        """
        Initialize capital flow model.
        
        Args:
            net_new_fraction: Fraction of locks that are net-new capital
            recycled_fraction: Fraction from idle holdings
            cannibalized_fraction: Fraction from lending/staking
            liquidity_regime: Market liquidity regime
            order_book_depth: Order book depth in FIL
        """
        self.net_new_fraction = net_new_fraction
        self.recycled_fraction = recycled_fraction
        self.cannibalized_fraction = cannibalized_fraction

        # Market multipliers based on liquidity regime
        multipliers = {
            "low": 4.0,  # $1 inflow → $4-5 market cap change
            "medium": 1.75,  # $1 inflow → $1.5-2 market cap change
            "high": 1.0  # $1 inflow → $1 market cap change
        }
        self.market_multiplier = multipliers.get(liquidity_regime, 1.75)
        self.liquidity_regime = liquidity_regime
        self.order_book_depth = order_book_depth

    def decompose_capital(
        self,
        total_locks: float,
        lending_pool_available: float = float('inf')
    ) -> CapitalSources:
        """
        Decompose total locks into capital sources.
        
        Args:
            total_locks: Total FIL being locked
            lending_pool_available: Available FIL in lending pool (for cannibalization)
            
        Returns:
            Capital source decomposition
        """
        net_new = total_locks * self.net_new_fraction
        recycled = total_locks * self.recycled_fraction
        cannibalized_raw = total_locks * self.cannibalized_fraction

        # Can't cannibalize more than available
        cannibalized = min(cannibalized_raw, lending_pool_available)

        # Adjust others if cannibalization is capped
        if cannibalized < cannibalized_raw:
            shortfall = cannibalized_raw - cannibalized
            # Redistribute shortfall proportionally
            net_new += shortfall * (self.net_new_fraction / (self.net_new_fraction + self.recycled_fraction))
            recycled += shortfall * (self.recycled_fraction / (self.net_new_fraction + self.recycled_fraction))

        return CapitalSources(
            net_new=net_new,
            recycled=recycled,
            cannibalized=cannibalized
        )

    def compute_price_impact(self, net_new_capital: float) -> float:
        """
        Compute price impact from net-new capital inflows.
        
        Uses inelastic market multiplier model.
        
        Args:
            net_new_capital: Net-new capital inflow
            
        Returns:
            Market cap change (as multiple of inflow)
        """
        if net_new_capital <= 0:
            return 0.0

        price_impact = net_new_capital * self.market_multiplier
        return price_impact

    def compute_slippage(
        self,
        amount: float,
        order_book: OrderBookDepth
    ) -> float:
        """
        Compute buy-side slippage for a given amount.
        
        Simplified model: assumes linear order book depth.
        More sophisticated: would integrate over depth profile.
        
        Args:
            amount: Amount to buy (in FIL)
            order_book: Order book depth profile
            
        Returns:
            Slippage cost in FIL (effective price premium)
        """
        if amount <= 0 or order_book.depth <= 0:
            return 0.0

        # Simple linear slippage model
        # Assumes constant depth, so slippage = (amount / depth) * price_impact_factor
        # Price impact factor approximates how much price moves per unit traded

        # For linear depth, average execution price premium is approximately:
        # (amount / (2 * depth)) when amount << depth
        # More generally, we use a simplified model:

        utilization = amount / order_book.depth
        if utilization <= 0.1:
            # Small trades: minimal slippage
            slippage_fraction = utilization * 0.1
        elif utilization <= 0.5:
            # Medium trades: moderate slippage
            slippage_fraction = 0.01 + (utilization - 0.1) * 0.2
        else:
            # Large trades: significant slippage
            slippage_fraction = 0.09 + (utilization - 0.5) * 0.3

        slippage_cost = amount * slippage_fraction
        return slippage_cost

    def compute_slippage_tax_apy(
        self,
        amount: float,
        lock_duration_years: float,
        order_book: OrderBookDepth
    ) -> float:
        """
        Compute slippage as APY reduction.
        
        Args:
            amount: Lock amount
            lock_duration_years: Lock duration
            order_book: Order book depth
            
        Returns:
            APY reduction due to slippage (as fraction, e.g., 0.01 = 1%)
        """
        if lock_duration_years <= 0 or amount <= 0:
            return 0.0

        slippage_cost = self.compute_slippage(amount, order_book)
        slippage_apy = slippage_cost / (amount * lock_duration_years)
        return slippage_apy

    def compute_net_flow(
        self,
        daily_buy_inflow: float,
        daily_sell_pressure: float
    ) -> float:
        """
        Compute net capital flow.
        
        Args:
            daily_buy_inflow: Daily buy inflows
            daily_sell_pressure: Daily sell pressure from unlocks
            
        Returns:
            Net flow (positive = healthy)
        """
        return daily_buy_inflow - daily_sell_pressure

    def compute_sell_pressure(
        self,
        rewards_vesting: float,
        sell_rate: float = 0.7
    ) -> float:
        """
        Compute sell pressure from reward vesting.
        
        Args:
            rewards_vesting: Rewards being unlocked/vesting
            sell_rate: Fraction of rewards that get sold
            
        Returns:
            Daily sell pressure
        """
        return rewards_vesting * sell_rate

    def compute_buy_inflow(
        self,
        new_lockers: int,
        avg_lock_size: float,
        net_new_fraction: float = None
    ) -> float:
        """
        Compute buy inflow from new lockers.
        
        Args:
            new_lockers: Number of new lockers
            avg_lock_size: Average lock size
            net_new_fraction: Fraction that is net-new (overrides default if provided)
            
        Returns:
            Daily buy inflow
        """
        if net_new_fraction is None:
            net_new_fraction = self.net_new_fraction

        total_locks = new_lockers * avg_lock_size
        buy_inflow = total_locks * net_new_fraction
        return buy_inflow
