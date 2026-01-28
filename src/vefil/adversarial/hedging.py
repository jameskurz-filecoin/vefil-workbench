"""Delta-neutral hedging analysis - Can users hedge out price risk?"""

from dataclasses import dataclass


@dataclass
class HedgingCosts:
    """Hedging cost components."""
    funding_rate_cost: float  # Cost from perpetual funding rate
    borrow_rate_cost: float  # Cost from borrowing to short
    total_cost: float  # Total hedging cost


class DeltaNeutralAnalyzer:
    """Analyze delta-neutral hedging strategies."""

    def __init__(
        self,
        funding_rate_daily: float = 0.0001,  # 0.01% daily
        borrow_rate_daily: float = 0.001  # 0.1% daily
    ):
        """
        Initialize delta-neutral analyzer.
        
        Args:
            funding_rate_daily: Daily perpetual funding rate
            borrow_rate_daily: Daily borrow rate for shorting
        """
        self.funding_rate_daily = funding_rate_daily
        self.borrow_rate_daily = borrow_rate_daily

    def compute_hedging_costs(
        self,
        amount: float,
        duration_days: float
    ) -> HedgingCosts:
        """
        Compute total hedging costs for a delta-neutral position.
        
        Strategy: Lock 1M FIL + Short 1M FIL on derivatives market
        
        Args:
            amount: Amount to hedge (FIL)
            duration_days: Lock duration in days
            
        Returns:
            Hedging costs
        """
        # Funding rate cost (for perpetuals)
        funding_rate_cost = amount * self.funding_rate_daily * duration_days

        # Borrow rate cost (if borrowing to short)
        borrow_rate_cost = amount * self.borrow_rate_daily * duration_days

        total_cost = funding_rate_cost + borrow_rate_cost

        return HedgingCosts(
            funding_rate_cost=funding_rate_cost,
            borrow_rate_cost=borrow_rate_cost,
            total_cost=total_cost
        )

    def compute_delta_neutral_profit(
        self,
        vefil_apy: float,
        lock_amount: float,
        duration_years: float
    ) -> dict:
        """
        Compute delta-neutral profit (yield after hedging costs).
        
        Args:
            vefil_apy: veFIL APY (as fraction, e.g., 0.15 = 15%)
            lock_amount: Amount locked
            duration_years: Lock duration in years
            
        Returns:
            Dictionary with profit analysis
        """
        duration_days = duration_years * 365.25

        # Annual veFIL yield
        annual_yield = lock_amount * vefil_apy

        # Hedging costs
        costs = self.compute_hedging_costs(lock_amount, duration_days)

        # Annualize hedging costs
        if duration_years > 0:
            annual_hedging_cost = costs.total_cost / duration_years
        else:
            annual_hedging_cost = 0.0

        # Net yield
        net_yield = annual_yield - annual_hedging_cost
        net_apy = net_yield / lock_amount if lock_amount > 0 else 0.0

        is_profitable = net_apy > 0
        risk_free_yield = net_apy if is_profitable else 0.0

        return {
            'vefil_apy': vefil_apy,
            'vefil_annual_yield': annual_yield,
            'annual_hedging_cost': annual_hedging_cost,
            'net_apy': net_apy,
            'net_annual_yield': net_yield,
            'is_profitable': is_profitable,
            'risk_free_yield': risk_free_yield,
            'hedging_costs': costs,
            'skin_in_game_eroded': is_profitable  # If profitable, alignment is eroded
        }
