"""Opportunity cost analysis and alternative yield comparison."""

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class AlternativeYields:
    """Alternative yield options."""
    ifil_apy: float = 0.10  # iFIL lending
    glif_apy: float = 0.10  # GLIF emits iFIL receipts (alias of iFIL)
    defi_apy: float = 0.12  # DeFi lending
    risk_free_rate: float = 0.04  # Stablecoin baseline

    def __post_init__(self) -> None:
        """Keep GLIF aligned with iFIL since GLIF issues iFIL receipts."""
        self.glif_apy = self.ifil_apy

    def get_best_alternative(self) -> tuple[str, float]:
        """
        Get the best alternative yield.
        
        Returns:
            (name, apy) tuple
        """
        alternatives = {
            "ifil": self.ifil_apy,
            "defi": self.defi_apy,
            "risk_free": self.risk_free_rate
        }
        best_name = max(alternatives, key=alternatives.get)
        best_apy = alternatives[best_name]
        return best_name, best_apy


class OpportunityCostCalculator:
    """Calculate opportunity costs and required premiums."""

    def __init__(
        self,
        alternatives: AlternativeYields,
        volatility: float = 0.6,
        alpha: float = 0.02,  # Illiquidity premium coefficient
        beta: float = 1.5  # Duration preference exponent
    ):
        """
        Initialize opportunity cost calculator.
        
        Args:
            alternatives: Alternative yield options
            volatility: Annual volatility for risk calculations
            alpha: Illiquidity premium coefficient
            beta: Duration preference exponent
        """
        self.alternatives = alternatives
        self.volatility = volatility
        self.alpha = alpha
        self.beta = beta

    def compute_required_apy(
        self,
        duration_years: float,
        base_rate: float = None
    ) -> float:
        """
        Compute required APY to induce locking at a given duration.
        
        Formula: Required_APY(d) = Base_Rate + Illiquidity_Premium(d) + Duration_Risk(d)
        
        Args:
            duration_years: Lock duration
            base_rate: Base rate (defaults to best alternative)
            
        Returns:
            Required APY (as fraction, e.g., 0.15 = 15%)
        """
        if base_rate is None:
            _, base_rate = self.alternatives.get_best_alternative()

        # Illiquidity premium: α × d^β
        illiquidity_premium = self.alpha * (duration_years ** self.beta)

        # Duration risk: Volatility × √d
        duration_risk = self.volatility * np.sqrt(duration_years) / 10.0  # Scaled for reasonable values

        required_apy = base_rate + illiquidity_premium + duration_risk
        return max(0.0, required_apy)

    def compute_time_preference_curve(
        self,
        durations: list[float],
        base_rate: float = None
    ) -> Dict[float, float]:
        """
        Compute time-preference indifference curve.
        
        Maps duration to required APY.
        
        Args:
            durations: List of durations in years
            base_rate: Base rate (defaults to best alternative)
            
        Returns:
            Dictionary mapping duration -> required APY
        """
        if base_rate is None:
            _, base_rate = self.alternatives.get_best_alternative()

        curve = {}
        for duration in durations:
            curve[duration] = self.compute_required_apy(duration, base_rate)

        return curve

    def compute_opportunity_cost_delta(
        self,
        vefil_apy: float,
        duration_years: float
    ) -> float:
        """
        Compute opportunity cost delta (veFIL yield vs best alternative).
        
        Positive = veFIL beats alternative
        Negative = alternative beats veFIL
        
        Args:
            vefil_apy: veFIL APY
            duration_years: Lock duration
            
        Returns:
            APY delta (veFIL - required)
        """
        required_apy = self.compute_required_apy(duration_years)
        return vefil_apy - required_apy

    def should_lock(
        self,
        vefil_apy: float,
        duration_years: float,
        slippage_cost: float = 0.0,
        lock_amount: float = 1.0
    ) -> bool:
        """
        Determine if a rational actor should lock.
        
        Args:
            vefil_apy: veFIL APY
            duration_years: Lock duration
            slippage_cost: Slippage cost (absolute)
            lock_amount: Lock amount (for slippage calculation)
            
        Returns:
            True if should lock, False otherwise
        """
        # Adjust APY for slippage
        if lock_amount > 0:
            slippage_apy_reduction = slippage_cost / (lock_amount * duration_years)
        else:
            slippage_apy_reduction = 0.0

        effective_vefil_apy = vefil_apy - slippage_apy_reduction

        required_apy = self.compute_required_apy(duration_years)
        return effective_vefil_apy >= required_apy
