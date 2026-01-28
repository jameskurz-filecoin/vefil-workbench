"""Module E: Lending Market Impact - Model capital flight and SP borrowing rate spikes."""

from dataclasses import dataclass


@dataclass
class LendingMarketState:
    """Lending market state."""
    pool_size: float  # Total FIL in lending pool
    sp_borrow_demand: float  # SP borrowing demand
    base_rate: float  # Base lending rate
    utilization: float  # Current utilization (borrows / pool)
    lending_rate: float  # Current lending rate
    utilization_elasticity: float  # Rate elasticity to utilization


class LendingMarketModel:
    """Model for lending market dynamics and veFIL cannibalization impact."""

    def __init__(
        self,
        base_rate: float = 0.08,
        utilization_elasticity: float = 2.0,
        sp_borrow_demand: float = 80000000.0
    ):
        """
        Initialize lending market model.
        
        Args:
            base_rate: Base lending rate (8% default)
            utilization_elasticity: Rate elasticity to utilization
            sp_borrow_demand: SP borrowing demand in FIL
        """
        self.base_rate = base_rate
        self.utilization_elasticity = utilization_elasticity
        self.sp_borrow_demand = sp_borrow_demand

    def compute_lending_rate(self, utilization: float) -> float:
        """
        Compute lending rate from utilization.
        
        Formula: Lending_Rate = Base_Rate × (1 + Utilization)^elasticity
        
        Args:
            utilization: Utilization ratio (0-1)
            
        Returns:
            Lending rate (as fraction, e.g., 0.12 = 12%)
        """
        if utilization < 0:
            utilization = 0.0
        if utilization > 1.0:
            utilization = 1.0

        lending_rate = self.base_rate * ((1.0 + utilization) ** self.utilization_elasticity)
        return lending_rate

    def compute_utilization(
        self,
        pool_size: float,
        borrows: float = None
    ) -> float:
        """
        Compute utilization ratio.
        
        Args:
            pool_size: Lending pool size
            borrows: Amount borrowed (defaults to SP demand)
            
        Returns:
            Utilization ratio (0-1)
        """
        if pool_size <= 0:
            return 1.0  # Fully utilized if no pool

        if borrows is None:
            borrows = self.sp_borrow_demand

        utilization = min(1.0, borrows / pool_size)
        return utilization

    def update_state(
        self,
        current_pool: float,
        cannibalized_to_vefil: float = 0.0,
        new_deposits: float = 0.0
    ) -> LendingMarketState:
        """
        Update lending market state after capital movements.
        
        Args:
            current_pool: Current lending pool size
            cannibalized_to_vefil: FIL moved from lending to veFIL
            new_deposits: New deposits to lending pool
            
        Returns:
            Updated lending market state
        """
        new_pool = current_pool - cannibalized_to_vefil + new_deposits
        new_pool = max(0.0, new_pool)  # Can't go negative

        utilization = self.compute_utilization(new_pool)
        lending_rate = self.compute_lending_rate(utilization)

        return LendingMarketState(
            pool_size=new_pool,
            sp_borrow_demand=self.sp_borrow_demand,
            base_rate=self.base_rate,
            utilization=utilization,
            lending_rate=lending_rate,
            utilization_elasticity=self.utilization_elasticity
        )

    def compute_sp_cost_increase(
        self,
        initial_state: LendingMarketState,
        final_state: LendingMarketState
    ) -> float:
        """
        Compute SP borrowing cost increase.
        
        Args:
            initial_state: Initial lending market state
            final_state: Final lending market state after cannibalization
            
        Returns:
            Rate increase (as fraction, e.g., 0.02 = 2 percentage points)
        """
        return final_state.lending_rate - initial_state.lending_rate

    def simulate_cannibalization_scenario(
        self,
        initial_pool: float,
        cannibalization_fraction: float
    ) -> dict:
        """
        Simulate a cannibalization scenario.
        
        Args:
            initial_pool: Initial lending pool size
            cannibalization_fraction: Fraction of pool to cannibalize (e.g., 0.25 = 25%)
            
        Returns:
            Dictionary with scenario results
        """
        initial_state = self.update_state(initial_pool, 0.0, 0.0)

        cannibalized_amount = initial_pool * cannibalization_fraction
        final_state = self.update_state(initial_pool, cannibalized_amount, 0.0)

        rate_increase = self.compute_sp_cost_increase(initial_state, final_state)
        rate_increase_percent = rate_increase * 100

        return {
            'initial_pool': initial_pool,
            'cannibalized_amount': cannibalized_amount,
            'final_pool': final_state.pool_size,
            'initial_rate': initial_state.lending_rate,
            'final_rate': final_state.lending_rate,
            'rate_increase': rate_increase,
            'rate_increase_percent': rate_increase_percent,
            'initial_utilization': initial_state.utilization,
            'final_utilization': final_state.utilization,
            'sp_cost_impact': rate_increase * self.sp_borrow_demand  # Annual cost in FIL
        }
