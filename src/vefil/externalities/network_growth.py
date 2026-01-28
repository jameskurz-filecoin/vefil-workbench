"""Network growth dependency - Fee-based yield requires network growth."""

from dataclasses import dataclass


@dataclass
class NetworkGrowthState:
    """Network growth state."""
    transaction_volume: float  # Daily transaction volume
    fee_per_transaction: float
    annual_fee_revenue: float
    growth_rate: float  # Annual growth rate
    years_projected: int


class NetworkGrowthModel:
    """Model network growth dependency for fee-based yield."""

    def __init__(
        self,
        base_transaction_volume: float = 1000000.0,
        fee_per_transaction: float = 0.001,
        growth_rate: float = 0.15  # 15% annual growth
    ):
        """
        Initialize network growth model.
        
        Args:
            base_transaction_volume: Base daily transaction volume
            fee_per_transaction: FIL per transaction
            growth_rate: Annual growth rate
        """
        self.base_transaction_volume = base_transaction_volume
        self.fee_per_transaction = fee_per_transaction
        self.growth_rate = growth_rate

    def compute_fee_revenue(
        self,
        transaction_volume: float,
        dt_days: float = 365.25
    ) -> float:
        """
        Compute fee revenue for a period.
        
        Args:
            transaction_volume: Daily transaction volume
            dt_days: Period length in days
            
        Returns:
            Total fee revenue (FIL)
        """
        return transaction_volume * self.fee_per_transaction * dt_days

    def project_fee_revenue(
        self,
        initial_volume: float = None,
        growth_rate: float = None,
        years: int = 5
    ) -> NetworkGrowthState:
        """
        Project fee revenue over time with growth.
        
        Args:
            initial_volume: Initial daily transaction volume
            growth_rate: Annual growth rate
            years: Years to project
            
        Returns:
            Final network growth state
        """
        if initial_volume is None:
            initial_volume = self.base_transaction_volume
        if growth_rate is None:
            growth_rate = self.growth_rate

        # Project volume with compound growth
        final_volume = initial_volume * ((1 + growth_rate) ** years)

        # Annual fee revenue at final state
        annual_fee_revenue = self.compute_fee_revenue(final_volume, 365.25)

        return NetworkGrowthState(
            transaction_volume=final_volume,
            fee_per_transaction=self.fee_per_transaction,
            annual_fee_revenue=annual_fee_revenue,
            growth_rate=growth_rate,
            years_projected=years
        )

    def analyze_fee_yield_sustainability(
        self,
        required_annual_emission: float,
        initial_volume: float = None,
        growth_rate: float = None,
        years: int = 5
    ) -> dict:
        """
        Analyze if fee-based yield can sustain required emissions.
        
        Args:
            required_annual_emission: Required annual emission from fees
            initial_volume: Initial daily transaction volume
            growth_rate: Annual growth rate
            years: Years to analyze
            
        Returns:
            Sustainability analysis
        """
        if initial_volume is None:
            initial_volume = self.base_transaction_volume
        if growth_rate is None:
            growth_rate = self.growth_rate

        # Current fee revenue
        current_revenue = self.compute_fee_revenue(initial_volume, 365.25)

        # Projected revenue
        projected_state = self.project_fee_revenue(initial_volume, growth_rate, years)

        # Sustainability checks
        current_sustainable = current_revenue >= required_annual_emission
        projected_sustainable = projected_state.annual_fee_revenue >= required_annual_emission

        # Growth required to become sustainable
        if current_revenue < required_annual_emission:
            # Solve: required = initial * (1 + r)^years * fee_rate * 365.25
            # (1 + r)^years = required / (initial * fee_rate * 365.25)
            denominator = initial_volume * self.fee_per_transaction * 365.25
            if denominator > 0 and years > 0:
                growth_ratio = required_annual_emission / denominator
                if growth_ratio > 0:
                    required_growth_rate = (growth_ratio ** (1.0 / years)) - 1.0
                else:
                    required_growth_rate = float('inf')
            else:
                required_growth_rate = float('inf')
        else:
            required_growth_rate = 0.0

        # Scenarios
        scenarios = {
            'network_grows': self.project_fee_revenue(initial_volume, 0.20, years),  # 20% growth
            'network_flat': self.project_fee_revenue(initial_volume, 0.0, years),  # No growth
            'network_shrinks': self.project_fee_revenue(initial_volume, -0.10, years)  # -10% decline
        }

        scenario_sustainable = {
            name: state.annual_fee_revenue >= required_annual_emission
            for name, state in scenarios.items()
        }

        return {
            'required_annual_emission': required_annual_emission,
            'current_revenue': current_revenue,
            'current_sustainable': current_sustainable,
            'current_shortfall': max(0.0, required_annual_emission - current_revenue),
            'projected_revenue': projected_state.annual_fee_revenue,
            'projected_sustainable': projected_sustainable,
            'required_growth_rate': required_growth_rate,
            'scenarios': scenarios,
            'scenario_sustainable': scenario_sustainable,
            'death_spiral_risk': not scenario_sustainable['network_shrinks'] and growth_rate < 0
        }
