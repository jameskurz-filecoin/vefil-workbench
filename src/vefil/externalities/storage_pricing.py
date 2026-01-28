"""Storage pricing feedback loop - How does locking affect storage costs?"""

from dataclasses import dataclass


@dataclass
class StoragePricingState:
    """Storage pricing state."""
    circulating_supply: float
    fil_price: float
    network_qap: float  # Quality-adjusted power (storage capacity)
    initial_pledge_per_tib: float
    storage_price_per_gb_month: float
    s3_price_per_gb_month: float = 0.023  # AWS S3 baseline


class StoragePricingModel:
    """Model storage pricing feedback from supply shocks."""

    def __init__(
        self,
        s3_price_per_gb_month: float = 0.023,
        pledge_elasticity: float = 0.5  # How much pledge changes with supply
    ):
        """
        Initialize storage pricing model.
        
        Args:
            s3_price_per_gb_month: S3 baseline price for comparison
            pledge_elasticity: Elasticity of pledge to supply changes
        """
        self.s3_price_per_gb_month = s3_price_per_gb_month
        self.pledge_elasticity = pledge_elasticity

    def compute_initial_pledge(
        self,
        network_qap: float,
        circulating_supply: float,
        fil_price: float,
        base_pledge_per_tib: float = 0.1  # Base pledge in FIL per TiB
    ) -> float:
        """
        Compute initial pledge per TiB.
        
        Simplified model: pledge scales with network size and inversely with supply/price.
        Real formula is more complex.
        
        Args:
            network_qap: Network quality-adjusted power (TiB)
            circulating_supply: Circulating FIL supply
            fil_price: FIL price in USD
            base_pledge_per_tib: Base pledge per TiB
            
        Returns:
            Initial pledge per TiB (in FIL)
        """
        # Simplified: pledge increases with network size, decreases with supply
        supply_factor = max(0.1, circulating_supply / 600_000_000)  # Normalize to ~600M baseline
        network_factor = network_qap / 10_000  # Normalize to ~10PiB baseline

        # Pledge per TiB
        pledge_per_tib = base_pledge_per_tib * (network_factor ** 0.5) / (supply_factor ** self.pledge_elasticity)

        return max(0.01, pledge_per_tib)

    def compute_storage_price(
        self,
        initial_pledge_per_tib: float,
        fil_price: float,
        sp_operational_cost_multiplier: float = 1.5
    ) -> float:
        """
        Compute storage price from pledge cost.
        
        Args:
            initial_pledge_per_tib: Initial pledge per TiB (FIL)
            fil_price: FIL price (USD)
            sp_operational_cost_multiplier: Multiplier for SP operational costs
        
        Returns:
            Storage price per GB per month (USD)
        """
        # Convert TiB to GB
        tib_to_gb = 1024  # 1 TiB = 1024 GB

        # Pledge cost per GB
        pledge_cost_per_gb = (initial_pledge_per_tib * fil_price) / (tib_to_gb * 12)  # Annualize then monthly

        # Add operational costs (storage, bandwidth, etc.)
        operational_cost = pledge_cost_per_gb * sp_operational_cost_multiplier

        # Total storage price (with margin)
        storage_price = operational_cost * 1.2  # 20% margin

        return storage_price

    def compute_competitiveness(
        self,
        fil_storage_price: float,
        s3_price: float = None
    ) -> dict:
        """
        Compare Filecoin storage price to S3.
        
        Args:
            fil_storage_price: Filecoin storage price per GB/month
            s3_price: S3 price (defaults to class default)
            
        Returns:
            Competitiveness analysis
        """
        if s3_price is None:
            s3_price = self.s3_price_per_gb_month

        price_ratio = fil_storage_price / s3_price if s3_price > 0 else float('inf')
        is_competitive = fil_storage_price <= s3_price
        premium = (price_ratio - 1.0) * 100 if price_ratio > 1.0 else 0.0

        return {
            'fil_price': fil_storage_price,
            's3_price': s3_price,
            'price_ratio': price_ratio,
            'is_competitive': is_competitive,
            'premium_percent': premium,
            'competitive_advantage': s3_price - fil_storage_price if is_competitive else 0.0
        }

    def analyze_supply_shock_impact(
        self,
        initial_circulating: float,
        locked_amount: float,
        network_qap: float,
        fil_price: float
    ) -> dict:
        """
        Analyze impact of supply shock (locking) on storage pricing.
        
        Args:
            initial_circulating: Initial circulating supply
            locked_amount: Amount being locked
            network_qap: Network QAP (TiB)
            fil_price: FIL price
            
        Returns:
            Impact analysis
        """
        new_circulating = initial_circulating - locked_amount

        # Initial pledge before and after
        pledge_before = self.compute_initial_pledge(network_qap, initial_circulating, fil_price)
        pledge_after = self.compute_initial_pledge(network_qap, new_circulating, fil_price)

        # Storage prices
        price_before = self.compute_storage_price(pledge_before, fil_price)
        price_after = self.compute_storage_price(pledge_after, fil_price)

        # Competitiveness
        comp_before = self.compute_competitiveness(price_before)
        comp_after = self.compute_competitiveness(price_after)

        return {
            'circulating_change': -locked_amount,
            'circulating_change_percent': (-locked_amount / initial_circulating) * 100,
            'pledge_change': pledge_after - pledge_before,
            'pledge_change_percent': ((pledge_after - pledge_before) / pledge_before) * 100,
            'storage_price_change': price_after - price_before,
            'storage_price_change_percent': ((price_after - price_before) / price_before) * 100,
            'competitiveness_before': comp_before,
            'competitiveness_after': comp_after,
            'becomes_uncompetitive': comp_before['is_competitive'] and not comp_after['is_competitive']
        }
