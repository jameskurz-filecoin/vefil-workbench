"""Module A: Supply & Accounting Engine - Deterministic tracking of all FIL flows."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SystemState:
    """System state at a point in time.

    Supply Ledger Semantics (Max-Supply Ledger):
    - total_supply: The max supply cap (constant, e.g., 2B FIL)
    - reserve: Mining reserve (300M FIL bucket that funds veFIL rewards)
    - other_allocations: Other allocations (SAFT, team, foundation, etc.) - not used for veFIL
    - burned_cumulative: FIL permanently destroyed (explicit bucket)
    - circulating: Liquid FIL available for transfer/lock
    - locked_vefil: FIL locked in the veFIL program
    - lending_pool: FIL deposited in lending protocols
    - sp_collateral: FIL pledged as SP collateral

    Conservation Identity:
    total_supply = circulating + locked_vefil + lending_pool + sp_collateral + reserve + other_allocations + burned_cumulative

    Outstanding Supply (derived):
    outstanding = total_supply - reserve - other_allocations - burned_cumulative
    """
    t: float  # Time in days
    total_supply: float
    circulating: float
    locked_vefil: float
    reserve: float
    lending_pool: float
    sp_collateral: float
    other_allocations: float = 0.0  # Other allocations (SAFT, team, foundation, etc.)
    burned_cumulative: float = 0.0  # FIL permanently destroyed

    @property
    def outstanding_supply(self) -> float:
        """Compute outstanding supply (minted and not burned)."""
        return self.total_supply - self.reserve - self.other_allocations - self.burned_cumulative

    def validate_conservation(self, tolerance: float = 1e-6) -> tuple[bool, Optional[str]]:
        """
        Validate conservation law including burns and other allocations:
        Total = Circulating + Locked + Reserve + Other + Lending + Collateral + Burned

        Returns:
            (is_valid, error_message)
        """
        computed_total = (
            self.circulating +
            self.locked_vefil +
            self.reserve +
            self.other_allocations +
            self.lending_pool +
            self.sp_collateral +
            self.burned_cumulative
        )

        diff = abs(self.total_supply - computed_total)
        # Allow tiny floating-point drift relative to total supply
        scaled_tolerance = max(tolerance, self.total_supply * 1e-9)
        if diff > scaled_tolerance:
            return False, (
                f"Conservation violation at t={self.t}: "
                f"Total={self.total_supply:.2f}, "
                f"Sum={computed_total:.2f}, "
                f"Diff={diff:.2f}, "
                f"(circ={self.circulating:.2f}, locked={self.locked_vefil:.2f}, "
                f"reserve={self.reserve:.2f}, other={self.other_allocations:.2f}, "
                f"lending={self.lending_pool:.2f}, collateral={self.sp_collateral:.2f}, "
                f"burned={self.burned_cumulative:.2f})"
            )
        return True, None

    def validate_non_negative(self, tolerance: float = 1e-6) -> tuple[bool, Optional[str]]:
        """Validate all buckets are non-negative."""
        buckets = [
            ('circulating', self.circulating),
            ('locked_vefil', self.locked_vefil),
            ('reserve', self.reserve),
            ('other_allocations', self.other_allocations),
            ('lending_pool', self.lending_pool),
            ('sp_collateral', self.sp_collateral),
            ('burned_cumulative', self.burned_cumulative),
        ]
        for name, value in buckets:
            if value < -tolerance:
                return False, f"Negative bucket at t={self.t}: {name}={value:.2f}"
        return True, None


@dataclass
class Flows:
    """Flow variables for a timestep."""
    mints: float = 0.0  # New FIL minted
    burns: float = 0.0  # FIL burned
    emission: float = 0.0  # Emission from reserve
    emission_to_circulating: float = 0.0  # Portion of emission sold into circulation
    net_locks: float = 0.0  # New FIL locked this period (gross new locks)
    unlocks: float = 0.0  # FIL unlocked from expired positions
    net_lending_withdrawals: float = 0.0  # Net withdrawals from lending (can be negative for deposits)
    lending_cannibalized: float = 0.0  # FIL moved from lending to veFIL
    sp_collateral_change: float = 0.0  # Change in SP collateral
    reward_relocks: float = 0.0  # Rewards that go directly from reserve to locked (bypass circulating)


class AccountingEngine:
    """Supply and accounting engine with conservation validation."""

    def __init__(self, initial_state: SystemState):
        """
        Initialize accounting engine.

        Args:
            initial_state: Initial system state
        """
        self.initial_state = initial_state
        self.current_state = initial_state
        self.history: list[SystemState] = [initial_state]
        self._recent_flows: Optional[Flows] = None  # Track actual flows for metrics

    def step(self, flows: Flows, dt: float) -> SystemState:
        """
        Advance state by dt, applying all flows.

        Args:
            flows: Flow variables for this timestep
            dt: Timestep in days

        Returns:
            New system state

        Raises:
            ValueError: If conservation is violated
        """
        # Store flows for accurate metrics calculation
        self._recent_flows = flows

        # Update circulating supply
        # Note: net_locks includes locks from multiple sources:
        # - From circulating (market purchases, idle holdings)
        # - From lending pool (cannibalized)
        # - From reserve via rewards (reward_relocks)
        # We add back cannibalized and reward_relocks since they don't come from circulating.
        emission_to_circulating = flows.emission_to_circulating
        new_circulating = (
            self.current_state.circulating +
            flows.mints -
            flows.burns +
            emission_to_circulating -
            flows.net_locks +
            flows.lending_cannibalized +  # Add back: comes from lending, not circulating
            flows.reward_relocks +  # Add back: comes from reserve via emission, not circulating
            flows.unlocks +
            flows.net_lending_withdrawals
        )

        # Update reserve (emissions deplete reserve)
        new_reserve = self.current_state.reserve - flows.emission

        # Update locked veFIL
        new_locked = (
            self.current_state.locked_vefil +
            flows.net_locks -
            flows.unlocks
        )

        # Update lending pool (withdrawals and cannibalization)
        # Note: net_lending_withdrawals is positive when FIL flows OUT of lending to circulation
        # so we subtract it (lending pool decreases)
        new_lending = (
            self.current_state.lending_pool -
            flows.net_lending_withdrawals -
            flows.lending_cannibalized
        )

        # Update SP collateral
        new_sp_collateral = (
            self.current_state.sp_collateral +
            flows.sp_collateral_change
        )

        # Handle negative values by returning overflow to circulating (conservation)
        lending_overflow = 0.0
        collateral_overflow = 0.0
        if new_lending < 0:
            lending_overflow = -new_lending
            new_lending = 0.0
        if new_sp_collateral < 0:
            collateral_overflow = -new_sp_collateral
            new_sp_collateral = 0.0

        # Add any overflow back to circulating to maintain conservation
        new_circulating += lending_overflow + collateral_overflow

        # Update burned cumulative (burns go to explicit bucket)
        new_burned = self.current_state.burned_cumulative + flows.burns

        # Create new state
        # Note: With burns going to burned_cumulative, total_supply stays constant
        # (max supply ledger). Burns don't reduce total_supply, they move to burned bucket.
        # other_allocations is carried forward unchanged (not used for veFIL rewards).
        new_state = SystemState(
            t=self.current_state.t + dt,
            total_supply=self.current_state.total_supply,  # Max supply is constant
            circulating=new_circulating,
            locked_vefil=new_locked,
            reserve=new_reserve,
            lending_pool=new_lending,
            sp_collateral=new_sp_collateral,
            other_allocations=self.current_state.other_allocations,  # Carry forward unchanged
            burned_cumulative=new_burned
        )

        # Validate conservation
        is_valid, error_msg = new_state.validate_conservation()
        if not is_valid:
            raise ValueError(error_msg)

        # Update state
        self.current_state = new_state
        self.history.append(new_state)

        return new_state

    def compute_inflation_metrics(self, dt: float = 30.0) -> dict:
        """
        Compute inflation metrics for the current timestep.

        Args:
            dt: Timestep in days

        Returns:
            Dictionary of inflation metrics
        """
        if len(self.history) < 2:
            return {
                'net_inflation_rate': 0.0,
                'gross_emission_rate': 0.0,
                'effective_inflation': 0.0
            }

        prev_state = self.history[-2]
        curr_state = self.current_state

        # Net inflation rate (annualized)
        delta_circulating = curr_state.circulating - prev_state.circulating
        if prev_state.circulating > 0:
            net_inflation_rate = (delta_circulating / prev_state.circulating) * (365.25 / dt)
        else:
            net_inflation_rate = 0.0

        # Use stored flows if available, otherwise estimate
        flows = self._recent_flows if self._recent_flows else self._estimate_flows()

        # Gross emission rate (annualized)
        if prev_state.reserve > 0:
            gross_emission_rate = (flows.emission / prev_state.reserve) * (365.25 / dt)
        else:
            gross_emission_rate = 0.0

        # Effective inflation (emission minus net lock change, annualized)
        # Net lock change = new locks - unlocks; unlocks return to circulation
        # This correctly accounts for unlocks adding back to circulating supply
        if prev_state.circulating > 0:
            net_lock_change = flows.net_locks - flows.unlocks
            emission_to_circulating = flows.emission_to_circulating
            effective_inflation = (
                (emission_to_circulating - net_lock_change) / prev_state.circulating
            ) * (365.25 / dt)
        else:
            effective_inflation = 0.0

        return {
            'net_inflation_rate': net_inflation_rate,
            'gross_emission_rate': gross_emission_rate,
            'effective_inflation': effective_inflation
        }

    def _estimate_flows(self) -> Flows:
        """Estimate flows from recent history (fallback for metrics calculation)."""
        if len(self.history) < 2:
            return Flows()

        prev_state = self.history[-2]
        curr_state = self.current_state

        # Estimate emission from reserve change
        emission = prev_state.reserve - curr_state.reserve

        # Estimate net lock change from locked supply change
        net_lock_change = curr_state.locked_vefil - prev_state.locked_vefil

        # Cannot separate net_locks from unlocks without actual flow data
        # Return as net_locks with zero unlocks (will underestimate effective inflation
        # if there were unlocks, but this is a fallback path)
        if net_lock_change >= 0:
            net_locks = net_lock_change
            unlocks = 0.0
        else:
            net_locks = 0.0
            unlocks = -net_lock_change

        return Flows(
            emission=emission,
            emission_to_circulating=emission,
            net_locks=net_locks,
            unlocks=unlocks
        )

    def get_reserve_runway_years(self) -> float:
        """
        Estimate reserve runway in years at current emission rate.

        Returns:
            Years until reserve exhaustion, or float('inf') if no emission
        """
        if len(self.history) < 2:
            return float('inf')

        flows = self._recent_flows if self._recent_flows else self._estimate_flows()
        if flows.emission <= 0:
            return float('inf')

        # Annualize emission rate
        time_delta = self.current_state.t - self.history[-2].t
        if time_delta <= 0:
            return float('inf')

        annual_emission = flows.emission * (365.25 / time_delta)

        if annual_emission <= 0:
            return float('inf')

        runway_years = self.current_state.reserve / annual_emission
        return max(0.0, runway_years)
