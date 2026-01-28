"""Crisis behavior modeling - How do lockers behave during extreme price drops?"""

from dataclasses import dataclass
from typing import List, Literal


@dataclass
class CrisisState:
    """Crisis state definition."""
    price_drop_fraction: float  # Price drop (e.g., 0.8 = 80% drop)
    time_into_lock_years: float  # How far into lock (e.g., 2.0 = 2 years)
    total_lock_duration_years: float  # Total lock duration
    initial_lock_value: float  # Initial FIL value locked
    current_lock_value: float  # Current FIL value


@dataclass
class CrisisBehavior:
    """Predicted behavior during crisis."""
    behavior_type: Literal["diamond_hands", "desperate_extraction", "abandon"]
    probability: float
    expected_actions: List[str]


class CrisisBehaviorModel:
    """Model behavioral responses to crisis scenarios."""

    def __init__(
        self,
        diamond_hands_threshold: float = 0.5,  # >50% remaining value
        abandonment_threshold: float = 0.2  # <20% remaining value
    ):
        """
        Initialize crisis behavior model.
        
        Args:
            diamond_hands_threshold: Value threshold for diamond hands behavior
            abandonment_threshold: Value threshold for abandonment behavior
        """
        self.diamond_hands_threshold = diamond_hands_threshold
        self.abandonment_threshold = abandonment_threshold

    def compute_crisis_state(
        self,
        initial_value: float,
        price_drop_fraction: float
    ) -> CrisisState:
        """
        Compute crisis state from price drop.
        
        Args:
            initial_value: Initial locked value
            price_drop_fraction: Price drop (0-1)
            
        Returns:
            Crisis state
        """
        current_value = initial_value * (1 - price_drop_fraction)

        return CrisisState(
            price_drop_fraction=price_drop_fraction,
            time_into_lock_years=0.0,  # To be set by caller
            total_lock_duration_years=0.0,  # To be set by caller
            initial_lock_value=initial_value,
            current_lock_value=current_value
        )

    def predict_behavior(
        self,
        crisis_state: CrisisState
    ) -> CrisisBehavior:
        """
        Predict behavior during crisis.
        
        Args:
            crisis_state: Crisis state
            
        Returns:
            Predicted behavior with probabilities
        """
        value_ratio = crisis_state.current_lock_value / crisis_state.initial_lock_value
        time_ratio = crisis_state.time_into_lock_years / crisis_state.total_lock_duration_years

        # Behavioral probabilities based on value and time
        if value_ratio >= self.diamond_hands_threshold:
            # High value retention -> diamond hands
            behavior_type = "diamond_hands"
            probability = 0.7
            expected_actions = [
                "Continue holding",
                "Wait for recovery",
                "Maintain voting participation"
            ]

        elif value_ratio <= self.abandonment_threshold:
            # Severe drawdown -> abandonment
            behavior_type = "abandon"
            probability = 0.6
            expected_actions = [
                "Stop participating",
                "Write off sunk cost",
                "Minimal engagement"
            ]

        else:
            # Moderate drawdown -> desperate extraction
            behavior_type = "desperate_extraction"
            probability = 0.5

            # If early in lock, more likely to extract
            if time_ratio < 0.4:
                probability = 0.7

            expected_actions = [
                "Vote for short-term extractive policies",
                "Seek early exit mechanisms",
                "Advocate for reward increases"
            ]

        # Time-based adjustments
        if time_ratio > 0.8:  # Near unlock
            # Less desperate if close to natural unlock
            if behavior_type == "desperate_extraction":
                probability *= 0.7

        return CrisisBehavior(
            behavior_type=behavior_type,
            probability=probability,
            expected_actions=expected_actions
        )

    def simulate_crisis_scenario(
        self,
        lock_amount: float,
        lock_duration_years: float,
        time_into_lock_years: float,
        price_drop_fraction: float
    ) -> dict:
        """
        Simulate a crisis scenario.
        
        Args:
            lock_amount: Amount locked (FIL)
            lock_duration_years: Total lock duration
            time_into_lock_years: Time elapsed into lock
            price_drop_fraction: Price drop (0-1)
            
        Returns:
            Crisis simulation results
        """
        initial_value = lock_amount
        crisis_state = CrisisState(
            price_drop_fraction=price_drop_fraction,
            time_into_lock_years=time_into_lock_years,
            total_lock_duration_years=lock_duration_years,
            initial_lock_value=initial_value,
            current_lock_value=initial_value * (1 - price_drop_fraction)
        )

        behavior = self.predict_behavior(crisis_state)

        # Compute utility shift
        value_loss = initial_value - crisis_state.current_lock_value
        utility_shift = -value_loss / initial_value  # Negative utility shift

        return {
            'crisis_state': crisis_state,
            'predicted_behavior': behavior,
            'value_loss': value_loss,
            'utility_shift': utility_shift,
            'sunk_cost_written_off': behavior.behavior_type == "abandon",
            'alignment_risk': behavior.behavior_type == "desperate_extraction"
        }
