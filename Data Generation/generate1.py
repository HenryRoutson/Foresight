import numpy as np
import pandas as pd
import random
from collections import deque
from typing import Dict, List, Any, Tuple, Union

# AI: Define types for better type safety
DayNumber = int
PriceValue = float
EventValue = float
ImpactValue = float

def generate_complex_stock_data(num_days: DayNumber = 250, initial_price: PriceValue = 100) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Generates synthetic stock price data influenced by a sequence of events,
    where event impact depends on both event type, associated values,
    and the history of previous events.
    """
    prices: List[PriceValue] = [initial_price]
    event_log: List[Dict[str, Any]] = [] # To store details of each event and its impact

    # Define event types and their base impacts
    event_types: Dict[str, Dict[str, Union[Tuple[int, int], float]]] = {
        "CEO_Change": {"base_impact_range": (-5, 10), "value_factor": 0.5}, # Proportion of board vote
        "Product_Launch": {"base_impact_range": (0, 7), "value_factor": 0.3}, # Anticipation score (0-1)
        "Earnings_Report": {"base_impact_range": (-3, 5), "value_factor": 0.7}, # % Surprise
        "Competitor_Action": {"base_impact_range": (-4, 2), "value_factor": 0.4}, # Severity (0-1)
    }

    # Track recent CEO changes to simulate a "honeymoon" or "instability" period
    last_ceo_change_day: DayNumber = -999999 # Use large negative number instead of -np.inf
    ceo_honeymoon_period: DayNumber = 30 # days

    # Track product launch success to influence subsequent launches
    successful_product_launches: deque[int] = deque(maxlen=3) # Keep track of recent success

    for day in range(1, num_days):
        current_price: PriceValue = prices[-1]
        daily_change: float = np.random.normal(0, 1) # Base daily fluctuation

        # Simulate events with varying probabilities
        event_details: Dict[str, Any] = {}

        if random.random() < 0.15: # ~15% chance of an event on any given day
            event_type: str = random.choice(list(event_types.keys()))
            event_config = event_types[event_type]
            event_value: EventValue = 0 # Placeholder for the specific value of the event
            impact: ImpactValue = 0 # Initialize impact

            if event_type == "CEO_Change":
                event_value = random.uniform(0.4, 1.0) # Proportion of board vote (40% to 100%)
                base_range = event_config["base_impact_range"]
                # AI: Fix numpy uniform call by ensuring proper type casting
                if isinstance(base_range, tuple):
                    impact = np.random.uniform(float(base_range[0]), float(base_range[1])) * event_value
                event_details = {"type": event_type, "value": event_value, "impact": impact}

                # Sequence dependency 1: CEO change impact can be amplified or reduced
                # if there was a recent CEO change (instability)
                if day - last_ceo_change_day < ceo_honeymoon_period:
                    impact *= (1 - (day - last_ceo_change_day) / ceo_honeymoon_period) * 1.5 # More volatile
                    event_details["sequence_factor"] = "recent_CEO_change_instability"
                else:
                    impact *= 1.2 # Slightly positive baseline for new CEO if stable
                    event_details["sequence_factor"] = "stable_CEO_change"
                last_ceo_change_day = day

            elif event_type == "Product_Launch":
                event_value = random.uniform(0, 1) # Anticipation score
                base_range = event_config["base_impact_range"]
                # AI: Fix numpy uniform call by ensuring proper type casting
                if isinstance(base_range, tuple):
                    impact = np.random.uniform(float(base_range[0]), float(base_range[1])) * (1 + event_value)
                event_details = {"type": event_type, "value": event_value, "impact": impact}

                # Sequence dependency 2: Prior product launch success influences current launch impact
                if successful_product_launches and np.mean(list(successful_product_launches)) > 0.7:
                    impact *= 1.5 # Boost if recent launches were successful
                    event_details["sequence_factor"] = "recent_product_success_boost"
                elif successful_product_launches and np.mean(list(successful_product_launches)) < 0.3:
                    impact *= 0.7 # Penalty if recent launches were poor
                    event_details["sequence_factor"] = "recent_product_failure_penalty"

                # Store the success of this product launch (for future sequence dependency)
                successful_product_launches.append(1 if impact > 0 else 0)

            elif event_type == "Earnings_Report":
                event_value = random.uniform(-0.2, 0.2) # % Surprise in earnings
                base_range = event_config["base_impact_range"]
                # AI: Fix numpy uniform call by ensuring proper type casting
                if isinstance(base_range, tuple):
                    impact = np.random.uniform(float(base_range[0]), float(base_range[1])) * (1 + event_value * 5)
                event_details = {"type": event_type, "value": event_value, "impact": impact}

                # Sequence dependency 3: Impact can be amplified if previous earnings were also a big surprise (momentum)
                if event_log:
                    prev_earnings = [e for e in event_log if e.get("type") == "Earnings_Report" and e.get("day") == day - 1]
                    if prev_earnings and abs(prev_earnings[0].get("value", 0)) > 0.15 and abs(event_value) > 0.15:
                        prev_value = prev_earnings[0].get("value", 0)
                        impact *= 1.5 if np.sign(prev_value) == np.sign(event_value) else 0.5 # Momentum or reversal
                        event_details["sequence_factor"] = "earnings_momentum_reversal"

            elif event_type == "Competitor_Action":
                event_value = random.uniform(0, 1) # Severity of competitor action
                base_range = event_config["base_impact_range"]
                # AI: Fix numpy uniform call by ensuring proper type casting
                if isinstance(base_range, tuple):
                    impact = np.random.uniform(float(base_range[0]), float(base_range[1])) * (1 + event_value * 2)
                event_details = {"type": event_type, "value": event_value, "impact": impact}

                # Sequence dependency 4: Response to competitor action might be stronger if the company is in a vulnerable state
                # (e.g., recent CEO change or poor earnings)
                recent_earnings_check = any(e.get("type") == "Earnings_Report" and e.get("value", 0) < -0.1 for e in event_log[-30:])
                if day - last_ceo_change_day < 60 or (event_log and recent_earnings_check):
                    impact *= 1.3 # More sensitive to competitor moves
                    event_details["sequence_factor"] = "vulnerable_state_competitor"

            daily_change += impact
            event_details["day"] = day
            event_log.append(event_details)

        new_price: PriceValue = max(10.0, current_price + daily_change) # Ensure price doesn't go below 10
        prices.append(new_price)

    df: pd.DataFrame = pd.DataFrame({"Price": prices})
    df.index = pd.RangeIndex(start=0, stop=len(prices), name="Day")
    return df, event_log

# Generate the data
stock_df, events_log = generate_complex_stock_data(num_days=500)

print("Generated Stock Prices (first 10 days):\n", stock_df.head(10))
print("\nGenerated Event Log (first 10 events):\n", events_log[:10])
