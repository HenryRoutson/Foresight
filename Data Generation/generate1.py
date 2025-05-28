import numpy as np
import pandas as pd
import random
from collections import deque

def generate_complex_stock_data(num_days=250, initial_price=100):
    """
    Generates synthetic stock price data influenced by a sequence of events,
    where event impact depends on both event type, associated values,
    and the history of previous events.
    """
    prices = [initial_price]
    event_log = [] # To store details of each event and its impact

    # Define event types and their base impacts
    event_types = {
        "CEO_Change": {"base_impact_range": (-5, 10), "value_factor": 0.5}, # Proportion of board vote
        "Product_Launch": {"base_impact_range": (0, 7), "value_factor": 0.3}, # Anticipation score (0-1)
        "Earnings_Report": {"base_impact_range": (-3, 5), "value_factor": 0.7}, # % Surprise
        "Competitor_Action": {"base_impact_range": (-4, 2), "value_factor": 0.4}, # Severity (0-1)
    }

    # Track recent CEO changes to simulate a "honeymoon" or "instability" period
    last_ceo_change_day = -np.inf
    ceo_honeymoon_period = 30 # days

    # Track product launch success to influence subsequent launches
    successful_product_launches = deque(maxlen=3) # Keep track of recent success

    for day in range(1, num_days):
        current_price = prices[-1]
        daily_change = np.random.normal(0, 1) # Base daily fluctuation

        # Simulate events with varying probabilities
        event_happened = False
        event_details = {}

        if random.random() < 0.15: # ~15% chance of an event on any given day
            event_type = random.choice(list(event_types.keys()))
            event_config = event_types[event_type]
            event_value = 0 # Placeholder for the specific value of the event

            if event_type == "CEO_Change":
                event_value = random.uniform(0.4, 1.0) # Proportion of board vote (40% to 100%)
                impact = np.random.uniform(*event_config["base_impact_range"]) * event_value
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
                impact = np.random.uniform(*event_config["base_impact_range"]) * (1 + event_value)
                event_details = {"type": event_type, "value": event_value, "impact": impact}

                # Sequence dependency 2: Prior product launch success influences current launch impact
                if successful_product_launches and np.mean(successful_product_launches) > 0.7:
                    impact *= 1.5 # Boost if recent launches were successful
                    event_details["sequence_factor"] = "recent_product_success_boost"
                elif successful_product_launches and np.mean(successful_product_launches) < 0.3:
                    impact *= 0.7 # Penalty if recent launches were poor
                    event_details["sequence_factor"] = "recent_product_failure_penalty"

                # Store the success of this product launch (for future sequence dependency)
                successful_product_launches.append(1 if impact > 0 else 0)

            elif event_type == "Earnings_Report":
                event_value = random.uniform(-0.2, 0.2) # % Surprise in earnings
                impact = np.random.uniform(*event_config["base_impact_range"]) * (1 + event_value * 5)
                event_details = {"type": event_type, "value": event_value, "impact": impact}

                # Sequence dependency 3: Impact can be amplified if previous earnings were also a big surprise (momentum)
                if event_log:
                    prev_earnings = [e for e in event_log if e.get("type") == "Earnings_Report" and e.get("day") == day - 1]
                    if prev_earnings and abs(prev_earnings[0].get("value")) > 0.15 and abs(event_value) > 0.15:
                        impact *= 1.5 if np.sign(prev_earnings[0].get("value")) == np.sign(event_value) else 0.5 # Momentum or reversal
                        event_details["sequence_factor"] = "earnings_momentum_reversal"

            elif event_type == "Competitor_Action":
                event_value = random.uniform(0, 1) # Severity of competitor action
                impact = np.random.uniform(*event_config["base_impact_range"]) * (1 + event_value * 2)
                event_details = {"type": event_type, "value": event_value, "impact": impact}

                # Sequence dependency 4: Response to competitor action might be stronger if the company is in a vulnerable state
                # (e.g., recent CEO change or poor earnings)
                if day - last_ceo_change_day < 60 or (event_log and any(e.get("type") == "Earnings_Report" and e.get("value",0) < -0.1 for e in event_log[-30:])):
                    impact *= 1.3 # More sensitive to competitor moves
                    event_details["sequence_factor"] = "vulnerable_state_competitor"

            daily_change += impact
            event_details["day"] = day
            event_log.append(event_details)

        new_price = max(10, current_price + daily_change) # Ensure price doesn't go below 10
        prices.append(new_price)

    df = pd.DataFrame({"Price": prices})
    df.index.name = "Day"
    return df, event_log

# Generate the data
stock_df, events_log = generate_complex_stock_data(num_days=500)

print("Generated Stock Prices (first 10 days):\n", stock_df.head(10))
print("\nGenerated Event Log (first 10 events):\n", events_log[:10])
