import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, List, Any, Tuple

# AI: Define types for better type safety
DayNumber = int
PriceValue = float
EventValue = float
ImpactValue = float

def generate_deterministic_stock_data(num_days: DayNumber = 250, initial_price: PriceValue = 100) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Generates DETERMINISTIC synthetic stock price data influenced by a sequence of events.
    
    DETERMINISTIC DESIGN EXPLANATION:
    ================================
    This function is designed to be completely deterministic - no randomness whatsoever.
    When a model has fully learned this generating function, it should be able to predict:
    
    1. EXACT event timing: Events occur on predictable days based on mathematical patterns
    2. EXACT event types: Event types follow a deterministic cycle
    3. EXACT event values: All event parameters are calculated using deterministic formulas
    4. EXACT price changes: All price movements are based on deterministic calculations
    5. EXACT sequence dependencies: All historical influences follow clear mathematical rules
    
    NEURAL NETWORK FRIENDLY DESIGN:
    ==============================
    This version avoids division operations that can be difficult for neural networks:
    - Uses addition/subtraction instead of division for scaling
    - Uses modulo operations sparingly and only with small constants
    - Uses multiplication and simple arithmetic progressions
    - Replaces complex ratios with linear transformations
    
    LEARNING VERIFICATION:
    =====================
    A model has "learned" this function when it can:
    - Predict the exact price on any future day
    - Predict exactly when events will occur
    - Predict the exact type and impact of each event
    - Understand how past events influence future events through the sequence dependencies
    
    If there's ANY unpredictability in the model's outputs, it hasn't fully learned the function.
    """
    prices: List[PriceValue] = [initial_price]
    event_log: List[Dict[str, Any]] = []

    # AI: Deterministic event configuration - no random ranges
    event_types: Dict[str, Dict[str, float]] = {
        "CEO_Change": {"base_impact": 2.5, "value_factor": 0.5},
        "Product_Launch": {"base_impact": 3.5, "value_factor": 0.3}, 
        "Earnings_Report": {"base_impact": 1.0, "value_factor": 0.7},
        "Competitor_Action": {"base_impact": -1.0, "value_factor": 0.4},
    }
    
    event_type_list : list[str] = list(event_types.keys())

    # AI: Track sequence dependencies deterministically
    last_ceo_change_day: DayNumber = -999999
    ceo_honeymoon_period: DayNumber = 30
    successful_product_launches: deque[int] = deque(maxlen=3)

    for day in range(1, num_days):
        current_price: PriceValue = prices[-1]
        
        # AI: Deterministic daily fluctuation - no division, use simple scaling
        daily_change: float = 0.5 * np.sin(day * 0.1) + 0.3 * np.cos(day * 0.05)
        
        # AI: Deterministic event occurrence - events happen every 7 days starting from day 5
        event_details: Dict[str, Any] = {}
        
        if (day - 5) % 7 == 0 and day >= 5:  # Events on days 5, 12, 19, 26, etc.
            # AI: Deterministic event type selection - cycle through event types (avoid division)
            # Instead of division, use modulo directly with list length
            event_type_index : int = ((day - 5) // 7) % len(event_type_list)
            event_type: str = event_type_list[event_type_index]
            event_config : dict[str, float] = event_types[event_type]
            
            # AI: Deterministic event value calculation based on day
            event_value: EventValue = 0
            impact: ImpactValue = 0

            if event_type == "CEO_Change":
                # AI: Board vote proportion - avoid division, use linear scaling
                # Instead of (sin + 1) / 2, use sin * 0.3 + 0.7 for range [0.4, 1.0]
                event_value : float = 0.4 + 0.6 * (np.sin(day * 0.2) * 0.5 + 0.5)
                impact : float = event_config["base_impact"] * event_value
                event_details : dict[str, Any] = {"type": event_type, "value": event_value, "impact": impact}

                # AI: Deterministic sequence dependency for CEO changes - avoid division
                if day - last_ceo_change_day < ceo_honeymoon_period:
                    # Instead of division, use subtraction and multiplication
                    days_since_change = day - last_ceo_change_day
                    instability_factor = 1.0 - (days_since_change * 0.033)  # 0.033 ≈ 1/30
                    impact *= instability_factor * 1.5
                    event_details["sequence_factor"] = "recent_CEO_change_instability"
                else:
                    impact *= 1.2
                    event_details["sequence_factor"] = "stable_CEO_change"
                last_ceo_change_day = day

            elif event_type == "Product_Launch":
                # AI: Anticipation score - avoid division, use direct scaling
                event_value = np.cos(day * 0.15) * 0.5 + 0.5  # Range [0, 1]
                impact = event_config["base_impact"] * (1 + event_value)
                event_details = {"type": event_type, "value": event_value, "impact": impact}

                # AI: Deterministic sequence dependency for product launches
                if successful_product_launches:
                    # AI: Avoid division by using sum and comparison instead of mean
                    success_sum = sum(successful_product_launches)
                    total_launches = len(successful_product_launches)
                    # Instead of mean > 0.7, use sum > 0.7 * total_launches
                    if success_sum > 0.7 * total_launches:
                        impact *= 1.5
                        event_details["sequence_factor"] = "recent_product_success_boost"
                    elif success_sum < 0.3 * total_launches:
                        impact *= 0.7
                        event_details["sequence_factor"] = "recent_product_failure_penalty"

                # AI: Deterministic success tracking
                successful_product_launches.append(1 if impact > 0 else 0)

            elif event_type == "Earnings_Report":
                # AI: Earnings surprise cycles between -0.2 and 0.2 - no division needed
                event_value = 0.2 * np.sin(day * 0.25)
                impact = event_config["base_impact"] * (1 + event_value * 5)
                event_details = {"type": event_type, "value": event_value, "impact": impact}

                # AI: Deterministic earnings momentum check
                if event_log:
                    prev_earnings = [e for e in event_log if e.get("type") == "Earnings_Report" and e.get("day") == day - 7]
                    if prev_earnings:
                        prev_value = prev_earnings[0].get("value", 0)
                        if abs(prev_value) > 0.15 and abs(event_value) > 0.15:
                            momentum_factor = 1.5 if np.sign(prev_value) == np.sign(event_value) else 0.5
                            impact *= momentum_factor
                            event_details["sequence_factor"] = "earnings_momentum_reversal"

            elif event_type == "Competitor_Action":
                # AI: Severity - avoid division, use direct scaling
                event_value = np.sin(day * 0.3) * 0.5 + 0.5  # Range [0, 1]
                impact = event_config["base_impact"] * (1 + event_value * 2)
                event_details = {"type": event_type, "value": event_value, "impact": impact}

                # AI: Deterministic vulnerability check
                recent_earnings_check = any(
                    e.get("type") == "Earnings_Report" and e.get("value", 0) < -0.1 
                    for e in event_log[-4:] if e.get("day", 0) > day - 30
                )
                if day - last_ceo_change_day < 60 or recent_earnings_check:
                    impact *= 1.3
                    event_details["sequence_factor"] = "vulnerable_state_competitor"

            daily_change += impact
            event_details["day"] = day
            event_log.append(event_details)

        # AI: Deterministic price calculation with floor - no division
        new_price: PriceValue = max(10.0, current_price + daily_change)
        prices.append(new_price)

    df: pd.DataFrame = pd.DataFrame({"Price": prices})
    df.index = pd.RangeIndex(start=0, stop=len(prices), name="Day")
    return df, event_log

# AI: Generate deterministic data - same output every time
stock_df, events_log = generate_deterministic_stock_data(num_days=500)

print("DETERMINISTIC Stock Prices (first 10 days):\n", stock_df.head(10))
print("\nDETERMINISTIC Event Log (first 10 events):\n", events_log[:10])
