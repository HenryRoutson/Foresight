from typing import Dict, List, Any, Tuple

# AI: Simple types for clarity
DayNumber = int
PriceValue = float
EventValue = float
ImpactValue = float

def generate_simple_stock_data(num_days: DayNumber = 250, initial_price: PriceValue = 100) -> Tuple[List[PriceValue], List[Dict[str, Any]]]:
    """
    ULTRA-SIMPLE deterministic stock data for easy model learning.
    
    SIMPLE RULES:
    ============
    1. Events happen every 5 days starting from day 5
    2. Event types cycle: CEO -> Product -> Earnings -> Competitor
    3. Event values are just: day % 10 (gives 0-9)
    4. Price changes are: base_impact + (event_value * 0.1)
    5. Sequence effects: simple +1 or -1 modifiers
    
    NO TRIGONOMETRY, NO DIVISION, NO COMPLEX MATH
    """
    prices: List[PriceValue] = [initial_price]
    event_log: List[Dict[str, Any]] = []
    
    # AI: Ultra-simple event configuration
    event_types: List[str] = ["CEO_Change", "Product_Launch", "Earnings_Report", "Competitor_Action"]
    base_impacts: List[float] = [2.0, 3.0, 1.0, -1.0]  # Fixed impacts per event type
    
    # AI: Simple sequence tracking
    last_ceo_day: DayNumber = -100
    good_earnings_count: int = 0
    
    for day in range(1, num_days):
        current_price: PriceValue = prices[-1]
        daily_change: float = 0.0
        
        # AI: Events every 5 days, starting day 5
        if day >= 5 and (day - 5) % 5 == 0:
            # AI: Simple event type cycling
            event_index: int = ((day - 5) // 5) % len(event_types)
            event_type: str = event_types[event_index]
            base_impact: float = base_impacts[event_index]
            
            # AI: Simple event value: just day modulo 10
            event_value: EventValue = float(day % 10)
            
            # AI: Simple impact calculation
            impact: ImpactValue = base_impact + (event_value * 0.1)
            
            # AI: Simple sequence effects
            sequence_modifier: float = 0.0
            sequence_factor: str = "none"
            
            if event_type == "CEO_Change":
                if day - last_ceo_day < 20:  # Recent CEO change
                    sequence_modifier = -1.0
                    sequence_factor = "recent_ceo_instability"
                else:
                    sequence_modifier = 1.0
                    sequence_factor = "stable_ceo_change"
                last_ceo_day = day
                
            elif event_type == "Earnings_Report":
                if event_value >= 5:  # Good earnings
                    good_earnings_count += 1
                    sequence_modifier = 1.0
                    sequence_factor = "good_earnings"
                else:  # Bad earnings
                    good_earnings_count = max(0, good_earnings_count - 1)
                    sequence_modifier = -1.0
                    sequence_factor = "bad_earnings"
                    
            elif event_type == "Product_Launch":
                if good_earnings_count >= 2:  # Recent good earnings help
                    sequence_modifier = 2.0
                    sequence_factor = "earnings_boost"
                else:
                    sequence_modifier = 0.0
                    sequence_factor = "no_boost"
                    
            elif event_type == "Competitor_Action":
                if day - last_ceo_day < 30:  # Vulnerable during CEO transition
                    sequence_modifier = -2.0
                    sequence_factor = "vulnerable_state"
                else:
                    sequence_modifier = 0.0
                    sequence_factor = "stable_state"
            
            final_impact: float = impact + sequence_modifier
            daily_change = final_impact
            
            # AI: Log the event
            event_details: Dict[str, Any] = {
                "day": day,
                "type": event_type,
                "value": event_value,
                "impact": impact,
                "sequence_modifier": sequence_modifier,
                "sequence_factor": sequence_factor,
                "final_impact": final_impact
            }
            event_log.append(event_details)
        
        # AI: Update price with minimum floor
        new_price: PriceValue = max(10.0, current_price + daily_change)
        prices.append(new_price)
    
    return prices, event_log

def main() -> None:
    """Generate and display simple deterministic data."""
    prices, events = generate_simple_stock_data(num_days=100)
    
    print("SIMPLE Stock Prices (first 20 days):")
    for i, price in enumerate(prices[:20]):
        print(f"Day {i}: ${price:.2f}")
    
    print("\nSIMPLE Event Log (first 10 events):")
    for i, event in enumerate(events[:10]):
        print(f"Event {i+1}: {event}")
    
    print(f"\nTOTAL EVENTS: {len(events)}")
    print(f"FINAL PRICE: ${prices[-1]:.2f}")

if __name__ == "__main__":
    main()
