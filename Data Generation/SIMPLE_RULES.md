# Ultra-Simple Data Generation Rules

## Core Concept
This generates stock price data based on **trivially simple** rules that any model should easily learn. The goal is to demonstrate sequence dependencies where events affect future outcomes.

## The Rules (No Math Complexity)

### 1. Event Timing
- Events happen every **5 days** starting from day 5
- Days with events: 5, 10, 15, 20, 25, 30, 35, 40...

### 2. Event Types (Cycle)
Events cycle in this exact order:
1. CEO_Change (impact: +2.0)
2. Product_Launch (impact: +3.0) 
3. Earnings_Report (impact: +1.0)
4. Competitor_Action (impact: -1.0)

### 3. Event Values
- Event value = `day % 10`
- This gives values 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

### 4. Basic Impact Calculation
- Impact = base_impact + (event_value * 0.1)
- Example: CEO_Change on day 15 → 2.0 + (5 * 0.1) = 2.5

### 5. Sequence Dependencies (The Key Learning)

#### CEO_Change Events:
- If last CEO change was < 20 days ago: **-1.0** modifier (instability)
- Otherwise: **+1.0** modifier (stable)

#### Earnings_Report Events:
- If event_value >= 5: "good earnings" → **+1.0** modifier, increment good_earnings_count
- If event_value < 5: "bad earnings" → **-1.0** modifier, decrement good_earnings_count

#### Product_Launch Events:
- If good_earnings_count >= 2: **+2.0** modifier (earnings boost)
- Otherwise: **0.0** modifier (no boost)

#### Competitor_Action Events:
- If last CEO change was < 30 days ago: **-2.0** modifier (vulnerable)
- Otherwise: **0.0** modifier (stable)

### 6. Final Calculation
- final_impact = base_impact + (event_value * 0.1) + sequence_modifier
- new_price = max(10.0, current_price + final_impact)

## Example Walkthrough

**Day 5:** CEO_Change, value=5
- Base impact: 2.0
- Event modifier: 5 * 0.1 = 0.5
- Sequence modifier: +1.0 (stable, no recent CEO)
- Final impact: 2.0 + 0.5 + 1.0 = 3.5
- Price: 100.0 + 3.5 = 103.5

**Day 10:** Product_Launch, value=0
- Base impact: 3.0
- Event modifier: 0 * 0.1 = 0.0
- Sequence modifier: 0.0 (good_earnings_count = 0)
- Final impact: 3.0 + 0.0 + 0.0 = 3.0
- Price: 103.5 + 3.0 = 106.5

**Day 15:** Earnings_Report, value=5
- Base impact: 1.0
- Event modifier: 5 * 0.1 = 0.5
- Sequence modifier: +1.0 (good earnings, value >= 5)
- Final impact: 1.0 + 0.5 + 1.0 = 2.5
- Price: 106.5 + 2.5 = 109.0
- good_earnings_count becomes 1

## Why This Is Perfect for Learning

1. **No trigonometry** - just basic arithmetic
2. **No division** - only addition, subtraction, multiplication
3. **Clear patterns** - events cycle predictably
4. **Simple sequence effects** - easy if/then rules
5. **Deterministic** - same input always gives same output

A model that learns this should be able to:
- Predict exact event timing (every 5 days)
- Predict exact event types (cycling pattern)
- Predict exact event values (day % 10)
- Understand sequence dependencies (CEO instability, earnings momentum, etc.) 