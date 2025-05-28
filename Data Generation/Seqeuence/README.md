# Complex Sequence-Dependent Stock Data Generation

This project demonstrates complex data generation that is difficult to visualize with simple tools and cannot be easily modeled by simple sequence models without considering the data values within each sequence entry.

## Overview

The system generates stock price data influenced by discrete events (CEO hiring, board meetings, scandals, etc.) where:

1. **Event sequences matter**: The next event type depends on the history of previous events
2. **Event data values matter**: Each event contains complex data (board vote percentage, media sentiment, market volatility, etc.) that affects stock prices
3. **Non-linear relationships**: Multiple factors interact in complex ways to determine price changes

## Key Features

### Complex Event Generation
- 8 different event types with realistic transition probabilities
- Event-specific data values that correlate with stock impact
- Historical dependency (recent events influence future event probabilities)

### Transformer Model
- Predicts next event type based on sequence history
- **Ignores data values** - only uses event type sequences
- Demonstrates the limitation of sequence-only modeling

### Why This Is Hard to Model

1. **Simple visualization fails**: Stock prices appear random without event context
2. **Sequence dependency**: Event transitions depend on complex historical patterns
3. **Data values crucial**: Board votes, sentiment, volatility all affect outcomes
4. **Non-linear interactions**: Multiple factors combine in complex ways

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete demonstration
python sequence.py
```

## Output

The script will:
1. Generate complex stock event sequences
2. Create visualizations showing why simple approaches fail
3. Train a transformer model to predict next event types
4. Display model predictions and training progress
5. Save plots: `complexity_demonstration.png` and `training_loss.png`

## Architecture

- **StockDataGenerator**: Creates realistic event sequences with complex interdependencies
- **EventTransformer**: PyTorch transformer model for sequence prediction
- **SequenceDataset**: Handles data preprocessing for model training

The transformer learns patterns in event type sequences but cannot access the rich data values that actually drive stock price changes, demonstrating the limitation of sequence-only approaches. 