import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import random
from dataclasses import dataclass
from enum import Enum

# AI: Define event types that affect stock prices
class EventType(Enum):
    CEO_HIRE = "ceo_hire"
    BOARD_MEETING = "board_meeting"
    EARNINGS_REPORT = "earnings_report"
    MERGER_ANNOUNCEMENT = "merger_announcement"
    REGULATORY_CHANGE = "regulatory_change"
    PRODUCT_LAUNCH = "product_launch"
    SCANDAL = "scandal"
    MARKET_CRASH = "market_crash"

@dataclass
class StockEvent:
    event_type: EventType
    timestamp: int
    # AI: Complex data values that correlate with stock impact
    board_vote_percentage: float  # 0-100
    media_sentiment: float  # -1 to 1
    market_volatility: float  # 0-1
    company_size_factor: float  # 0.1-10
    previous_event_impact: float  # -1 to 1

# AI: Type aliases for better code clarity
EventSequence = List[StockEvent]
PriceSequence = List[float]

class StockDataGenerator:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.base_price = 100.0
        
    def generate_event_sequence(self, length: int) -> EventSequence:
        """Generate a sequence of stock events with complex interdependencies"""
        events: EventSequence = []
        current_price = self.base_price
        
        for i in range(length):
            # AI: Event type depends on previous events and market conditions
            event_type = self._select_next_event_type(events)
            
            # AI: Generate complex data values that affect stock price
            event = StockEvent(
                event_type=event_type,
                timestamp=i,
                board_vote_percentage=self._generate_board_vote(event_type, events),
                media_sentiment=self._generate_media_sentiment(event_type, events),
                market_volatility=self._generate_market_volatility(events),
                company_size_factor=self._generate_company_size_factor(events),
                previous_event_impact=self._calculate_previous_impact(events)
            )
            
            events.append(event)
            
        return events
    
    def _select_next_event_type(self, previous_events: EventSequence) -> EventType:
        """Select next event type based on sequence patterns"""
        if not previous_events:
            return random.choice(list(EventType))
        
        last_event = previous_events[-1]
        
        # AI: Complex transition probabilities based on event history
        if last_event.event_type == EventType.CEO_HIRE:
            # After CEO hire, board meetings are more likely
            weights = [0.1, 0.4, 0.2, 0.1, 0.1, 0.05, 0.03, 0.02]
        elif last_event.event_type == EventType.SCANDAL:
            # After scandal, regulatory changes and board meetings more likely
            weights = [0.3, 0.3, 0.1, 0.05, 0.2, 0.02, 0.02, 0.01]
        elif len(previous_events) >= 3 and all(e.event_type == EventType.EARNINGS_REPORT for e in previous_events[-3:]):
            # After 3 earnings reports, something dramatic is likely
            weights = [0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.15, 0.05]
        else:
            # Default distribution
            weights = [0.15, 0.2, 0.25, 0.1, 0.1, 0.1, 0.05, 0.05]
        
        event_types = list(EventType)
        return event_types[np.random.choice(len(event_types), p=weights)]
    
    def _generate_board_vote(self, event_type: EventType, previous_events: EventSequence) -> float:
        """Generate board vote percentage based on event type and history"""
        base_vote = {
            EventType.CEO_HIRE: 75.0,
            EventType.BOARD_MEETING: 85.0,
            EventType.MERGER_ANNOUNCEMENT: 60.0,
            EventType.REGULATORY_CHANGE: 90.0,
        }.get(event_type, 50.0)
        
        # AI: Adjust based on recent scandals or positive events
        recent_scandals = sum(1 for e in previous_events[-5:] if e.event_type == EventType.SCANDAL)
        adjustment = -10 * recent_scandals + np.random.normal(0, 10)
        
        return np.clip(base_vote + adjustment, 0, 100)
    
    def _generate_media_sentiment(self, event_type: EventType, previous_events: EventSequence) -> float:
        """Generate media sentiment based on event type and momentum"""
        base_sentiment = {
            EventType.CEO_HIRE: 0.2,
            EventType.PRODUCT_LAUNCH: 0.6,
            EventType.SCANDAL: -0.8,
            EventType.MARKET_CRASH: -0.9,
            EventType.MERGER_ANNOUNCEMENT: 0.3,
        }.get(event_type, 0.0)
        
        # AI: Sentiment momentum from previous events
        if previous_events:
            recent_sentiment = np.mean([e.media_sentiment for e in previous_events[-3:]])
            momentum = 0.3 * recent_sentiment
        else:
            momentum = 0
        
        return np.clip(base_sentiment + momentum + np.random.normal(0, 0.2), -1, 1)
    
    def _generate_market_volatility(self, previous_events: EventSequence) -> float:
        """Generate market volatility based on recent event density"""
        if len(previous_events) < 5:
            return np.random.uniform(0.1, 0.3)
        
        # AI: Higher volatility with more frequent dramatic events
        dramatic_events = sum(1 for e in previous_events[-5:] 
                            if e.event_type in [EventType.SCANDAL, EventType.MARKET_CRASH, EventType.MERGER_ANNOUNCEMENT])
        
        base_volatility = 0.2 + 0.15 * dramatic_events
        return np.clip(base_volatility + np.random.normal(0, 0.1), 0, 1)
    
    def _generate_company_size_factor(self, previous_events: EventSequence) -> float:
        """Generate company size factor that evolves with events"""
        base_size = 1.0
        
        # AI: Company grows/shrinks based on event history
        for event in previous_events:
            if event.event_type == EventType.MERGER_ANNOUNCEMENT:
                base_size *= 1.2
            elif event.event_type == EventType.SCANDAL:
                base_size *= 0.9
            elif event.event_type == EventType.PRODUCT_LAUNCH:
                base_size *= 1.05
        
        return np.clip(base_size + np.random.normal(0, 0.1), 0.1, 10)
    
    def _calculate_previous_impact(self, previous_events: EventSequence) -> float:
        """Calculate cumulative impact of previous events"""
        if not previous_events:
            return 0.0
        
        # AI: Weighted impact of recent events with decay
        impact = 0.0
        for i, event in enumerate(previous_events[-5:]):
            weight = 0.8 ** (len(previous_events[-5:]) - i - 1)  # Decay factor
            event_impact = {
                EventType.CEO_HIRE: 0.3,
                EventType.SCANDAL: -0.6,
                EventType.PRODUCT_LAUNCH: 0.4,
                EventType.MARKET_CRASH: -0.8,
                EventType.MERGER_ANNOUNCEMENT: 0.5,
            }.get(event.event_type, 0.0)
            
            impact += weight * event_impact
        
        return np.clip(impact, -1, 1)
    
    def calculate_stock_prices(self, events: EventSequence) -> PriceSequence:
        """Calculate stock prices based on complex event interactions"""
        prices = [self.base_price]
        
        for i, event in enumerate(events):
            current_price = prices[-1]
            
            # AI: Complex price calculation considering all event data
            base_impact = self._get_base_event_impact(event.event_type)
            
            # AI: Modify impact based on event data values
            vote_modifier = (event.board_vote_percentage - 50) / 100  # -0.5 to 0.5
            sentiment_modifier = event.media_sentiment * 0.5
            volatility_modifier = event.market_volatility * np.random.normal(0, 0.3)
            size_modifier = np.log(event.company_size_factor) * 0.1
            history_modifier = event.previous_event_impact * 0.3
            
            total_impact = (base_impact + vote_modifier + sentiment_modifier + 
                          volatility_modifier + size_modifier + history_modifier)
            
            # AI: Apply impact with some randomness
            price_change = current_price * total_impact + np.random.normal(0, 2)
            new_price = max(1.0, current_price + price_change)  # Prevent negative prices
            
            prices.append(new_price)
        
        return prices[1:]  # Remove initial price
    
    def _get_base_event_impact(self, event_type: EventType) -> float:
        """Get base impact percentage for each event type"""
        return {
            EventType.CEO_HIRE: 0.05,
            EventType.BOARD_MEETING: 0.01,
            EventType.EARNINGS_REPORT: 0.03,
            EventType.MERGER_ANNOUNCEMENT: 0.15,
            EventType.REGULATORY_CHANGE: -0.02,
            EventType.PRODUCT_LAUNCH: 0.08,
            EventType.SCANDAL: -0.12,
            EventType.MARKET_CRASH: -0.25,
        }[event_type]

class SequenceDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, sequences: List[EventSequence], sequence_length: int = 10):
        self.sequences = sequences
        self.sequence_length = sequence_length
        self.label_encoder = LabelEncoder()
        
        # AI: Encode event types to integers
        all_event_types = [event.event_type.value for seq in sequences for event in seq]
        self.label_encoder.fit(all_event_types)
        
        self.samples = self._create_samples()
    
    def _create_samples(self) -> List[Tuple[List[int], int]]:
        """Create training samples from sequences"""
        samples = []
        
        for sequence in self.sequences:
            if len(sequence) <= self.sequence_length:
                continue
                
            for i in range(len(sequence) - self.sequence_length):
                # AI: Input: sequence of event types (ignoring data values)
                input_events = [int(self.label_encoder.transform([event.event_type.value])[0]) 
                              for event in sequence[i:i+self.sequence_length]]
                
                # AI: Target: next event type
                target_event = int(self.label_encoder.transform([sequence[i+self.sequence_length].event_type.value])[0])
                
                samples.append((input_events, target_event))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq, target = self.samples[idx]
        # AI: Ensure proper integer conversion for tensor creation
        input_tensor = torch.tensor(input_seq, dtype=torch.long)
        target_tensor = torch.tensor(target, dtype=torch.long)
        return input_tensor, target_tensor

class EventTransformer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, num_heads: int = 4, 
                 num_layers: int = 2, sequence_length: int = 10):
        super().__init__()
        self.embed_dim = embed_dim
        self.sequence_length = sequence_length
        
        # AI: Embedding layer for event types
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(sequence_length, embed_dim))
        
        # AI: Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # AI: Output layer to predict next event type
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # AI: Add embeddings and positional encoding
        embedded = self.embedding(x) + self.positional_encoding[:x.size(1)]
        
        # AI: Apply transformer
        transformed = self.transformer(embedded)
        
        # AI: Use last token for prediction
        output = self.output_layer(transformed[:, -1, :])
        
        return output

def train_model(model: EventTransformer, dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]], epochs: int = 50) -> List[float]:
    """Train the transformer model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    return losses

def demonstrate_complexity() -> List[EventSequence]:
    """Demonstrate why this data is hard to model with simple approaches"""
    generator = StockDataGenerator()
    
    # AI: Generate multiple sequences
    sequences = []
    all_prices = []
    
    for i in range(20):
        events = generator.generate_event_sequence(50)
        prices = generator.calculate_stock_prices(events)
        sequences.append(events)
        all_prices.extend(prices)
    
    print("=== Data Complexity Demonstration ===")
    print(f"Generated {len(sequences)} sequences with {sum(len(s) for s in sequences)} total events")
    
    # AI: Show why simple visualization fails
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(all_prices[:200])
    plt.title("Stock Prices Over Time\n(Hard to see event patterns)")
    plt.ylabel("Price")
    
    # AI: Show event type distribution
    plt.subplot(2, 2, 2)
    all_event_types = [event.event_type.value for seq in sequences for event in seq]
    event_counts = pd.Series(all_event_types).value_counts()
    event_counts.plot(kind='bar', rot=45)
    plt.title("Event Type Distribution\n(Ignores sequence dependencies)")
    
    # AI: Show correlation between event data and price changes
    plt.subplot(2, 2, 3)
    sample_events = sequences[0][:30]
    sample_prices = generator.calculate_stock_prices(sample_events)
    
    board_votes = [e.board_vote_percentage for e in sample_events]
    price_changes = [sample_prices[i+1] - sample_prices[i] if i < len(sample_prices)-1 else 0 
                    for i in range(len(sample_events))]
    
    plt.scatter(board_votes, price_changes[:len(board_votes)])
    plt.xlabel("Board Vote %")
    plt.ylabel("Price Change")
    plt.title("Board Vote vs Price Change\n(Complex, non-linear relationship)")
    
    # AI: Show sequence dependency
    plt.subplot(2, 2, 4)
    event_type_nums = [list(EventType).index(e.event_type) for e in sample_events]
    plt.plot(event_type_nums, 'o-')
    plt.title("Event Type Sequence\n(Pattern depends on history)")
    plt.ylabel("Event Type Index")
    
    plt.tight_layout()
    plt.savefig("complexity_demonstration.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return sequences

def main() -> None:
    """Main function to generate data and train transformer model"""
    print("Generating complex sequence-dependent stock data...")
    
    # AI: Generate training data
    sequences = demonstrate_complexity()
    
    # AI: Create dataset and train transformer
    print("\nTraining transformer model to predict next event type...")
    dataset = SequenceDataset(sequences, sequence_length=10)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    vocab_size = len(dataset.label_encoder.classes_)
    model = EventTransformer(vocab_size=vocab_size)
    
    print(f"Model vocabulary size: {vocab_size}")
    print(f"Training samples: {len(dataset)}")
    
    # AI: Train model
    losses = train_model(model, dataloader, epochs=1000)
    
    # AI: Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("Transformer Training Loss\n(Predicting Next Event Type)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("training_loss.png", dpi=150, bbox_inches='tight')
    
    # AI: Test model predictions
    print("\n=== Model Predictions ===")
    model.eval()
    with torch.no_grad():
        test_sequence = sequences[0][:10]
        test_input = torch.tensor([dataset.label_encoder.transform([e.event_type.value])[0] 
                                 for e in test_sequence]).unsqueeze(0)
        
        prediction = model(test_input)
        predicted_class = torch.argmax(prediction, dim=1).item()
        predicted_event = dataset.label_encoder.inverse_transform([predicted_class])[0]
        
        print("Input sequence (last 10 events):")
        for i, event in enumerate(test_sequence):
            print(f"  {i+1}. {event.event_type.value}")
        
        print(f"\nPredicted next event: {predicted_event}")
        print(f"Actual next event: {sequences[0][10].event_type.value}")
    
    print("\n=== Why This Is Complex ===")
    print("1. Simple visualization fails: Stock prices look random without event context")
    print("2. Event values matter: Board vote %, sentiment, volatility all affect outcomes")
    print("3. Sequence dependency: Event transitions depend on history, not just current state")
    print("4. Non-linear relationships: Multiple factors interact in complex ways")
    print("5. Transformer learns patterns: Predicts next event type from sequence history")

if __name__ == "__main__":
    main()
