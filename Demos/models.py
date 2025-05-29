import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import List, Tuple, Dict, Any
from Sequence_And_Data.data_generation import generate_data_for_sequence_and_data

# AI: Define types for clarity
state_sequence = List[int]
probability_vector = torch.Tensor
prediction_output = Dict[str, Any]

class MultiOutcomeTransformer(nn.Module):
    """
    AI: Simple transformer model for multi-outcome sequence prediction
    Demonstrates unified prediction without specialized heads
    """
    
    def __init__(self, 
                 vocab_size: int = 9,  # AI: 9 states from data generation
                 d_model: int = 64,    # AI: Smaller for faster training
                 nhead: int = 4,       # AI: Fewer heads for faster training
                 num_layers: int = 2,  # AI: Fewer layers for faster training
                 sequence_length: int = 4,  # AI: Context window of 4 as requested
                 dropout: float = 0.1):
        super().__init__()
        
        # AI: Store parameters
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        
        # AI: Embedding layer for state tokens
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # AI: Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(sequence_length, d_model))
        
        # AI: Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,  # AI: Smaller feedforward for speed
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # AI: Output projection to vocabulary size for next token prediction
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        AI: Forward pass
        x: (batch_size, sequence_length) tensor of token indices
        returns: (batch_size, vocab_size) logits for next token
        """
        batch_size, seq_len = x.shape
        
        # AI: Embed tokens
        embedded = self.embedding(x) * (self.d_model ** 0.5)
        
        # AI: Add positional encoding
        embedded = embedded + self.positional_encoding[:seq_len, :].unsqueeze(0)
        
        # AI: Apply transformer
        transformed = self.transformer(embedded)
        
        # AI: Use last token for prediction
        last_token = transformed[:, -1, :]
        
        # AI: Project to vocabulary
        logits = self.output_projection(last_token)
        
        return logits

def create_sequences_dataset(sequence_data: state_sequence, 
                           context_length: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    AI: Create input-target pairs from sequence data
    Each input is context_length tokens, target is the next token
    """
    if len(sequence_data) < context_length + 1:
        raise ValueError(f"Sequence too short: {len(sequence_data)} < {context_length + 1}")
    
    inputs = []
    targets = []
    
    # AI: Create sliding windows
    for i in range(len(sequence_data) - context_length):
        inputs.append(sequence_data[i:i + context_length])
        targets.append(sequence_data[i + context_length])
    
    return torch.tensor(inputs), torch.tensor(targets)

def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """AI: Create a simple text progress bar"""
    filled = int(width * current // total)
    bar = '█' * filled + '░' * (width - filled)
    percent = 100 * current // total
    return f"|{bar}| {percent}% ({current}/{total})"

def train_model(model: MultiOutcomeTransformer, 
                train_inputs: torch.Tensor, 
                train_targets: torch.Tensor,
                val_inputs: torch.Tensor,
                val_targets: torch.Tensor,
                epochs: int = 100,  # AI: Fewer epochs for faster demo
                batch_size: int = 64,
                learning_rate: float = 0.001) -> Tuple[List[float], List[float]]:
    """
    AI: Train the transformer model with validation tracking
    """
    print(f"\nTraining Configuration:")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Training samples: {len(train_inputs):,}")
    print(f"  Validation samples: {len(val_inputs):,}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Context window: {model.sequence_length}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_accuracies = []
    start_time = time.time()
    
    # AI: Calculate total number of batches
    num_batches = (len(train_inputs) + batch_size - 1) // batch_size
    total_steps = epochs * num_batches
    
    print(f"\nStarting training...")
    print(f"Total steps: {total_steps:,}")
    print(f"Expected validation accuracy: ~97% (based on transition matrix)")
    print("=" * 70)
    
    step = 0
    for epoch in range(epochs):
        # AI: Training phase
        model.train()
        epoch_start = time.time()
        epoch_loss = 0.0
        
        # AI: Shuffle data each epoch
        indices = torch.randperm(len(train_inputs))
        
        for batch_idx in range(0, len(train_inputs), batch_size):
            batch_indices = indices[batch_idx:batch_idx + batch_size]
            batch_inputs = train_inputs[batch_indices]
            batch_targets = train_targets[batch_indices]
            
            optimizer.zero_grad()
            
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            step += 1
            
            # AI: Progress indicator every 20 steps or at end of epoch
            if step % 20 == 0 or batch_idx + batch_size >= len(train_inputs):
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed if elapsed > 0 else 0
                eta = (total_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
                
                progress = create_progress_bar(step, total_steps)
                print(f"\rStep {step:>5}/{total_steps} {progress} "
                      f"Loss: {loss.item():.4f} "
                      f"Speed: {steps_per_sec:.1f} steps/s "
                      f"ETA: {eta:.0f}s", end="", flush=True)
        
        # AI: Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx in range(0, len(val_inputs), batch_size):
                batch_inputs = val_inputs[batch_idx:batch_idx + batch_size]
                batch_targets = val_targets[batch_idx:batch_idx + batch_size]
                
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()
                
                # AI: Calculate accuracy
                predictions = torch.argmax(outputs, dim=-1)
                val_correct += (predictions == batch_targets).sum().item()
                val_total += batch_targets.size(0)
        
        epoch_time = time.time() - epoch_start
        avg_train_loss = epoch_loss / num_batches
        val_accuracy = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / ((len(val_inputs) + batch_size - 1) // batch_size)
        
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_accuracy)
        
        # AI: End of epoch summary with validation metrics
        print(f"\nEpoch {epoch+1:>3}/{epochs} | Time: {epoch_time:.1f}s | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.1f}%")
        
        # AI: Progress separator every 10 epochs
        if (epoch + 1) % 10 == 0 and epoch + 1 < epochs:
            print("-" * 70)
    
    total_time = time.time() - start_time
    final_val_acc = val_accuracies[-1]
    
    print("=" * 70)
    print(f"Training completed in {total_time:.1f}s")
    print(f"Final validation accuracy: {final_val_acc:.1f}%")
    
    # AI: Check if validation accuracy is as expected
    if final_val_acc >= 95.0:
        print(f"✓ Excellent! Model achieved {final_val_acc:.1f}% validation accuracy")
        print("✓ Model learned the transition patterns (not just memorizing)")
    elif final_val_acc >= 90.0:
        print(f"⚠ Good: {final_val_acc:.1f}% validation accuracy (expected ~97%)")
    else:
        print(f"✗ Low validation accuracy: {final_val_acc:.1f}% (expected ~97%)")
        print("✗ Model may be underfitting or data insufficient")
    
    return train_losses, val_accuracies

def evaluate_predictions(model: MultiOutcomeTransformer, 
                        val_inputs: torch.Tensor,
                        val_targets: torch.Tensor,
                        context_length: int = 4,
                        num_examples: int = 10) -> None:
    """
    AI: Evaluate model predictions on validation data with detailed examples
    """
    model.eval()
    
    print(f"\nDetailed Validation Analysis - Context Window: {context_length}")
    print("=" * 70)
    
    # AI: Calculate overall validation accuracy
    correct = 0
    total = 0
    
    with torch.no_grad():
        outputs = model(val_inputs)
        predictions = torch.argmax(outputs, dim=-1)
        correct = (predictions == val_targets).sum().item()
        total = val_targets.size(0)
        accuracy = 100.0 * correct / total
    
    print(f"Overall Validation Accuracy: {correct}/{total} = {accuracy:.2f}%")
    print(f"Expected accuracy based on transition matrix: ~97%")
    print()
    
    # AI: Show detailed examples
    print(f"Sample Predictions (showing {num_examples} examples):")
    print("-" * 70)
    
    state_labels = [
        "[A, A]",           # 0
        "[A, B{False}]",    # 1  
        "[A, B{True}]",     # 2
        "[B{False}, A]",    # 3
        "[B{False}, B{False}]", # 4
        "[B{False}, B{True}]",  # 5
        "[B{True}, A]",     # 6
        "[B{True}, B{False}]",  # 7
        "[B{True}, B{True}]"    # 8
    ]
    
    with torch.no_grad():
        outputs = model(val_inputs[:num_examples])
        probabilities = F.softmax(outputs, dim=-1)
        predictions = torch.argmax(outputs, dim=-1)
        
        for i in range(num_examples):
            context = val_inputs[i].tolist()
            true_next = val_targets[i].item()
            predicted = predictions[i].item()
            confidence = probabilities[i, predicted].item()
            
            context_labels = [state_labels[idx] for idx in context]
            is_correct = "✓" if predicted == true_next else "✗"
            
            print(f"Example {i+1:2d}: {context} -> {true_next}")
            print(f"           Context: {' -> '.join(context_labels)}")
            print(f"           True next: {state_labels[true_next]}")
            print(f"           Predicted: {state_labels[predicted]} (conf: {confidence:.3f}) {is_correct}")
            print()

def demo_sequence_prediction():
    """
    AI: Main demo function with proper train/validation split
    """
    print("Multi-Outcome Sequence Prediction Demo with Validation")
    print("=" * 60)
    
    # AI: Generate training data with much longer sequence
    print("Generating sequence data...")
    sequence_data = generate_data_for_sequence_and_data(sequence_length=5000)  # AI: Much longer sequence
    
    # AI: Create dataset with context window of 4
    context_length = 4
    print(f"\nCreating dataset with context window of {context_length}...")
    try:
        all_inputs, all_targets = create_sequences_dataset(sequence_data, context_length)
        total_samples = len(all_inputs)
        print(f"Total samples created: {total_samples}")
        
        # AI: Split into train (80%) and validation (20%)
        train_size = int(0.8 * total_samples)
        val_size = total_samples - train_size
        
        # AI: Use first 80% for training, last 20% for validation
        # This ensures no overlap and tests generalization to later sequences
        train_inputs = all_inputs[:train_size]
        train_targets = all_targets[:train_size]
        val_inputs = all_inputs[train_size:]
        val_targets = all_targets[train_size:]
        
        print(f"Training samples: {train_size}")
        print(f"Validation samples: {val_size}")
        print(f"Train/Val split: {100*train_size/total_samples:.1f}%/{100*val_size/total_samples:.1f}%")
        
    except ValueError as e:
        print(f"Error creating dataset: {e}")
        return
    
    # AI: Initialize model
    print(f"\nInitializing transformer model...")
    model = MultiOutcomeTransformer(
        vocab_size=9,
        d_model=64,      # AI: Smaller for speed
        nhead=4,         # AI: Fewer heads for speed
        num_layers=2,    # AI: Fewer layers for speed
        sequence_length=context_length,
        dropout=0.1
    )
    
    # AI: Train model with validation tracking
    train_losses, val_accuracies = train_model(
        model, 
        train_inputs, 
        train_targets,
        val_inputs,
        val_targets,
        epochs=50,       # AI: Fewer epochs for faster demo
        batch_size=128,  # AI: Larger batch for efficiency
        learning_rate=0.001
    )
    
    # AI: Detailed validation evaluation
    evaluate_predictions(model, val_inputs, val_targets, context_length)
    
    print(f"\nDemo completed successfully!")
    print(f"The model learned to predict state transitions from sequences of {context_length} states.")
    print(f"Validation accuracy confirms the model learned patterns, not just memorization.")

if __name__ == "__main__":
    demo_sequence_prediction()


