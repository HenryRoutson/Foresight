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

def create_collapsed_mapping(sequence_data: state_sequence) -> Tuple[state_sequence, Dict[int, int]]:
    """
    AI: Create collapsed state mapping where B{True} and B{False} are indistinguishable
    Original states:
    0: [A, A], 1: [A, B{False}], 2: [A, B{True}], 3: [B{False}, A], 
    4: [B{False}, B{False}], 5: [B{False}, B{True}], 6: [B{True}, A], 
    7: [B{True}, B{False}], 8: [B{True}, B{True}]
    
    Collapsed states (B{True} and B{False} -> B):
    0: [A, A], 1: [A, B], 2: [B, A], 3: [B, B]
    """
    # AI: Define the mapping from original states to collapsed states
    original_to_collapsed = {
        0: 0,  # [A, A] -> [A, A]
        1: 1,  # [A, B{False}] -> [A, B]  
        2: 1,  # [A, B{True}] -> [A, B]
        3: 2,  # [B{False}, A] -> [B, A]
        4: 3,  # [B{False}, B{False}] -> [B, B]
        5: 3,  # [B{False}, B{True}] -> [B, B]
        6: 2,  # [B{True}, A] -> [B, A]
        7: 3,  # [B{True}, B{False}] -> [B, B]
        8: 3,  # [B{True}, B{True}] -> [B, B]
    }
    
    # AI: Apply mapping to sequence
    collapsed_sequence = [original_to_collapsed[state] for state in sequence_data]
    
    return collapsed_sequence, original_to_collapsed

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
                model_name: str = "Model",
                epochs: int = 100,  # AI: Fewer epochs for faster demo
                batch_size: int = 64,
                learning_rate: float = 0.001,
                verbose: bool = True) -> Tuple[List[float], List[float]]:
    """
    AI: Train the transformer model with validation tracking
    """
    if verbose:
        print(f"\n{model_name} Training Configuration:")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Vocabulary size: {model.vocab_size}")
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
    
    if verbose:
        print(f"\nStarting {model_name} training...")
        print(f"Total steps: {total_steps:,}")
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
            if verbose and (step % 20 == 0 or batch_idx + batch_size >= len(train_inputs)):
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed if elapsed > 0 else 0
                eta = (total_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
                
                progress = create_progress_bar(step, total_steps)
                print(f"\r{model_name} Step {step:>5}/{total_steps} {progress} "
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
        if verbose:
            print(f"\n{model_name} Epoch {epoch+1:>3}/{epochs} | Time: {epoch_time:.1f}s | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val Acc: {val_accuracy:.1f}%")
        
        # AI: Progress separator every 10 epochs
        if verbose and (epoch + 1) % 10 == 0 and epoch + 1 < epochs:
            print("-" * 70)
    
    total_time = time.time() - start_time
    final_val_acc = val_accuracies[-1]
    
    if verbose:
        print("=" * 70)
        print(f"{model_name} training completed in {total_time:.1f}s")
        print(f"Final validation accuracy: {final_val_acc:.1f}%")
    
    return train_losses, val_accuracies

def evaluate_predictions(model: MultiOutcomeTransformer, 
                        val_inputs: torch.Tensor,
                        val_targets: torch.Tensor,
                        state_labels: List[str],
                        model_name: str = "Model",
                        context_length: int = 4,
                        num_examples: int = 10) -> float:
    """
    AI: Evaluate model predictions on validation data with detailed examples
    """
    model.eval()
    
    print(f"\n{model_name} Validation Analysis - Context Window: {context_length}")
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
    print()
    
    # AI: Show detailed examples
    print(f"Sample Predictions (showing {num_examples} examples):")
    print("-" * 70)
    
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
    
    return accuracy

def compare_models_with_and_without_internal_data():
    """
    AI: Main comparison function demonstrating the importance of internal data
    """
    print("Multi-Outcome Sequence Prediction: Internal Data Importance Demo")
    print("=" * 80)
    print("Comparing models WITH and WITHOUT access to internal B state information")
    print("=" * 80)
    
    # AI: Generate original sequence data
    print("Generating sequence data...")
    original_sequence = generate_data_for_sequence_and_data(sequence_length=5000)
    
    # AI: Create collapsed sequence (B{True} and B{False} become indistinguishable)
    print("Creating collapsed sequence (removing internal B state information)...")
    collapsed_sequence, mapping = create_collapsed_mapping(original_sequence)
    
    print(f"Original vocabulary size: 9 states")
    print(f"Collapsed vocabulary size: 4 states")
    print(f"State mapping: {mapping}")
    
    # AI: Create datasets for both versions
    context_length = 4
    
    # Original data (with internal state info)
    original_inputs, original_targets = create_sequences_dataset(original_sequence, context_length)
    print(f"\nOriginal dataset: {len(original_inputs)} samples")
    
    # Collapsed data (without internal state info)  
    collapsed_inputs, collapsed_targets = create_sequences_dataset(collapsed_sequence, context_length)
    print(f"Collapsed dataset: {len(collapsed_inputs)} samples")
    
    # AI: Split both datasets
    train_size = int(0.8 * len(original_inputs))
    
    # Original splits
    orig_train_inputs = original_inputs[:train_size]
    orig_train_targets = original_targets[:train_size]
    orig_val_inputs = original_inputs[train_size:]
    orig_val_targets = original_targets[train_size:]
    
    # Collapsed splits
    coll_train_inputs = collapsed_inputs[:train_size]
    coll_train_targets = collapsed_targets[:train_size]
    coll_val_inputs = collapsed_inputs[train_size:]
    coll_val_targets = collapsed_targets[train_size:]
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {len(orig_val_inputs)}")
    
    # AI: Initialize models
    print(f"\nInitializing models...")
    
    # Model with full state information
    model_with_data = MultiOutcomeTransformer(
        vocab_size=9,  # Original 9 states
        d_model=64,
        nhead=4,
        num_layers=2,
        sequence_length=context_length,
        dropout=0.1
    )
    
    # Model without internal state information
    model_without_data = MultiOutcomeTransformer(
        vocab_size=4,  # Collapsed 4 states
        d_model=64,
        nhead=4,
        num_layers=2,
        sequence_length=context_length,
        dropout=0.1
    )
    
    # AI: Train both models
    print(f"\n" + "="*80)
    print("TRAINING MODEL WITH INTERNAL DATA (B{True} vs B{False} distinguishable)")
    print("="*80)
    
    with_data_losses, with_data_accuracies = train_model(
        model_with_data,
        orig_train_inputs,
        orig_train_targets, 
        orig_val_inputs,
        orig_val_targets,
        model_name="WITH Internal Data",
        epochs=30,  # Fewer epochs for demo
        batch_size=128,
        learning_rate=0.001,
        verbose=True
    )
    
    print(f"\n" + "="*80)
    print("TRAINING MODEL WITHOUT INTERNAL DATA (B{True} and B{False} indistinguishable)")  
    print("="*80)
    
    without_data_losses, without_data_accuracies = train_model(
        model_without_data,
        coll_train_inputs,
        coll_train_targets,
        coll_val_inputs, 
        coll_val_targets,
        model_name="WITHOUT Internal Data",
        epochs=30,  # Fewer epochs for demo
        batch_size=128,
        learning_rate=0.001,
        verbose=True
    )
    
    # AI: Define state labels for evaluation
    original_state_labels = [
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
    
    collapsed_state_labels = [
        "[A, A]",    # 0
        "[A, B]",    # 1
        "[B, A]",    # 2  
        "[B, B]"     # 3
    ]
    
    # AI: Evaluate both models
    with_data_acc = evaluate_predictions(
        model_with_data, 
        orig_val_inputs, 
        orig_val_targets,
        original_state_labels,
        "WITH Internal Data",
        context_length
    )
    
    without_data_acc = evaluate_predictions(
        model_without_data,
        coll_val_inputs,
        coll_val_targets, 
        collapsed_state_labels,
        "WITHOUT Internal Data",
        context_length
    )
    
    # AI: Final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON: IMPACT OF INTERNAL DATA ON ACCURACY")
    print("="*80)
    print(f"Model WITH internal data (B{{True}} vs B{{False}}):    {with_data_acc:.1f}%")
    print(f"Model WITHOUT internal data (B only):               {without_data_acc:.1f}%")
    print(f"Accuracy difference:                                {with_data_acc - without_data_acc:.1f} percentage points")
    
    if with_data_acc > without_data_acc:
        print(f"\n✓ CONCLUSION: Internal structured data improves prediction accuracy!")
        print(f"✓ The {with_data_acc - without_data_acc:.1f}% improvement demonstrates the value of modeling")
        print(f"  complete event structures rather than just event types.")
    else:
        print(f"\n⚠ Unexpected result: No improvement from internal data")
        print(f"  This might indicate the transition patterns don't depend on B's internal state")
    
    print(f"\nThis demonstrates why the multi-outcome prediction library models")
    print(f"both sequence patterns AND internal structured data simultaneously.")

if __name__ == "__main__":
    compare_models_with_and_without_internal_data()


