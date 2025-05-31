import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import List, Tuple, Dict, Any
from Sequence_And_Data.data_generation import generate_data_for_sequence_and_data

# AI: Define types for clarity
event_sequence = List[int]  # Sequence of events (0=A, 1=B{False}, 2=B{True})
probability_vector = torch.Tensor
prediction_output = Dict[str, Any]

def create_collapsed_mapping(sequence_data: event_sequence) -> Tuple[event_sequence, Dict[int, int]]:
    """
    AI: Create collapsed event mapping where B{True} and B{False} are indistinguishable
    Original events:
    0: A, 1: B{False}, 2: B{True}
    
    Collapsed events (B{True} and B{False} -> B):
    0: A, 1: B
    """
    # AI: Define the mapping from original events to collapsed events
    original_to_collapsed = {
        0: 0,  # A -> A
        1: 1,  # B{False} -> B  
        2: 1,  # B{True} -> B
    }
    
    # AI: Apply mapping to sequence
    collapsed_sequence = [original_to_collapsed[event] for event in sequence_data]
    
    return collapsed_sequence, original_to_collapsed

class MultiOutcomeTransformer(nn.Module):
    """
    AI: Simple transformer model for multi-outcome sequence prediction
    Demonstrates unified prediction without specialized heads
    """
    
    def __init__(self, 
                 vocab_size: int = 3,     # AI: 3 events: A, B{False}, B{True}
                 d_model: int = 64,       # AI: Smaller for faster training
                 nhead: int = 4,          # AI: Fewer heads for faster training
                 num_layers: int = 2,     # AI: Fewer layers for faster training
                 sequence_length: int = 2, # AI: Context window of 2 as requested
                 dropout: float = 0.1):
        super().__init__()
        
        # AI: Store parameters
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        
        # AI: Embedding layer for event tokens
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
        seq_len = x.shape[1]
        
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

def create_sequences_dataset(sequence_data: event_sequence, 
                           context_length: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    AI: Create input-target pairs from event sequence data
    Each input is context_length events, target is the next event
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
                        event_labels: List[str],
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
    
    # AI: Show diverse examples by finding unique sequence patterns
    print(f"Sample Predictions (showing diverse examples):")
    print("-" * 70)
    
    with torch.no_grad():
        # AI: Find unique sequence patterns to show diverse examples
        unique_patterns = {}
        pattern_indices = []
        
        for i in range(len(val_inputs)):
            # AI: Convert input to tuple to use as dictionary key
            pattern = tuple(val_inputs[i].tolist())
            target = val_targets[i].item()
            pattern_key = (pattern, target)
            
            if pattern_key not in unique_patterns and len(pattern_indices) < num_examples:
                unique_patterns[pattern_key] = i
                pattern_indices.append(i)
        
        # AI: If we don't have enough unique patterns, fill with random samples
        if len(pattern_indices) < num_examples:
            remaining_indices = list(range(len(val_inputs)))
            # AI: Remove already selected indices
            for idx in pattern_indices:
                if idx in remaining_indices:
                    remaining_indices.remove(idx)
            
            # AI: Sample remaining examples randomly
            import random
            random.shuffle(remaining_indices)
            pattern_indices.extend(remaining_indices[:num_examples - len(pattern_indices)])
        
        # AI: Get outputs for selected examples
        selected_inputs = val_inputs[pattern_indices]
        selected_targets = val_targets[pattern_indices]
        outputs = model(selected_inputs)
        probabilities = F.softmax(outputs, dim=-1)
        predictions = torch.argmax(outputs, dim=-1)
        
        for i, original_idx in enumerate(pattern_indices):
            context = selected_inputs[i].tolist()
            true_next = int(selected_targets[i].item())
            predicted = int(predictions[i].item())
            confidence = float(probabilities[i, predicted].item())
            
            context_labels = [event_labels[idx] for idx in context]
            is_correct = "✓" if predicted == true_next else "✗"
            
            print(f"Example {i+1:2d}: {context} -> {true_next}")
            print(f"           Context: {' -> '.join(context_labels)}")
            print(f"           True next: {event_labels[true_next]}")
            print(f"           Predicted: {event_labels[predicted]} (conf: {confidence:.3f}) {is_correct}")
            print()
    
    return accuracy

def compare_models_with_and_without_internal_data():
    """
    AI: Main comparison function demonstrating the importance of internal data
    Run multiple experiments to test consistency
    """
    print("Multi-Outcome Sequence Prediction: Internal Data Importance Demo")
    print("=" * 80)
    print("Running 3 experiments with different random seeds to test consistency")
    print("=" * 80)
    
    # AI: Store results from all experiments
    all_results = []
    
    for experiment in range(1, 4):
        print(f"\n" + "="*80)
        print(f"EXPERIMENT {experiment}/3")
        print("="*80)
        
        # AI: Set different random seed for each experiment
        import torch
        import numpy as np
        torch.manual_seed(42 + experiment * 100)
        np.random.seed(42 + experiment * 100)
        
        # AI: Generate original event sequence data
        print("Generating event sequence data...")
        original_sequence = generate_data_for_sequence_and_data(sequence_length=5000)
        
        # AI: Create collapsed sequence (B{True} and B{False} become indistinguishable)
        print("Creating collapsed sequence (removing internal B state information)...")
        collapsed_sequence, mapping = create_collapsed_mapping(original_sequence)
        
        print(f"Original vocabulary size: 3 events (A, B{{False}}, B{{True}})")
        print(f"Collapsed vocabulary size: 2 events (A, B)")
        
        # AI: Create datasets for both versions
        context_length = 2
        
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
        
        # Model with full event information
        model_with_data = MultiOutcomeTransformer(
            vocab_size=3,  # Original 3 events
            d_model=64,
            nhead=4,
            num_layers=2,
            sequence_length=context_length,
            dropout=0.1
        )
        
        # Model without internal state information
        model_without_data = MultiOutcomeTransformer(
            vocab_size=2,  # Collapsed 2 events
            d_model=64,
            nhead=4,
            num_layers=2,
            sequence_length=context_length,
            dropout=0.1
        )
        
        # AI: Train both models
        print(f"\nTraining WITH internal data model...")
        
        train_model(
            model_with_data,
            orig_train_inputs,
            orig_train_targets, 
            orig_val_inputs,
            orig_val_targets,
            model_name=f"Exp{experiment}-WITH Internal Data",
            epochs=50,  # AI: More epochs for better convergence
            batch_size=128,
            learning_rate=0.001,
            verbose=False  # AI: Reduce verbosity for multiple runs
        )
        
        print(f"\nTraining WITHOUT internal data model...")
        
        train_model(
            model_without_data,
            coll_train_inputs,
            coll_train_targets,
            coll_val_inputs, 
            coll_val_targets,
            model_name=f"Exp{experiment}-WITHOUT Internal Data",
            epochs=50,  # AI: More epochs for better convergence
            batch_size=128,
            learning_rate=0.001,
            verbose=False  # AI: Reduce verbosity for multiple runs
        )
        
        # AI: Define event labels for evaluation
        original_event_labels = ["A", "B{False}", "B{True}"]
        collapsed_event_labels = ["A", "B"]
        
        # AI: Evaluate both models
        with_data_acc = evaluate_predictions(
            model_with_data, 
            orig_val_inputs, 
            orig_val_targets,
            original_event_labels,
            f"Exp{experiment}-WITH Internal Data",
            context_length,
            num_examples=5  # AI: Fewer examples for multiple runs
        )
        
        without_data_acc = evaluate_predictions(
            model_without_data,
            coll_val_inputs,
            coll_val_targets, 
            collapsed_event_labels,
            f"Exp{experiment}-WITHOUT Internal Data",
            context_length,
            num_examples=5  # AI: Fewer examples for multiple runs
        )
        
        # AI: Store results
        result = {
            'experiment': experiment,
            'with_data_acc': with_data_acc,
            'without_data_acc': without_data_acc,
            'difference': with_data_acc - without_data_acc
        }
        all_results.append(result)
        
        print(f"\nEXPERIMENT {experiment} RESULTS:")
        print(f"  WITH internal data:    {with_data_acc:.1f}%")
        print(f"  WITHOUT internal data: {without_data_acc:.1f}%")
        print(f"  Difference:            {with_data_acc - without_data_acc:.1f} percentage points")
    
    # AI: Summary of all experiments
    print("\n" + "="*80)
    print("SUMMARY: ALL EXPERIMENT RESULTS")
    print("="*80)
    print("Expected based on theory:")
    print("  WITH internal data:    ~100% (deterministic)")
    print("  WITHOUT internal data: ~81% (uncertain due to collapsed B states)")
    print("  Expected difference:   ~19 percentage points")
    print()
    print("Actual results:")
    
    for result in all_results:
        exp = result['experiment']
        with_acc = result['with_data_acc']
        without_acc = result['without_data_acc']
        diff = result['difference']
        print(f"  Experiment {exp}: WITH={with_acc:.1f}%, WITHOUT={without_acc:.1f}%, DIFF={diff:.1f}pp")
    
    # AI: Calculate averages
    avg_with = sum(r['with_data_acc'] for r in all_results) / len(all_results)
    avg_without = sum(r['without_data_acc'] for r in all_results) / len(all_results)
    avg_diff = sum(r['difference'] for r in all_results) / len(all_results)
    
    print(f"\nAverages across all experiments:")
    print(f"  WITH internal data:    {avg_with:.1f}%")
    print(f"  WITHOUT internal data: {avg_without:.1f}%")
    print(f"  Difference:            {avg_diff:.1f} percentage points")
    
    print(f"\n" + "="*80)
    print("ANALYSIS:")
    if avg_without > 90:
        print("⚠ PROBLEM: Model WITHOUT internal data is achieving >90% accuracy")
        print("  This is much higher than the theoretical 81% expected from the outline.")
        print("  This suggests the data generation is not creating enough uncertain scenarios")
        print("  where the internal state of B is crucial for prediction.")
    elif avg_diff > 15:
        print("✓ GOOD: Significant difference between models as expected")
    else:
        print("⚠ ISSUE: Difference is smaller than the theoretical ~19 percentage points")
    
    print(f"\nThis demonstrates why the multi-outcome prediction library models")
    print(f"both sequence patterns AND internal structured data simultaneously.")

if __name__ == "__main__":
    compare_models_with_and_without_internal_data()


