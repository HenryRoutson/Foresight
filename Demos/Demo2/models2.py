# AI: models2.py (Transformer Implementation)
# This file implements a Transformer to predict sequence outcomes using vectorized data.
# It uses a hybrid loss function and a simple Transformer encoder architecture.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import confusion_matrix, accuracy_score
from typing import List, Tuple, Any, Dict
import sys, os
import random
import math
import copy

# AI: Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# AI: Suppressing the print output from the imported data2.py for cleaner execution.
original_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
try:
    # AI: Import the vectorized data and helper functions/variables from data2.py.
    from .data2 import (
        TRAINING_DATA_WITH_CONTEXT_VECTORISED,
        TRAINING_DATA_WITHOUT_CONTEXT_VECTORISED,
        events_id_list,
        event_id_to_vectorizer,
        TRAINING_DATA_WITH_CONTEXT_VECTORISED_COERCION,
        get_vectorizer_output_length
    )
except (ImportError, ModuleNotFoundError):
    from Demos.Demo2.data2 import (
        TRAINING_DATA_WITH_CONTEXT_VECTORISED,
        TRAINING_DATA_WITHOUT_CONTEXT_VECTORISED,
        events_id_list,
        event_id_to_vectorizer,
        TRAINING_DATA_WITH_CONTEXT_VECTORISED_COERCION,
        get_vectorizer_output_length
    )
finally:
    # AI: Restore stdout right after the import.
    sys.stdout.close()
    sys.stdout = original_stdout

# AI: Define global constants for the model and training
INPUT_SIZE = len(TRAINING_DATA_WITH_CONTEXT_VECTORISED[0][0])
NUM_EVENT_TYPES = len(events_id_list)
DATA_VECTOR_SIZE = INPUT_SIZE - NUM_EVENT_TYPES
# AI: Transformer specific hyperparameters
D_MODEL = 32 # AI: Reverted to larger model
NHEAD = 4    # AI: Reverted to larger model
NUM_LAYERS = 2 # AI: Reverted to larger model

EPOCHS = 1000 # AI: Increased epochs for better convergence
LEARNING_RATE = 0.001 # AI: Adjusted learning rate for more stable training
PATIENCE = 100 # AI: Added for early stopping

def prepare_data(vectorised_data: List[List[np.ndarray[Any, Any]]]) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    AI: Prepares data, splitting the target into classification and regression parts.
    Input `None` values are converted to 0. Target `None` values are converted to NaN for masking.
    """
    processed_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for sequence in vectorised_data:
        # AI: Create a copy of the sequence to modify
        sequence_np = np.array([list(item) for item in sequence], dtype=object)
        
        # AI: Process context (inputs): replace None with 0
        context_np = sequence_np[:2, :].copy()
        context_np[context_np == None] = 0.0  # type: ignore
        context_tensor = torch.tensor(context_np.astype(np.float32))

        # AI: Process target: replace None with NaN for loss calculation
        target_np = sequence_np[2, :].copy()
        target_np[target_np == None] = np.nan  # type: ignore
        target_tensor = torch.tensor(target_np.astype(np.float32))
        
        # AI: The one-hot encoding part of the target should be clean of NaNs, but this is safer.
        target_class_one_hot = target_tensor[:NUM_EVENT_TYPES]
        target_class = torch.argmax(torch.nan_to_num(target_class_one_hot, nan=-1.0))

        target_data = target_tensor[NUM_EVENT_TYPES:]
        
        processed_data.append((context_tensor, target_class, target_data))
    return processed_data

class PositionalEncoding(nn.Module):
    """
    AI: Injects positional information into the input embeddings.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class TransformerPredictor(nn.Module):
    """
    AI: A Transformer-based model for sequence prediction with positional encoding.
    """
    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int, num_classes: int, data_vector_size: int):
        super(TransformerPredictor, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=d_model*4, dropout=0.0)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classification_head = nn.Linear(d_model, num_classes)
        self.regression_head = nn.Linear(d_model, data_vector_size)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src = self.input_projection(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        last_output = output[:, -1, :]
        class_pred = self.classification_head(last_output)
        data_pred = self.regression_head(last_output)
        return class_pred, data_pred

def hybrid_loss(class_pred: torch.Tensor, data_pred: torch.Tensor, 
                target_class: torch.Tensor, target_data: torch.Tensor) -> torch.Tensor:
    """
    AI: Combines Cross-Entropy loss for classification and a masked MSE for regression.
    """
    # AI: Classification loss
    class_loss = nn.functional.cross_entropy(class_pred, target_class)
    
    # AI: Regression loss (masked)
    mask = ~torch.isnan(target_data)
    if not mask.any():
        data_loss = torch.tensor(0.0, device=data_pred.device)
    else:
        data_pred_masked = torch.masked_select(data_pred, mask)
        target_data_masked = torch.masked_select(target_data, mask)
        data_loss = nn.functional.mse_loss(data_pred_masked, target_data_masked)
        
    # AI: Weighting data_loss to prevent it from overwhelming classification loss.
    return class_loss + 1.0 * data_loss

def train_model(model: nn.Module, train_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], val_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[Dict[str, List[Any]], int]:
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = 0
    history: Dict[str, List[Any]] = {'epochs': [], 'train_loss': [], 'val_loss': [], 'train_mse': [], 'val_mse': []}

    # AI: Collate data into a single batch for stable training
    train_contexts = torch.stack([item[0] for item in train_data])
    train_target_classes = torch.stack([item[1] for item in train_data])
    train_target_datas = torch.stack([item[2] for item in train_data])

    val_contexts = torch.stack([item[0] for item in val_data])
    val_target_classes = torch.stack([item[1] for item in val_data])
    val_target_datas = torch.stack([item[2] for item in val_data])

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        class_pred, data_pred = model(train_contexts)
        loss = hybrid_loss(class_pred, data_pred, train_target_classes, train_target_datas)

        if not torch.isnan(loss):
            loss.backward()
            optimizer.step()
            scheduler.step()

        # AI: Validation and metrics phase
        model.eval()
        with torch.no_grad():
            train_mse = calculate_targeted_mse(class_pred, data_pred, train_target_classes, train_target_datas)
            
            val_class_pred, val_data_pred = model(val_contexts)
            val_loss = hybrid_loss(val_class_pred, val_data_pred, val_target_classes, val_target_datas)
            val_mse = calculate_targeted_mse(val_class_pred, val_data_pred, val_target_classes, val_target_datas)

        history['epochs'].append(epoch)
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss.item())
        history['train_mse'].append(train_mse)
        history['val_mse'].append(val_mse)

        if epoch > 0 and epoch % 100 == 0:
            print(f"Epoch {epoch}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}, Train MSE: {train_mse:.6f}, Val MSE: {val_mse:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch} due to no improvement in validation loss for {PATIENCE} epochs.")
            break
            
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return history, best_epoch

def vector_to_label(predicted_class_idx: int, predicted_data: np.ndarray[Any, Any]) -> str:
    """
    AI: Converts model output (class index and data vector) to a simple string label.
    """
    event_id = events_id_list[predicted_class_idx]
    if event_id == "A":
        return "A"
    if event_id == "B_without_context":
        return "B"
    if event_id == "B_with_context":
        vectorizer = event_id_to_vectorizer.get(event_id)
        if not vectorizer:
            return "B"
        
        try:
            feature_names = list(vectorizer.get_feature_names_out())
            bool_data_idx = feature_names.index("bool_data")
        except (ValueError, AttributeError):
            return "B"

        # AI: Calculate the starting index for this event type's data within the regression vector.
        regression_data_start_idx = 0
        for i in range(predicted_class_idx):
            regression_data_start_idx += get_vectorizer_output_length(events_id_list[i])
        
        # AI: The final index is the start index plus the feature's index within its own vectorizer.
        final_bool_idx = regression_data_start_idx + bool_data_idx
        
        if final_bool_idx >= len(predicted_data):
            return "B"

        return "T" if predicted_data[final_bool_idx] > 0.5 else "F"
    return "Unknown"

def get_relevant_indices_for_event(event_class_idx: int) -> Tuple[int, int]:
    """
    AI: Calculates the start and end slice indices for an event's data in the regression vector.
    """
    start_idx = 0
    for i in range(event_class_idx):
        start_idx += get_vectorizer_output_length(events_id_list[i])
    
    length = get_vectorizer_output_length(events_id_list[event_class_idx])
    return start_idx, start_idx + length

def calculate_targeted_mse(
    class_pred: torch.Tensor, data_pred: torch.Tensor, 
    target_classes: torch.Tensor, target_datas: torch.Tensor
) -> float:
    """
    AI: Calculates the 'targeted' MSE, focusing only on the regression data
    relevant to the true class for each item in the batch.
    """
    targeted_mses = []
    num_items = class_pred.shape[0]

    with torch.no_grad():
        for i in range(num_items):
            actual_class_idx = int(target_classes[i].item())
            
            start, end = get_relevant_indices_for_event(actual_class_idx)
            if end > start:
                relevant_preds = data_pred[i, start:end]
                relevant_targets = target_datas[i, start:end]

                mask = ~torch.isnan(relevant_targets)
                if mask.any():
                    mse = nn.functional.mse_loss(
                        torch.masked_select(relevant_preds, mask),
                        torch.masked_select(relevant_targets, mask)
                    ).item()
                    targeted_mses.append(float(mse))

    if not targeted_mses:
        return float('nan')
    return float(np.mean(targeted_mses))

def evaluate_model(model: nn.Module, data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
    """
    AI: Evaluates the model's performance, printing metrics and returning the targeted MSE.
    """
    model.eval()
    actual_labels: list[str] = []
    predicted_labels: list[str] = []
    targeted_mses: list[float] = []
    all_predictions: list[np.ndarray[Any, Any]] = []
    all_targets: list[np.ndarray[Any, Any]] = []

    with torch.no_grad():
        for context, target_class, target_data in data:
            class_pred, data_pred = model(context.unsqueeze(0))
            
            pred_class_idx = int(torch.argmax(class_pred.squeeze(0)).item())
            pred_data_squeezed = data_pred.squeeze(0)

            actual_class_idx = int(target_class.item())
            
            # AI: Store predictions and targets for detailed analysis
            all_predictions.append(pred_data_squeezed.cpu().numpy())
            all_targets.append(target_data.cpu().numpy())

            # AI: Calculate Targeted MSE
            start, end = get_relevant_indices_for_event(actual_class_idx)
            if end > start: # AI: Only if there is regression data for this event
                relevant_preds = pred_data_squeezed[start:end]
                relevant_targets = target_data[start:end]

                # AI: Ensure we don't compare against NaN in the correct "WITH CONTEXT" case
                mask = ~torch.isnan(relevant_targets)
                if mask.any():
                    mse = nn.functional.mse_loss(
                        torch.masked_select(relevant_preds, mask),
                        torch.masked_select(relevant_targets, mask)
                    ).item()
                    targeted_mses.append(float(mse))

            # AI: Get string labels for classification metrics
            pred_data_np: np.ndarray[Any, Any] = pred_data_squeezed.cpu().numpy()
            actual_data_np: np.ndarray[Any, Any] = target_data.cpu().numpy()
            predicted_labels.append(vector_to_label(pred_class_idx, pred_data_np))
            actual_labels.append(vector_to_label(actual_class_idx, actual_data_np))
    
    labels: list[str] = sorted(list(set(actual_labels)))
    if not labels:
        print("No data to evaluate.")
        return float('nan'), {}, {}
        
    cm: np.ndarray[Any, Any] = confusion_matrix(actual_labels, predicted_labels, labels=labels)
    acc: float = float(accuracy_score(actual_labels, predicted_labels))
    
    print(f"Labels for Confusion Matrix: {labels}")
    print("Confusion Matrix:")
    header = "       " + " ".join([f"{label:>5}" for label in labels])
    print(header)
    print("     Predicted↓ True→")
    for i, label_true in enumerate(labels):
        row_str = f"{label_true:>5} |"
        for val in cm[i]:
            row_str += f"{val:>5} "
        print(row_str)
    
    print(f"Accuracy: {acc:.4f} ({int(acc * len(actual_labels))}/{len(actual_labels)})")
    
    avg_targeted_mse = float('nan')
    if targeted_mses:
        avg_targeted_mse = float(np.mean(targeted_mses))
        print(f"Average Targeted MSE (on relevant data only): {avg_targeted_mse:.6f}")

    classification_metrics = {
        'accuracy': acc,
        'confusion_matrix': cm,
        'labels': labels
    }
    evaluation_data = {
        'predictions': all_predictions,
        'targets': all_targets
    }

    return avg_targeted_mse, classification_metrics, evaluation_data

def main():
    """
    AI: Main execution block. Trains and evaluates models for both with and without context data.
    """
    print("Preparing data...")
    data_with_context = prepare_data(TRAINING_DATA_WITH_CONTEXT_VECTORISED)
    data_without_context = prepare_data(TRAINING_DATA_WITHOUT_CONTEXT_VECTORISED)
    data_with_context_coercion = prepare_data(TRAINING_DATA_WITH_CONTEXT_VECTORISED_COERCION)


    # AI: Shuffle and split data for training and validation
    random.shuffle(data_with_context)
    split_idx_wc = int(len(data_with_context) * 0.8)
    train_data_with_context = data_with_context[:split_idx_wc]
    val_data_with_context = data_with_context[split_idx_wc:]
    if not train_data_with_context or not val_data_with_context:
        train_data_with_context = data_with_context
        val_data_with_context = data_with_context

    random.shuffle(data_with_context_coercion)
    split_idx_wcc = int(len(data_with_context_coercion) * 0.8)
    train_data_with_context_coercion = data_with_context_coercion[:split_idx_wcc]
    val_data_with_context_coercion = data_with_context_coercion[split_idx_wcc:]
    if not train_data_with_context_coercion or not val_data_with_context_coercion:
        train_data_with_context_coercion = data_with_context_coercion
        val_data_with_context_coercion = data_with_context_coercion

    # --- WITH CONTEXT ---
    # performance should be perfect
    print("\nTraining model WITH context...")
    model_with_context = TransformerPredictor(INPUT_SIZE, D_MODEL, NHEAD, NUM_LAYERS, NUM_EVENT_TYPES, DATA_VECTOR_SIZE)
    history_wc, best_epoch_wc = train_model(model_with_context, train_data_with_context, val_data_with_context)
    print("Evaluating model WITH context...")
    mse_with_context, metrics_wc, evals_wc = evaluate_model(model_with_context, data_with_context)
    print("-" * 40)

    # # --- WITHOUT CONTEXT ---
    # # performance should be ok
    # print("\nTraining model WITHOUT context...")
    # model_without_context = TransformerPredictor(INPUT_SIZE, D_MODEL, NHEAD, NUM_LAYERS, NUM_EVENT_TYPES, DATA_VECTOR_SIZE)
    # train_model(model_without_context, data_without_context)
    # print("Evaluating model WITHOUT context...")
    # evaluate_model(model_without_context, data_without_context)
    # print("-" * 40)


    # --- WITH CONTEXT BUT WITHOUT BACKPROP NONE VALUES ---
    # performance should be bad
    print("\nTraining model WITHOUT BACKPROP NONE VALUES...")
    model_without_backprop_none_values = TransformerPredictor(INPUT_SIZE, D_MODEL, NHEAD, NUM_LAYERS, NUM_EVENT_TYPES, DATA_VECTOR_SIZE)
    history_wcc, best_epoch_wcc = train_model(model_without_backprop_none_values, train_data_with_context_coercion, val_data_with_context_coercion)
    print("Evaluating model WITHOUT BACKPROP NONE VALUES...")
    mse_without_backprop, metrics_wcc, evals_wcc = evaluate_model(model_without_backprop_none_values, data_with_context_coercion)
    print("-" * 40)

    # --- Save Results ---
    results = {
        'num_trials': 1,
        'correct_model': {
            'mse_results': [mse_with_context],
            'mean_mse': mse_with_context,
            'std_mse': 0,
            'epochs': [best_epoch_wc],
            'all_learning_curves': [history_wc],
            'all_evaluations': [evals_wc],
            'final_metrics': metrics_wc
        },
        'incorrect_model': {
            'mse_results': [mse_without_backprop],
            'mean_mse': mse_without_backprop,
            'std_mse': 0,
            'epochs': [best_epoch_wcc],
            'all_learning_curves': [history_wcc],
            'all_evaluations': [evals_wcc],
            'final_metrics': metrics_wcc
        }
    }

    results_path = os.path.join(os.path.dirname(__file__), 'results.npy')
    np.save(results_path, results, allow_pickle=True)
    print(f"\nResults saved to {results_path}")


    # --- FINAL CHECK ---
    print("\n--- Final Quantitative Check ---")
    print(f"Correct Model (with masking) Targeted MSE:   {mse_with_context:.6f}")
    print(f"Incorrect Model (coerced to 0) Targeted MSE: {mse_without_backprop:.6f}")

    if np.isnan(mse_with_context) or np.isnan(mse_without_backprop):
        print("\nCHECK SKIPPED: Could not retrieve MSE values for comparison.")
    elif mse_without_backprop > mse_with_context:
        print("\nSUCCESS: The model trained with incorrect zero-coercion has a higher regression error.")
        print("This quantitatively proves that masking is the superior method.")
    else:
        print("\nFAILURE: The targeted MSE check did not show the expected difference.")
        print(f"Difference: {mse_with_context - mse_without_backprop:.8f}. This may be due to model parameters, data simplicity, or random seed.")



if __name__ == "__main__":
    main()