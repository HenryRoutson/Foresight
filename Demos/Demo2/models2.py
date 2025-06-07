# AI: models2.py (Transformer Implementation)
# This file implements a Transformer to predict sequence outcomes using vectorized data.
# It uses a hybrid loss function and a simple Transformer encoder architecture.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import confusion_matrix, accuracy_score
from typing import List, Tuple, Any
import sys, os
import random
import math

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

EPOCHS = 1500 # AI: Increased epochs for better convergence
LEARNING_RATE = 0.001 # AI: Adjusted learning rate for more stable training

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
        context_np[context_np == None] = 0.0
        context_tensor = torch.tensor(context_np.astype(np.float32))

        # AI: Process target: replace None with NaN for loss calculation
        target_np = sequence_np[2, :].copy()
        target_np[target_np == None] = np.nan
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
    return class_loss + 0.1 * data_loss

def train_model(model: nn.Module, data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    model.train()

    # AI: Collate data into a single batch for stable training
    contexts = torch.stack([item[0] for item in data])
    target_classes = torch.stack([item[1] for item in data])
    target_datas = torch.stack([item[2] for item in data])

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        class_pred, data_pred = model(contexts)
        loss = hybrid_loss(class_pred, data_pred, target_classes, target_datas)

        if epoch > 0 and epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

        if not torch.isnan(loss):
            loss.backward()
            optimizer.step()
            scheduler.step()

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

def evaluate_model(model: nn.Module, data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    model.eval()
    actual_labels: list[str] = []
    predicted_labels: list[str] = []

    with torch.no_grad():
        for context, target_class, target_data in data:
            class_pred, data_pred = model(context.unsqueeze(0))
            
            pred_class_idx = int(torch.argmax(class_pred.squeeze(0)).item())
            pred_data_np: np.ndarray[Any, Any] = data_pred.squeeze(0).cpu().numpy()

            actual_class_idx = int(target_class.item())
            actual_data_np: np.ndarray[Any, Any] = target_data.cpu().numpy()
            
            predicted_labels.append(vector_to_label(pred_class_idx, pred_data_np))
            actual_labels.append(vector_to_label(actual_class_idx, actual_data_np))
    
    labels: list[str] = sorted(list(set(actual_labels)))
    if not labels:
        print("No data to evaluate.")
        return
        
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

def main():
    """
    AI: Main execution block. Trains and evaluates models for both with and without context data.
    """
    print("Preparing data...")
    data_with_context = prepare_data(TRAINING_DATA_WITH_CONTEXT_VECTORISED)
    data_without_context = prepare_data(TRAINING_DATA_WITHOUT_CONTEXT_VECTORISED)
    data_with_context_coercion = prepare_data(TRAINING_DATA_WITH_CONTEXT_VECTORISED_COERCION)


    # --- WITH CONTEXT ---
    # performance should be perfect
    print("\nTraining model WITH context...")
    model_with_context = TransformerPredictor(INPUT_SIZE, D_MODEL, NHEAD, NUM_LAYERS, NUM_EVENT_TYPES, DATA_VECTOR_SIZE)
    train_model(model_with_context, data_with_context)
    print("Evaluating model WITH context...")
    evaluate_model(model_with_context, data_with_context)
    print("-" * 40)

    # --- WITHOUT CONTEXT ---
    # performance should be ok
    print("\nTraining model WITHOUT context...")
    model_without_context = TransformerPredictor(INPUT_SIZE, D_MODEL, NHEAD, NUM_LAYERS, NUM_EVENT_TYPES, DATA_VECTOR_SIZE)
    train_model(model_without_context, data_without_context)
    print("Evaluating model WITHOUT context...")
    evaluate_model(model_without_context, data_without_context)
    print("-" * 40)


    # --- WITHOUT BACKPROP NONE VALUES ---
    # performance should be bad
    print("\nTraining model WITHOUT BACKPROP NONE VALUES...")
    model_without_backprop_none_values = TransformerPredictor(INPUT_SIZE, D_MODEL, NHEAD, NUM_LAYERS, NUM_EVENT_TYPES, DATA_VECTOR_SIZE)
    train_model(model_without_backprop_none_values, data_with_context_coercion)
    print("Evaluating model WITHOUT BACKPROP NONE VALUES...")
    evaluate_model(model_without_backprop_none_values, data_with_context_coercion)
    print("-" * 40)



if __name__ == "__main__":
    main()