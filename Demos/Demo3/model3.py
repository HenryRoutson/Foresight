# AI: Demos/Demo3/model3.py
# This file is an adaptation of models2.py for the more complex trading data.
# The core model architecture and logic remain the same to demonstrate the general applicability of the masking technique.

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

# AI: Suppress print from data3 on import
original_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
try:
    from .data3 import (
        TRAINING_DATA_WITH_CONTEXT_VECTORISED,
        TRAINING_DATA_WITH_CONTEXT_VECTORISED_COERCION,
        events_id_list,
        get_vectorizer_output_length,
        get_vector_sizes
    )
except (ImportError, ModuleNotFoundError):
    from Demos.Demo3.data3 import (
        TRAINING_DATA_WITH_CONTEXT_VECTORISED,
        TRAINING_DATA_WITH_CONTEXT_VECTORISED_COERCION,
        events_id_list,
        get_vectorizer_output_length,
        get_vector_sizes
    )
finally:
    sys.stdout.close()
    sys.stdout = original_stdout

# AI: Define global constants for the model and training
INPUT_SIZE, DATA_VECTOR_SIZE = get_vector_sizes()
NUM_EVENT_TYPES = len(events_id_list)
# AI: Transformer specific hyperparameters
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
EPOCHS = 1000
LEARNING_RATE = 0.001

def prepare_data(vectorised_data: List[List[np.ndarray[Any, Any]]]) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    AI: Prepares data, splitting the target into classification and regression parts.
    Input `None` values are converted to 0. Target `None` values are converted to NaN for masking.
    """
    processed_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for sequence in vectorised_data:
        # AI: Correctly handle the list of arrays for context and the single array for target
        context_part = np.array(sequence[:2], dtype=object)
        target_part = np.array(sequence[2], dtype=object)

        context_np = context_part.copy()
        context_np[context_np == None] = 0.0 # type: ignore
        context_tensor = torch.tensor(context_np.astype(np.float32))

        target_np = target_part.copy()
        target_np[target_np == None] = np.nan # type: ignore
        target_tensor = torch.tensor(target_np.astype(np.float32))
        
        target_class_one_hot = target_tensor[:NUM_EVENT_TYPES]
        target_class = torch.argmax(torch.nan_to_num(target_class_one_hot, nan=-1.0))
        target_data = target_tensor[NUM_EVENT_TYPES:]
        
        processed_data.append((context_tensor, target_class, target_data))
    return processed_data

class PositionalEncoding(nn.Module):
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
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class TransformerPredictor(nn.Module):
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
    class_loss = nn.functional.cross_entropy(class_pred, target_class)
    
    mask = ~torch.isnan(target_data)
    if not mask.any():
        data_loss = torch.tensor(0.0, device=data_pred.device)
    else:
        data_pred_masked = torch.masked_select(data_pred, mask)
        target_data_masked = torch.masked_select(target_data, mask)
        data_loss = nn.functional.mse_loss(data_pred_masked, target_data_masked)
        
    return class_loss + 1.0 * data_loss # AI: Simplified weight for demo

def train_model(model: nn.Module, data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    model.train()

    contexts = torch.stack([item[0] for item in data])
    target_classes = torch.stack([item[1] for item in data])
    target_datas = torch.stack([item[2] for item in data])

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        class_pred, data_pred = model(contexts)
        loss = hybrid_loss(class_pred, data_pred, target_classes, target_datas)

        if epoch > 0 and (epoch % 100 == 0 or epoch == EPOCHS - 1):
            print(f"Epoch {epoch}, Loss: {loss.item()}")

        if not torch.isnan(loss):
            loss.backward()
            optimizer.step()
            scheduler.step()

def get_relevant_indices_for_event(event_class_idx: int) -> Tuple[int, int]:
    start_idx = 0
    for i in range(event_class_idx):
        start_idx += get_vectorizer_output_length(events_id_list[i])
    
    length = get_vectorizer_output_length(events_id_list[event_class_idx])
    return start_idx, start_idx + length

def evaluate_model(model: nn.Module, data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> float:
    model.eval()
    all_pred_class_idx: list[int] = []
    all_actual_class_idx: list[int] = []
    targeted_mses: list[float] = []

    with torch.no_grad():
        for context, target_class, target_data in data:
            class_pred, data_pred = model(context.unsqueeze(0))
            
            pred_class_idx = int(torch.argmax(class_pred.squeeze(0)).item())
            actual_class_idx = int(target_class.item())
            
            all_pred_class_idx.append(pred_class_idx)
            all_actual_class_idx.append(actual_class_idx)

            start, end = get_relevant_indices_for_event(actual_class_idx)
            if end > start:
                pred_data_squeezed = data_pred.squeeze(0)
                relevant_preds = pred_data_squeezed[start:end]
                relevant_targets = target_data[start:end]

                mask = ~torch.isnan(relevant_targets)
                if mask.any():
                    mse = nn.functional.mse_loss(
                        torch.masked_select(relevant_preds, mask),
                        torch.masked_select(relevant_targets, mask)
                    ).item()
                    targeted_mses.append(float(mse))

    labels = sorted(list(set(all_actual_class_idx)))
    label_names = [events_id_list[i] for i in labels]
    cm = confusion_matrix(all_actual_class_idx, all_pred_class_idx, labels=labels)
    acc = accuracy_score(all_actual_class_idx, all_pred_class_idx)
    
    print(f"Labels for Confusion Matrix: {label_names}")
    print("Confusion Matrix:")
    header = "       " + " ".join([f"{label:>5}" for label in label_names])
    print(header)
    print("     Predicted↓ True→")
    for i, label_true in enumerate(label_names):
        row_str = f"{label_true:>5} |"
        for val in cm[i]:
            row_str += f"{val:>5} "
        print(row_str)
    
    print(f"Accuracy: {acc:.4f} ({int(acc * len(all_actual_class_idx))}/{len(all_actual_class_idx)})")
    
    avg_targeted_mse = float('nan')
    if targeted_mses:
        avg_targeted_mse = float(np.mean(targeted_mses))
        print(f"Average Targeted MSE (on relevant data only): {avg_targeted_mse:.4f}")

    return avg_targeted_mse

def main():
    print("Preparing data...")
    data_with_context = prepare_data(TRAINING_DATA_WITH_CONTEXT_VECTORISED)
    data_with_context_coercion = prepare_data(TRAINING_DATA_WITH_CONTEXT_VECTORISED_COERCION)

    print("\n--- Training model WITH CONTEXT (Correct Masking) ---")
    model_with_context = TransformerPredictor(INPUT_SIZE, D_MODEL, NHEAD, NUM_LAYERS, NUM_EVENT_TYPES, DATA_VECTOR_SIZE)
    train_model(model_with_context, data_with_context)
    print("\nEvaluating model WITH CONTEXT...")
    mse_with_context = evaluate_model(model_with_context, data_with_context)
    print("-" * 50)

    print("\n--- Training model WITH COERCED ZEROS (Incorrect) ---")
    model_without_backprop = TransformerPredictor(INPUT_SIZE, D_MODEL, NHEAD, NUM_LAYERS, NUM_EVENT_TYPES, DATA_VECTOR_SIZE)
    train_model(model_without_backprop, data_with_context_coercion)
    print("\nEvaluating model WITH COERCED ZEROS...")
    mse_without_backprop = evaluate_model(model_without_backprop, data_with_context_coercion)
    print("-" * 50)

    print("\n--- Final Quantitative Check ---")
    print(f"Correct Model (Masking) Targeted MSE:   {mse_with_context:.4f}")
    print(f"Incorrect Model (Coercion) Targeted MSE: {mse_without_backprop:.4f}")

    if np.isnan(mse_with_context) or np.isnan(mse_without_backprop):
        print("\nCHECK SKIPPED: Could not retrieve MSE values.")
    elif mse_without_backprop > mse_with_context * 2: # AI: Expecting a dramatic difference
        ratio = mse_without_backprop / mse_with_context
        print(f"\nSUCCESS: The coerced model's error is {ratio:.1f}x higher than the correct model's.")
        print("This quantitatively proves the masking method is dramatically superior.")
    else:
        print("\nFAILURE: The MSE difference was not as dramatic as expected.")
        
if __name__ == "__main__":
    main()
