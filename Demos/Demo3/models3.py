# AI: Demos/Demo3/model3.py
# This file is an adaptation of models2.py for the more complex trading data.
# The core model architecture and logic remain the same to demonstrate the general applicability of the masking technique.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Any, Dict
import sys, os
import random
import math
import copy
from scipy.stats import ttest_ind # type: ignore # AI: Consider `pip install types-scipy` to remove this ignore

# AI: Add project root to path to resolve import errors
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# AI: Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# AI: Suppress print from data3 on import
original_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
try:
    from Demos.Demo3.data3 import (
        TRAINING_DATA_WITH_CONTEXT_VECTORISED,
        TRAINING_DATA_WITH_CONTEXT_VECTORISED_COERCION,
        events_id_list,
        get_vectorizer_output_length,
        get_vector_sizes
    )
except (ImportError, ModuleNotFoundError):
    # AI: This fallback is now less likely to be needed but kept for robustness.
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

# AI: Define a type alias for our data structure for clarity
DataType = List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

# AI: Define global constants for the model and training
INPUT_SIZE, DATA_VECTOR_SIZE = get_vector_sizes()
NUM_EVENT_TYPES = len(events_id_list)
# AI: Transformer specific hyperparameters
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
EPOCHS = 20000
LEARNING_RATE = 0.001

def prepare_data(vectorised_data: List[List[np.ndarray[Any, Any]]]) -> DataType:
    """
    AI: Prepares data, splitting the target into classification and regression parts.
    Input `None` values are converted to 0. Target `None` values are converted to NaN for masking.
    """
    processed_data: DataType = []
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

def calculate_mse(data_pred: torch.Tensor, target_data: torch.Tensor) -> torch.Tensor:
    """AI: Calculates the masked Mean Squared Error."""
    mask = ~torch.isnan(target_data)
    if not mask.any():
        return torch.tensor(0.0, device=data_pred.device)
    
    data_pred_masked = torch.masked_select(data_pred, mask)
    target_data_masked = torch.masked_select(target_data, mask)
    return nn.functional.mse_loss(data_pred_masked, target_data_masked)

def train_model(
    model: nn.Module, train_data: DataType, val_data: DataType, 
    patience: int = 100, capture_learning_curves: bool = False
) -> Tuple[int, dict[str, list[float]]]:
    """
    AI: Trains the model using the provided data, with early stopping based on validation loss.
    Returns the best epoch and, optionally, the learning curves.
    """
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # --- Early Stopping Setup ---
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    # --- Learning Curve Data ---
    learning_curves: dict[str, list[float]] = {"epochs": [], "train_loss": [], "val_loss": [], "train_mse": [], "val_mse": []}

    # --- Prepare data Tensors ---
    train_contexts = torch.stack([item[0] for item in train_data])
    train_target_classes = torch.stack([item[1] for item in train_data])
    train_target_datas = torch.stack([item[2] for item in train_data])

    val_contexts = torch.stack([item[0] for item in val_data])
    val_target_classes = torch.stack([item[1] for item in val_data])
    val_target_datas = torch.stack([item[2] for item in val_data])

    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        optimizer.zero_grad()
        class_pred, data_pred = model(train_contexts)
        train_loss = hybrid_loss(class_pred, data_pred, train_target_classes, train_target_datas)

        if not torch.isnan(train_loss):
            train_loss.backward()
            optimizer.step()
            scheduler.step()

        # --- Validation Phase ---
        model.eval()
        with torch.no_grad():
            val_class_pred, val_data_pred = model(val_contexts)
            val_loss = hybrid_loss(val_class_pred, val_data_pred, val_target_classes, val_target_datas)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
            if capture_learning_curves:
                train_mse = calculate_mse(data_pred, train_target_datas)
                val_mse = calculate_mse(val_data_pred, val_target_datas)
                learning_curves["epochs"].append(epoch)
                learning_curves["train_loss"].append(train_loss.item())
                learning_curves["val_loss"].append(val_loss.item())
                learning_curves["train_mse"].append(train_mse.item())
                learning_curves["val_mse"].append(val_mse.item())

        # --- Early Stopping Check ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
            break
            
    # --- Load best model weights ---
    print("\nFinished training. Restoring best model weights.")
    if best_model_state:
        model.load_state_dict(best_model_state)
    return best_epoch, learning_curves

def get_relevant_indices_for_event(event_class_idx: int) -> Tuple[int, int]:
    start_idx = 0
    for i in range(event_class_idx):
        start_idx += get_vectorizer_output_length(events_id_list[i])
    
    length = get_vectorizer_output_length(events_id_list[event_class_idx])
    return start_idx, start_idx + length

def evaluate_model(model: nn.Module, data: DataType) -> Dict[str, Any]:
    model.eval()
    all_pred_class_idx: list[int] = []
    all_actual_class_idx: list[int] = []
    targeted_mses: list[float] = []
    all_data_preds: list[np.ndarray[Any, Any]] = []
    all_target_datas: list[np.ndarray[Any, Any]] = []

    with torch.no_grad():
        for context, target_class, target_data in data:
            class_pred, data_pred = model(context.unsqueeze(0))
            
            pred_class_idx = int(torch.argmax(class_pred.squeeze(0)).item())
            actual_class_idx = int(target_class.item())
            
            all_pred_class_idx.append(pred_class_idx)
            all_actual_class_idx.append(actual_class_idx)
            
            all_data_preds.append(data_pred.squeeze(0).cpu().numpy())
            all_target_datas.append(target_data.cpu().numpy())

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

    return {
        'mse': avg_targeted_mse,
        'predictions': all_data_preds,
        'targets': all_target_datas
    }

def run_single_experiment(
    train_data_wc: DataType, val_data_wc: DataType, 
    train_data_woc: DataType, val_data_woc: DataType,
    capture_learning_curves: bool = False
) -> Tuple[float, float, int, int, dict[str, list[float]], dict[str, list[float]], Dict[str, Any], Dict[str, Any]]:
    """
    AI: Runs one full training and evaluation cycle for both models and returns their MSE and best epoch.
    Optionally captures and returns learning curves.
    """
    print("\n--- Running New Experiment Trial ---")
    
    # --- Train and evaluate the model with correct masking ---
    model_with_context = TransformerPredictor(INPUT_SIZE, D_MODEL, NHEAD, NUM_LAYERS, NUM_EVENT_TYPES, DATA_VECTOR_SIZE)
    print("Training model WITH CONTEXT (Correct Masking)...")
    epochs_correct, curves_correct = train_model(model_with_context, train_data_wc, val_data_wc, capture_learning_curves=capture_learning_curves)
    print("Evaluating model WITH CONTEXT...")
    eval_results_correct = evaluate_model(model_with_context, val_data_wc)
    mse_with_context = float('nan' if eval_results_correct['mse'] is None else eval_results_correct['mse'])

    # --- Train and evaluate the model with incorrect coercion ---
    model_without_backprop = TransformerPredictor(INPUT_SIZE, D_MODEL, NHEAD, NUM_LAYERS, NUM_EVENT_TYPES, DATA_VECTOR_SIZE)
    print("\nTraining model WITH COERCED ZEROS (Incorrect)...")
    epochs_incorrect, curves_incorrect = train_model(model_without_backprop, train_data_woc, val_data_woc, capture_learning_curves=capture_learning_curves)
    print("Evaluating model WITH COERCED ZEROS...")
    eval_results_incorrect = evaluate_model(model_without_backprop, val_data_woc)
    mse_without_backprop = float('nan' if eval_results_incorrect['mse'] is None else eval_results_incorrect['mse'])
    
    print(f"Trial Results -> Correct MSE: {mse_with_context:.4f} (at epoch {epochs_correct}), Incorrect MSE: {mse_without_backprop:.4f} (at epoch {epochs_incorrect})")
    
    return mse_with_context, mse_without_backprop, epochs_correct, epochs_incorrect, curves_correct, curves_incorrect, eval_results_correct, eval_results_incorrect

def main():
    print("Preparing data...")
    data_with_context = prepare_data(TRAINING_DATA_WITH_CONTEXT_VECTORISED)
    data_with_context_coercion = prepare_data(TRAINING_DATA_WITH_CONTEXT_VECTORISED_COERCION)

    num_trials : int = 10  # AI: Number of times to run the experiment
    results_correct : list[float] = []
    results_incorrect : list[float] = []
    epochs_correct_list: list[int] = []
    epochs_incorrect_list: list[int] = []
    all_curves_correct: list[dict[str, list[float]]] = []
    all_curves_incorrect: list[dict[str, list[float]]] = []
    all_evals_correct: list[Dict[str, Any]] = []
    all_evals_incorrect: list[Dict[str, Any]] = []

    print(f"\nRunning {num_trials} trials...")
    for i in range(num_trials):
        print(f"\n--- Trial {i+1}/{num_trials} ---")
        
        # AI: Re-seed and re-split the data for each trial to introduce variance
        trial_seed = 42 + i
        torch.manual_seed(trial_seed)
        np.random.seed(trial_seed)
        random.seed(trial_seed)
        
        train_data_wc, val_data_wc = train_test_split(data_with_context, test_size=0.2, random_state=trial_seed, shuffle=True)
        train_data_woc, val_data_woc = train_test_split(data_with_context_coercion, test_size=0.2, random_state=trial_seed, shuffle=True)
        print(f"Data re-split for Trial {i+1} with seed {trial_seed}. Train size: {len(train_data_wc)}, Val size: {len(val_data_wc)}")

        capture_curves = True
        (
            mse_correct, mse_incorrect, 
            epochs_correct, epochs_incorrect, 
            curves_correct, curves_incorrect, 
            eval_correct, eval_incorrect
        ) = run_single_experiment(
            train_data_wc, val_data_wc, train_data_woc, val_data_woc, capture_learning_curves=capture_curves
        )
        if not (np.isnan(mse_correct) or np.isnan(mse_incorrect)):
            results_correct.append(mse_correct)
            results_incorrect.append(mse_incorrect)
            epochs_correct_list.append(epochs_correct)
            epochs_incorrect_list.append(epochs_incorrect)
            if capture_curves:
                all_curves_correct.append(curves_correct)
                all_curves_incorrect.append(curves_incorrect)
            all_evals_correct.append(eval_correct)
            all_evals_incorrect.append(eval_incorrect)
    
    # --- Statistical Analysis ---
    mean_correct = float(np.mean(results_correct)) if results_correct else float('nan')
    std_correct = float(np.std(results_correct)) if results_correct else float('nan')
    mean_incorrect = float(np.mean(results_incorrect)) if results_incorrect else float('nan')
    std_incorrect = float(np.std(results_incorrect)) if results_incorrect else float('nan')

    print(f"Correct Model (Masking) MSE -> Mean: {mean_correct:.4f}, Std: {std_correct:.4f}")
    print(f"Incorrect Model (Coercion) MSE -> Mean: {mean_incorrect:.4f}, Std: {std_incorrect:.4f}")

    # --- Perform t-test ---
    if len(results_correct) > 1 and len(results_incorrect) > 1:
        t_stat, p_value = ttest_ind(results_correct, results_incorrect) # type: ignore
        print(f"\nIndependent t-test results: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
        if p_value < 0.05:
            print("The difference is statistically significant (p < 0.05).")
        else:
            print("The difference is not statistically significant (p >= 0.05).")
    else:
        t_stat, p_value = None, None
        print("\nCould not perform t-test: requires at least 2 data points for each model.")

    # AI: Save results to a .npy file
    results_data = {
        "num_trials": num_trials,
        "correct_model": {
            "mse_results": results_correct,
            "epochs": epochs_correct_list,
            "mean_mse": mean_correct,
            "std_mse": std_correct,
            "all_learning_curves": all_curves_correct,
            "all_evaluations": all_evals_correct
        },
        "incorrect_model": {
            "mse_results": results_incorrect,
            "epochs": epochs_incorrect_list,
            "mean_mse": mean_incorrect,
            "std_mse": std_incorrect,
            "all_learning_curves": all_curves_incorrect,
            "all_evaluations": all_evals_incorrect
        },
        "t_test": {
            "statistic": t_stat,
            "p_value": p_value
        }
    }

    results_path = os.path.join(os.path.dirname(__file__), 'results.npy')
    
    np.save(results_path, results_data, allow_pickle=True)
    
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()
