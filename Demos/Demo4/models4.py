# AI: Demos/Demo4/models4.py
# This file adapts the training and evaluation logic from Demo3 to the 'Conflicting Game Actions' dataset.
# The goal is to run a comparative experiment showing the performance difference between
# a model trained with masked loss and one trained with zero-coerced data.

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
from scipy.stats import ttest_ind

# AI: Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# AI: Suppress print from data4 on import
original_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
try:
    from Demos.Demo4.data4 import (
        TRAINING_DATA_WITH_CONTEXT_VECTORISED,
        TRAINING_DATA_WITH_CONTEXT_VECTORISED_COERCION,
        events_id_list,
        get_vectorizer_output_length,
        get_vector_sizes
    )
except (ImportError, ModuleNotFoundError):
    from data4 import (
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
D_MODEL = 64
NHEAD = 8
NUM_LAYERS = 3
EPOCHS = 1500
LEARNING_RATE = 0.0005
PATIENCE = 150

def prepare_data(vectorised_data: List[List[Any]]) -> DataType:
    """
    AI: Prepares data, splitting the target into classification and regression parts.
    Input `None` values are converted to 0. Target `None` values are converted to NaN for masking.
    This logic is identical to previous demos.
    """
    processed_data: DataType = []
    for sequence in vectorised_data:
        context_part = np.array(sequence[:2], dtype=object)
        target_part = np.array(sequence[2], dtype=object)

        context_np = context_part.copy()
        context_np[context_np == None] = 0.0
        context_tensor = torch.tensor(context_np.astype(np.float32))

        target_np = target_part.copy()
        target_np[target_np == None] = np.nan
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
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=d_model*4, dropout=0.1, activation='gelu')
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
        
    return class_loss + 1.0 * data_loss

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
    patience: int, capture_learning_curves: bool = False
) -> Tuple[int, Dict[str, List[float]]]:
    """
    AI: Trains the model, returning the best epoch and learning curves.
    """
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    learning_curves: Dict[str, List[float]] = {"epochs": [], "train_loss": [], "val_loss": [], "train_mse": [], "val_mse": []}

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
        train_loss = hybrid_loss(class_pred, data_pred, train_target_classes, train_target_datas)

        if not torch.isnan(train_loss):
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
            break
            
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
    all_pred_class_idx, all_actual_class_idx = [], []
    targeted_mses: List[float] = []
    all_data_preds, all_target_datas = [], []

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
                    targeted_mses.append(mse)

    labels = sorted(list(set(all_actual_class_idx)))
    label_names = [events_id_list[i].replace("_", " ").title() for i in labels]
    cm = confusion_matrix(all_actual_class_idx, all_pred_class_idx, labels=labels)
    acc = accuracy_score(all_actual_class_idx, all_pred_class_idx)
    
    print(f"\nLabels: {label_names}")
    print("Confusion Matrix:")
    header = "       " + " ".join([f"{label[:5]:>5}" for label in label_names])
    print(header)
    print("     Pred↓ True→")
    for i, label_true in enumerate(label_names):
        row_str = f"{label_true[:5]:>5} |"
        for val in cm[i]:
            row_str += f"{val:5} "
        print(row_str)
    
    print(f"\nAccuracy: {acc:.4f}")
    
    avg_targeted_mse = np.mean(targeted_mses) if targeted_mses else float('nan')
    print(f"Average Targeted MSE (on relevant data only): {avg_targeted_mse:.4f}")

    return {
        'mse': avg_targeted_mse,
        'predictions': all_data_preds,
        'targets': all_target_datas,
        'accuracy': acc,
        'confusion_matrix': cm,
        'labels': label_names
    }

def run_single_experiment(
    train_wc: DataType, val_wc: DataType, train_woc: DataType, val_woc: DataType,
    capture_curves: bool = False
) -> Tuple[float, float, int, int, Dict, Dict, Dict, Dict]:
    print("\n--- Running New Experiment Trial ---")
    
    model_correct = TransformerPredictor(INPUT_SIZE, D_MODEL, NHEAD, NUM_LAYERS, NUM_EVENT_TYPES, DATA_VECTOR_SIZE)
    print("\nTraining CORRECT model (Masking)...")
    epochs_c, curves_c = train_model(model_correct, train_wc, val_wc, PATIENCE, capture_curves)
    print("\nEvaluating CORRECT model...")
    eval_c = evaluate_model(model_correct, val_wc)

    model_incorrect = TransformerPredictor(INPUT_SIZE, D_MODEL, NHEAD, NUM_LAYERS, NUM_EVENT_TYPES, DATA_VECTOR_SIZE)
    print("\nTraining INCORRECT model (Coercion)...")
    epochs_ic, curves_ic = train_model(model_incorrect, train_woc, val_woc, PATIENCE, capture_curves)
    print("\nEvaluating INCORRECT model...")
    eval_ic = evaluate_model(model_incorrect, val_woc)
    
    print(f"\nTrial Results -> Correct MSE: {eval_c['mse']:.4f}, Incorrect MSE: {eval_ic['mse']:.4f}")
    
    return eval_c['mse'], eval_ic['mse'], epochs_c, epochs_ic, curves_c, curves_ic, eval_c, eval_ic

def main():
    print("Preparing data...")
    data_with_context = prepare_data(TRAINING_DATA_WITH_CONTEXT_VECTORISED)
    data_coerced = prepare_data(TRAINING_DATA_WITH_CONTEXT_VECTORISED_COERCION)

    num_trials = 10
    results: Dict[str, List] = {
        'correct_mse': [], 'incorrect_mse': [], 'correct_epochs': [], 'incorrect_epochs': [],
        'correct_curves': [], 'incorrect_curves': [], 'correct_evals': [], 'incorrect_evals': []
    }

    for i in range(num_trials):
        print(f"\n{'='*20} TRIAL {i+1}/{num_trials} {'='*20}")
        trial_seed = 42 + i
        torch.manual_seed(trial_seed)
        np.random.seed(trial_seed)
        random.seed(trial_seed)
        
        train_wc, val_wc = train_test_split(data_with_context, test_size=0.25, random_state=trial_seed)
        train_woc, val_woc = train_test_split(data_coerced, test_size=0.25, random_state=trial_seed)

        mse_c, mse_ic, ep_c, ep_ic, c_c, c_ic, ev_c, ev_ic = run_single_experiment(
            train_wc, val_wc, train_woc, val_woc, capture_curves=True
        )
        if not (np.isnan(mse_c) or np.isnan(mse_ic)):
            results['correct_mse'].append(mse_c)
            results['incorrect_mse'].append(mse_ic)
            results['correct_epochs'].append(ep_c)
            results['incorrect_epochs'].append(ep_ic)
            results['correct_curves'].append(c_c)
            results['incorrect_curves'].append(c_ic)
            results['correct_evals'].append(ev_c)
            results['incorrect_evals'].append(ev_ic)
    
    mean_c = np.mean(results['correct_mse'])
    std_c = np.std(results['correct_mse'])
    mean_ic = np.mean(results['incorrect_mse'])
    std_ic = np.std(results['incorrect_mse'])

    print(f"\n\n{'='*20} FINAL RESULTS {'='*20}")
    print(f"Correct Model (Masking)   -> MSE: {mean_c:.4f} ± {std_c:.4f}")
    print(f"Incorrect Model (Coercion) -> MSE: {mean_ic:.4f} ± {std_ic:.4f}")

    t_stat, p_value = ttest_ind(results['correct_mse'], results['incorrect_mse'])
    print(f"\nT-test: t-statistic = {t_stat:.4f}, p-value = {p_value:.6f}")
    if p_value < 0.05 and mean_c < mean_ic:
        print("SUCCESS: The difference is statistically significant, and the correct model performed better.")
    else:
        print("FAILURE: The difference is not statistically significant or the incorrect model performed better.")

    # Save results
    results_path = os.path.join(os.path.dirname(__file__), 'results.npy')
    saved_data = {
        "num_trials": num_trials,
        "correct_model": {
            "mse_results": results['correct_mse'], "epochs": results['correct_epochs'],
            "mean_mse": mean_c, "std_mse": std_c,
            "all_learning_curves": results['correct_curves'], "all_evaluations": results['correct_evals'],
            "final_metrics": results['correct_evals'][-1] if results['correct_evals'] else {}
        },
        "incorrect_model": {
            "mse_results": results['incorrect_mse'], "epochs": results['incorrect_epochs'],
            "mean_mse": mean_ic, "std_mse": std_ic,
            "all_learning_curves": results['incorrect_curves'], "all_evaluations": results['incorrect_evals'],
            "final_metrics": results['incorrect_evals'][-1] if results['incorrect_evals'] else {}
        },
        "t_test": {"statistic": t_stat, "p_value": p_value}
    }
    np.save(results_path, saved_data, allow_pickle=True)
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()
