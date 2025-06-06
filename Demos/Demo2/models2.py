# AI: models2.py (RNN Implementation)
# This file implements an RNN (LSTM) to predict sequence outcomes using vectorized data.
# It replaces the previous statistical model to handle complex, non-string-convertible data.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score
from typing import List, Tuple, Dict, Any, Optional
import sys, os

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
        get_vectorizer_output_length,
        Event
    )
except (ImportError, ModuleNotFoundError):
    from Demos.Demo2.data2 import (
        TRAINING_DATA_WITH_CONTEXT_VECTORISED,
        TRAINING_DATA_WITHOUT_CONTEXT_VECTORISED,
        events_id_list,
        event_id_to_vectorizer,
        get_vectorizer_output_length,
        Event
    )
finally:
    # AI: Restore stdout right after the import.
    sys.stdout.close()
    sys.stdout = original_stdout

# AI: Define global constants for the model and training
INPUT_SIZE = len(TRAINING_DATA_WITH_CONTEXT_VECTORISED[0][0])
HIDDEN_SIZE = 32 # AI: A reasonable hidden size for this complexity
OUTPUT_SIZE = INPUT_SIZE # AI: The model predicts a vector of the same size as the input
EPOCHS = 200 # AI: Sufficient epochs for convergence on this small dataset
LEARNING_RATE = 0.01

def prepare_data(vectorised_data: List[List[np.ndarray[Any, Any]]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    AI: Prepares the raw numpy data for the PyTorch model.
    This involves converting None to NaN, creating tensors, and splitting into (context, target) pairs.
    """
    processed_data : list[Tuple[torch.Tensor, torch.Tensor]] = []
    for sequence in vectorised_data:
        # AI: Convert the list of numpy arrays into a single numpy matrix.
        list_of_lists = [list(item) for item in sequence]
        sequence_np = np.array(list_of_lists, dtype=object)

        # AI: Replace python None with numpy.nan for numeric processing
        for i in range(sequence_np.shape[0]):
            for j in range(sequence_np.shape[1]):
                if sequence_np[i, j] is None:
                    sequence_np[i, j] = np.nan

        # AI: Convert to a float tensor
        sequence_tensor = torch.tensor(sequence_np.astype(np.float32))

        # AI: The context is the first two events, the target is the third.
        context = sequence_tensor[:2]
        target = sequence_tensor[2]
        processed_data.append((context, target))
    return processed_data

class RnnPredictor(nn.Module):
    """
    AI: An LSTM-based RNN to predict the next event vector in a sequence.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(RnnPredictor, self).__init__()
        # AI: LSTM layer for processing the sequence. batch_first=True expects
        # input tensors of shape (batch_size, sequence_length, input_size).
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # AI: A linear layer to map the LSTM's output to the desired prediction vector size.
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # AI: x shape: (batch_size, sequence_length=2, input_size)
        lstm_out, _ = self.lstm(x)
        # AI: We only need the output from the last time step for our prediction.
        last_hidden_state = lstm_out[:, -1, :]
        prediction = self.linear(last_hidden_state)
        return prediction

def masked_mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    AI: Custom loss function that calculates Mean Squared Error only on the non-NaN
    elements of the target vector. This is the core of the "Masked Loss Function".
    """
    mask = ~torch.isnan(target)
    # AI: If the mask is all False (e.g., target is all NaN), loss is 0.
    if not mask.any():
        return torch.tensor(0.0, device=prediction.device)
    
    # AI: Apply the mask to both prediction and target before calculating loss.
    prediction_masked = torch.masked_select(prediction, mask)
    target_masked = torch.masked_select(target, mask)
    
    return nn.functional.mse_loss(prediction_masked, target_masked)

def train_model(model: nn.Module, data: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    AI: The main training loop for the RNN model.
    """
    optimizer : optim.Optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train() # AI: Set the model to training mode

    for epoch in range(EPOCHS):
        total_loss = 0
        for context, target in data:
            optimizer.zero_grad()
            
            # AI: Add a batch dimension for the LSTM layer
            context = context.unsqueeze(0)
            
            prediction = model(context).squeeze(0)
            
            loss : torch.Tensor = masked_mse_loss(prediction, target)

            # AI: The loss can be NaN if prediction becomes NaN. Skip update if so.
            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        # if (epoch + 1) % (EPOCHS // 10) == 0:
        #     print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(data):.4f}")

def vector_to_label(vector: np.ndarray[Any, Any]) -> str:
    """
    AI: Converts a prediction or target vector back to a simple string label ('A', 'T', 'F', 'B')
    for use in the confusion matrix.
    """
    num_event_types = len(events_id_list)
    event_probs = vector[:num_event_types]
    predicted_idx = np.argmax(event_probs)
    event_id = events_id_list[predicted_idx]

    if event_id == "A":
        return "A"
    if event_id == "B_without_context":
        return "B"
    if event_id == "B_with_context":
        # AI: For 'B_with_context', we need to check the predicted boolean value
        # to distinguish between 'T' (True) and 'F' (False).
        vectorizer = event_id_to_vectorizer.get(event_id)
        if not vectorizer: return "B" # Fallback

        feature_names = list(vectorizer.get_feature_names_out())
        
        try:
            bool_data_idx = feature_names.index("bool_data")
        except ValueError:
            return "B" # Fallback if feature is not found

        # AI: Calculate the start index for this event's data in the vector
        data_start_idx = num_event_types
        for i in range(predicted_idx):
            data_start_idx += get_vectorizer_output_length(events_id_list[i])
        
        # AI: Check the predicted value. A threshold of 0.5 is used to classify as True/False.
        bool_value = vector[data_start_idx + bool_data_idx]
        return "T" if bool_value > 0.5 else "F"

    return "Unknown" # Fallback

def evaluate_model(model: nn.Module, data: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    AI: Evaluates the model, generates predictions, converts them to labels,
    and prints the confusion matrix and accuracy.
    """
    model.eval() # AI: Set the model to evaluation mode
    actual_labels : list[str] = []
    predicted_labels : list[str] = []

    with torch.no_grad():
        for context, target in data:
            prediction_vec = model(context.unsqueeze(0)).squeeze(0)
            
            # AI: Convert tensors to numpy for processing
            prediction_np: np.ndarray[Any, Any] = prediction_vec.cpu().numpy()
            target_np: np.ndarray[Any, Any] = target.cpu().numpy()

            actual_labels.append(vector_to_label(target_np))
            predicted_labels.append(vector_to_label(prediction_np))
    
    # AI: Use the set of actual labels to ensure the confusion matrix is ordered correctly.
    labels : list[str] = sorted(list(set(actual_labels)))
    
    cm: np.ndarray[Any, Any] = confusion_matrix(actual_labels, predicted_labels, labels=labels)
    acc: float = accuracy_score(actual_labels, predicted_labels)
    
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

    # --- WITH CONTEXT ---
    print("\nTraining model WITH context...")
    model_with_context = RnnPredictor(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    train_model(model_with_context, data_with_context)
    print("Evaluating model WITH context...")
    evaluate_model(model_with_context, data_with_context)
    print("-" * 40)

    # --- WITHOUT CONTEXT ---
    print("\nTraining model WITHOUT context...")
    model_without_context = RnnPredictor(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    train_model(model_without_context, data_without_context)
    print("Evaluating model WITHOUT context...")
    evaluate_model(model_without_context, data_without_context)
    print("-" * 40)

if __name__ == "__main__":
    main()