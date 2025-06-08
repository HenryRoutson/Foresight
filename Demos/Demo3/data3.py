# Demos/Demo3/data3.py

import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import List, Tuple, Any, Optional, NewType, TypedDict

# --- Configuration ---
# AI: Increased sample count and context window to create a more complex task
NUM_SAMPLES = 1000
CONTEXT_WINDOW = 15
PRICE_SERIES_LENGTH = NUM_SAMPLES + CONTEXT_WINDOW + 2

# Event IDs

event_id_t = NewType("event_id_t", str)

events_id_list : list[event_id_t] = [event_id_t("BUY"), event_id_t("SELL"), event_id_t("HOLD")]
event_to_id : dict[event_id_t, int] = {event: i for i, event in enumerate(events_id_list)}
NUM_EVENT_TYPES = len(events_id_list)

# AI: The regression vector must be large enough for the 'BUY' event's data (target_price, stop_loss)
MAX_DATA_VECTOR_SIZE = 2



# TODO the model is not yet using this
class Event(TypedDict):
    id: event_id_t
    context : dict[str, Any]




# --- Data Generation ---

def generate_price_series(length: int) -> np.ndarray[Any, Any]:
    """
    AI: Generates a more complex synthetic price series to provide richer context.
    """
    np.random.seed(42)
    base_price = 150.0
    noise = np.random.randn(length).cumsum() * 0.5
    trend = np.linspace(0, 50, length)
    # AI: Multiple sine waves to simulate complex market seasonality
    seasonality1 = 20 * np.sin(np.linspace(0, 2 * np.pi * 10, length))
    seasonality2 = 8 * np.sin(np.linspace(0, 2 * np.pi * 25, length))
    return base_price + noise + trend + seasonality1 + seasonality2

def generate_training_samples(price_series: np.ndarray[Any, Any]) -> List[Tuple[List[float], str, Tuple[Optional[float], Optional[float]]]]:
    """
    AI: Generates training samples based on rules that create conflicting regression targets.
    - 'BUY'/'SELL' events have high-value regression targets (~150-200).
    - 'HOLD' events have no regression target (None).
    This sets up the conflict for the incorrectly configured model.
    """
    samples: List[Tuple[List[float], str, Tuple[Optional[float], Optional[float]]]] = []
    for i in range(CONTEXT_WINDOW, len(price_series) - 1):
        context = list(price_series[i-CONTEXT_WINDOW:i])
        current_price = price_series[i]
        
        # AI: Use moving averages to define clearer BUY/SELL/HOLD signals
        avg_long = np.mean(price_series[i-CONTEXT_WINDOW:i])
        avg_short = np.mean(price_series[i-3:i])

        event_data: Tuple[Optional[float], Optional[float]]
        if current_price < avg_short and avg_short < avg_long:
            event_type = "BUY"
            # AI: High-value regression targets
            event_data = (current_price * 1.08, current_price * 0.92)
        elif current_price > avg_short and avg_short > avg_long:
            event_type = "SELL"
            # AI: A high-value target and a None
            event_data = (current_price * 1.10, None)
        else:
            event_type = "HOLD"
            # AI: No regression target
            event_data = (None, None)
            
        samples.append((context, event_type, event_data))
    return samples

# --- Vectorization ---

def vectorize_data(raw_data: List[Tuple[List[float], str, Tuple[Optional[float], Optional[float]]]], coerce_none_to_zero: bool) -> List[List[Any]]:
    """
    AI: Vectorizes the raw data into sequences for the Transformer.
    Each sequence is [input_vector_t-1, input_vector_t, target_vector_for_t]
    """
    event_ids = [event_to_id[sample[1]] for sample in raw_data]
    one_hot_encoder = OneHotEncoder(categories=[range(NUM_EVENT_TYPES)], sparse_output=False).fit(np.array(event_ids).reshape(-1, 1)) # type: ignore
    
    contexts = [sample[0] for sample in raw_data]
    context_scaler = StandardScaler().fit(contexts) # type: ignore

    vectorized_sequences: list[Any] = []
    
    # AI: Start from index 1 to ensure we always have a previous step (t-1) for context
    for i in range(1, len(raw_data)):
        context_t, event_type_t, event_data_t = raw_data[i]
        context_tm1, event_type_tm1, _ = raw_data[i-1]

        # --- Input vector for step 1 of the sequence (Time t-1) ---
        scaled_context_tm1 : np.ndarray[Any, Any] = context_scaler.transform(np.array(context_tm1).reshape(1, -1)).flatten() # type: ignore
        event_id_tm1 = event_to_id[event_type_tm1]
        event_one_hot_tm1 : np.ndarray[Any, Any] = one_hot_encoder.transform(np.array([[event_id_tm1]])).flatten() # type: ignore
        input_vector_tm1 : np.ndarray[Any, Any] = np.concatenate([scaled_context_tm1, event_one_hot_tm1]) # type: ignore
        
        # --- Input vector for step 2 of the sequence (Time t) ---
        scaled_context_t : np.ndarray[Any, Any] = context_scaler.transform(np.array(context_t).reshape(1, -1)).flatten() # type: ignore
        event_id_t = event_to_id[event_type_t]
        event_one_hot_t : np.ndarray[Any, Any] = one_hot_encoder.transform(np.array([[event_id_t]])).flatten() # type: ignore
        input_vector_t : np.ndarray[Any, Any] = np.concatenate([scaled_context_t, event_one_hot_t]) # type: ignore

        # --- Target Vector (for time t) ---
        target_class_one_hot : np.ndarray[Any, Any] = event_one_hot_t # type: ignore
        target_data_vector : np.ndarray[Any, Any] = np.full(MAX_DATA_VECTOR_SIZE, None, dtype=object)
        if event_data_t is not None: # type: ignore
            for j, val in enumerate(event_data_t):
                target_data_vector[j] = val

        if coerce_none_to_zero:
            # AI: This is where the data for the "bad" model is created.
            # AI: For 'HOLD' events, this forces the target to be [0.0, 0.0]
            # AI: For 'SELL' events, this forces the target to be [price, 0.0]
            target_data_vector[target_data_vector == None] = 0.0 # type: ignore

        target_vector : np.ndarray[Any, Any] = np.concatenate([target_class_one_hot, target_data_vector]) # type: ignore
        
        sequence : list[Any] = [input_vector_tm1, input_vector_t, target_vector]
        vectorized_sequences.append(sequence)
        
    return vectorized_sequences

# --- Main Data Generation and Export ---
price_series = generate_price_series(PRICE_SERIES_LENGTH)
raw_training_data = generate_training_samples(price_series)

TRAINING_DATA_WITH_CONTEXT_VECTORISED = vectorize_data(raw_training_data, coerce_none_to_zero=False)
TRAINING_DATA_WITH_CONTEXT_VECTORISED_COERCION = vectorize_data(raw_training_data, coerce_none_to_zero=True)

# --- Helpers for model3.py ---
def get_vectorizer_output_length(event_id: str) -> int:
    """Returns the number of regression parameters for a given event type."""
    if event_id == "BUY": return 2
    if event_id == "SELL": return 1
    if event_id == "HOLD": return 0
    return 0
    
def get_vector_sizes() -> Tuple[int, int]:
    """Returns the size of the input and data regression vectors."""
    input_size = CONTEXT_WINDOW + NUM_EVENT_TYPES
    data_vector_size = MAX_DATA_VECTOR_SIZE
    return input_size, data_vector_size

# --- Verification Script ---
if __name__ == "__main__":
    print("--- Demo 3 Data Generation ---")
    print(f"Generated {len(TRAINING_DATA_WITH_CONTEXT_VECTORISED)} training sequences.")
    
    # AI: Helper to find and print a sample of a given event type
    def print_sample_by_type(event_name : str):
        for i in range(len(raw_training_data)):
            if raw_training_data[i][1] == event_name:
                print(f"Sample '{event_name}' event:")
                print(f"  Raw regression data: {raw_training_data[i][2]}")
                # AI: Note: Indexing vectorized data at [i-1] because the loop starts at 1
                correct_vec = TRAINING_DATA_WITH_CONTEXT_VECTORISED[i-1][2][NUM_EVENT_TYPES:]
                coerced_vec = TRAINING_DATA_WITH_CONTEXT_VECTORISED_COERCION[i-1][2][NUM_EVENT_TYPES:]
                print(f"  Vectorized (Correct): {correct_vec}")
                print(f"  Vectorized (Coerced): {coerced_vec}")
                return
    
    print_sample_by_type('BUY')
    print_sample_by_type('SELL')
    print_sample_by_type('HOLD')
