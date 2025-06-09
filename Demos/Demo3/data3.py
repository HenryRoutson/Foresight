# Demos/Demo3/data3.py

import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction import DictVectorizer
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

# AI: The regression vector is now dynamically sized based on event contexts.




# TODO the model is not yet using this
class Event(TypedDict):
    id: event_id_t
    context : dict[str, Any]


# AI: Vectorizer registry for event contexts, similar to Demo2
event_id_to_vectorizer: dict[event_id_t, Optional[DictVectorizer]] = {}


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

def generate_training_samples(price_series: np.ndarray[Any, Any]) -> List[Tuple[List[float], Event]]:
    """
    AI: Generates training samples using the Event TypedDict.
    - 'BUY' events have context for target_price and stop_loss.
    - 'SELL' events have context for target_price.
    - 'HOLD' events have no context.
    """
    samples: List[Tuple[List[float], Event]] = []
    for i in range(CONTEXT_WINDOW, len(price_series) - 1):
        context = list(price_series[i-CONTEXT_WINDOW:i])
        current_price = price_series[i]
        
        # AI: Use moving averages to define clearer BUY/SELL/HOLD signals
        avg_long = np.mean(price_series[i-CONTEXT_WINDOW:i])
        avg_short = np.mean(price_series[i-3:i])

        event: Event
        if current_price < avg_short and avg_short < avg_long:
            event = {
                "id": event_id_t("BUY"),
                "context": {"target_price": current_price * 1.08, "stop_loss": current_price * 0.92}
            }
        elif current_price > avg_short and avg_short > avg_long:
            event = {
                "id": event_id_t("SELL"),
                "context": {"target_price": current_price * 1.10}
            }
        else:
            event = {
                "id": event_id_t("HOLD"),
                "context": {}
            }
            
        samples.append((context, event))
    return samples

# --- Vectorization ---

def _initialise_vectorizers(raw_data: List[Tuple[List[float], Event]]):
    """
    AI: Creates and fits DictVectorizers for each event type based on the data.
    """
    # AI: Collect all unique contexts for each event ID
    contexts_by_id: dict[event_id_t, list[dict[str, Any]]] = {eid: [] for eid in events_id_list}
    for _, event in raw_data:
        if event['context']:
            contexts_by_id[event['id']].append(event['context'])

    # AI: Fit a vectorizer for each event ID that has context
    for event_id, contexts in contexts_by_id.items():
        if not contexts:
            event_id_to_vectorizer[event_id] = None
        else:
            vectorizer = DictVectorizer(sparse=False)
            vectorizer.fit(contexts)
            event_id_to_vectorizer[event_id] = vectorizer

def vectorize_data(raw_data: List[Tuple[List[float], Event]], coerce_none_to_zero: bool) -> List[List[Any]]:
    """
    AI: Vectorizes data similarly to Demo2. Each sequence is [input_vector_t-1, input_vector_t, target_vector_for_t].
    The target vector is structured as [OHE, buy_data, sell_data, hold_data], with non-applicable data as None.
    """
    event_ids = [event_to_id[sample[1]['id']] for sample in raw_data]
    one_hot_encoder = OneHotEncoder(categories=[range(NUM_EVENT_TYPES)], sparse_output=False).fit(np.array(event_ids).reshape(-1, 1)) # type: ignore
    
    contexts = [sample[0] for sample in raw_data]
    context_scaler = StandardScaler().fit(contexts) # type: ignore

    vectorized_sequences: list[Any] = []
    
    # AI: Start from index 1 to ensure we always have a previous step (t-1) for context
    for i in range(1, len(raw_data)):
        price_context_t, event_t = raw_data[i]
        price_context_tm1, event_tm1 = raw_data[i-1]

        # --- Input vector for step 1 of the sequence (Time t-1) ---
        scaled_context_tm1 : np.ndarray[Any, Any] = context_scaler.transform(np.array(price_context_tm1).reshape(1, -1)).flatten() # type: ignore
        event_id_tm1 = event_to_id[event_tm1['id']]
        event_one_hot_tm1 : np.ndarray[Any, Any] = one_hot_encoder.transform(np.array([[event_id_tm1]])).flatten() # type: ignore
        input_vector_tm1 : np.ndarray[Any, Any] = np.concatenate([scaled_context_tm1, event_one_hot_tm1]) # type: ignore
        
        # --- Input vector for step 2 of the sequence (Time t) ---
        scaled_context_t : np.ndarray[Any, Any] = context_scaler.transform(np.array(price_context_t).reshape(1, -1)).flatten() # type: ignore
        event_id_t = event_to_id[event_t['id']]
        event_one_hot_t : np.ndarray[Any, Any] = one_hot_encoder.transform(np.array([[event_id_t]])).flatten() # type: ignore
        input_vector_t : np.ndarray[Any, Any] = np.concatenate([scaled_context_t, event_one_hot_t]) # type: ignore

        # --- Target Vector (for time t) ---
        target_class_one_hot : np.ndarray[Any, Any] = event_one_hot_t.copy() # type: ignore
        
        target_data_vectors: list[np.ndarray[Any, Any]] = [target_class_one_hot]
        
        for event_id_from_list in events_id_list:
            vectorizer = event_id_to_vectorizer.get(event_id_from_list)
            if vectorizer is None:
                continue

            num_features = len(vectorizer.get_feature_names_out())

            if event_t['id'] == event_id_from_list:
                # This is the event that occurred. Vectorize its context.
                if event_t['context']:
                    vectorized_context = vectorizer.transform([event_t['context']]).flatten()
                    target_data_vectors.append(vectorized_context)
                else:
                    # Should not happen if vectorizer exists, but for safety
                    target_data_vectors.append(np.full(num_features, 0.0))
            else:
                # This event did not occur. Fill with None.
                target_data_vectors.append(np.full(num_features, None, dtype=object))
        
        target_vector = np.concatenate(target_data_vectors)

        if coerce_none_to_zero:
            # AI: This is where the data for the "bad" model is created.
            # AI: For 'HOLD' events, this forces the target to be [0.0, 0.0]
            # AI: For 'SELL' events, this forces the target to be [price, 0.0]
            target_vector[target_vector == None] = 0.0 # type: ignore

        
        sequence : list[Any] = [input_vector_tm1, input_vector_t, target_vector]
        vectorized_sequences.append(sequence)
        
    return vectorized_sequences

# --- Main Data Generation and Export ---
price_series = generate_price_series(PRICE_SERIES_LENGTH)
raw_training_data = generate_training_samples(price_series)

# AI: Must initialize vectorizers before vectorizing data so helper functions have access to them.
_initialise_vectorizers(raw_training_data)

TRAINING_DATA_WITH_CONTEXT_VECTORISED = vectorize_data(raw_training_data, coerce_none_to_zero=False)
TRAINING_DATA_WITH_CONTEXT_VECTORISED_COERCION = vectorize_data(raw_training_data, coerce_none_to_zero=True)

# --- Helpers for model3.py ---
def get_vectorizer_output_length(event_id: event_id_t) -> int:
    """Returns the number of regression parameters for a given event type."""
    vectorizer = event_id_to_vectorizer.get(event_id)
    if vectorizer:
        return len(vectorizer.get_feature_names_out())
    return 0
    
def get_vector_sizes() -> Tuple[int, int]:
    """Returns the size of the input and data regression vectors."""
    input_size = CONTEXT_WINDOW + NUM_EVENT_TYPES
    data_vector_size = sum(get_vectorizer_output_length(eid) for eid in events_id_list)
    return input_size, data_vector_size

# --- Verification Script ---
if __name__ == "__main__":
    print("--- Demo 3 Data Generation ---")
    print(f"Generated {len(TRAINING_DATA_WITH_CONTEXT_VECTORISED)} training sequences.")
    
    input_size, data_vec_size = get_vector_sizes()
    print(f"Input vector size: {input_size}")
    print(f"Data regression vector size: {data_vec_size}")
    for eid in events_id_list:
        print(f"  - {eid}: {get_vectorizer_output_length(eid)} params")

    # AI: Helper to find and print a sample of a given event type
    def print_sample_by_type(event_name: event_id_t):
        for i, (price_context, event) in enumerate(raw_training_data):
            if event['id'] == event_name:
                print(f"\nSample '{event_name}' event:")
                print(f"  Raw context data: {event['context']}")
                # AI: Note: Indexing vectorized data at [i-1] because the loop starts at 1
                if i > 0:
                    correct_vec = TRAINING_DATA_WITH_CONTEXT_VECTORISED[i-1][2][NUM_EVENT_TYPES:]
                    coerced_vec = TRAINING_DATA_WITH_CONTEXT_VECTORISED_COERCION[i-1][2][NUM_EVENT_TYPES:]
                    print(f"  Vectorized (Correct): {correct_vec}")
                    print(f"  Vectorized (Coerced): {coerced_vec}")
                return
    
    print_sample_by_type(event_id_t('BUY'))
    print_sample_by_type(event_id_t('SELL'))
    print_sample_by_type(event_id_t('HOLD'))
