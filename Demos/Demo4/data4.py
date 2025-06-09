# Demos/Demo4/data4.py
# AI: This file generates a dataset to highlight the difference between masked and non-masked loss functions.
# The scenario involves a simple game where a player can move, attack, or idle.
# The key is that a non-occurring 'ATTACK' event, when its data is naively zero-filled,
# creates a spurious correlation with the valid 'ATTACK' data for target_id=0.

import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction import DictVectorizer
from typing import List, Tuple, Any, Optional, NewType, TypedDict

# --- Configuration ---
NUM_SAMPLES = 2000
CONTEXT_WINDOW = 5  # AI: Shorter window as the state is richer
SEQUENCE_LENGTH = NUM_SAMPLES + CONTEXT_WINDOW + 2
NUM_TARGETS = 3

# --- Event Definitions ---
event_id_t = NewType("event_id_t", str)

events_id_list: list[event_id_t] = [
    event_id_t("MOVE_TO_COORDS"), 
    event_id_t("ATTACK_TARGET"), 
    event_id_t("IDLE")
]
event_to_id: dict[event_id_t, int] = {event: i for i, event in enumerate(events_id_list)}
NUM_EVENT_TYPES = len(events_id_list)

class Event(TypedDict):
    id: event_id_t
    context: dict[str, Any]

# AI: Vectorizer registry, one for each event type that has context.
event_id_to_vectorizer: dict[event_id_t, Optional[DictVectorizer]] = {}

# --- Data Generation: Game Simulation ---

def generate_game_state_series(length: int) -> np.ndarray[Any, Any]:
    """
    AI: Generates a synthetic series of game states.
    State vector: [player_x, player_y, target_0_hp, target_1_hp, ..., target_N_hp]
    """
    np.random.seed(42)
    # AI: Initialize state with player at origin and targets at full health
    initial_state = np.array([0.0, 0.0] + [100.0] * NUM_TARGETS)
    
    states = [initial_state]
    for _ in range(1, length):
        last_state = states[-1].copy()
        # Player movement: random walk
        last_state[0:2] += np.random.randn(2) * 0.8
        # Target health decay: random decay, sometimes a target gets a burst of damage
        last_state[2:] -= np.random.rand(NUM_TARGETS) * 2
        if np.random.rand() < 0.05: # 5% chance of a burst damage event
            target_to_damage = np.random.randint(0, NUM_TARGETS)
            last_state[2 + target_to_damage] -= np.random.rand() * 20
        last_state[2:] = np.clip(last_state[2:], 0, 100) # Health can't be below 0
        states.append(last_state)
        
    return np.array(states)

def generate_training_samples(game_states: np.ndarray[Any, Any]) -> List[Tuple[np.ndarray[Any, Any], Event]]:
    """
    AI: Generates training samples based on the game state. This is where the crucial logic lies.
    - If target 0's health is critical, the player should ATTACK it (target_id=0).
    - If the player is far from all targets, they should MOVE towards the center (0,0).
    - Otherwise, the player is IDLE.
    """
    samples: List[Tuple[np.ndarray[Any, Any], Event]] = []
    
    for i in range(CONTEXT_WINDOW, len(game_states) - 1):
        # AI: The input context for the model is the game state at time t.
        context_vector = game_states[i]
        player_pos = context_vector[0:2]
        targets_hp = context_vector[2:]
        
        # AI: Calculate distance to a fictional center point for the 'MOVE' logic
        distance_to_center = np.linalg.norm(player_pos - np.array([0, 0]))

        event: Event
        # --- Decision Logic ---
        # AI: CRITICAL RULE: If target 0's health is low, the ONLY correct action is to attack it.
        # This will conflict with the zero-coerced data from the MOVE action.
        if targets_hp[0] < 20 and targets_hp[0] > 0:
            event = {
                "id": event_id_t("ATTACK_TARGET"),
                "context": {"target_id": 0.0} # AI: Note: target_id is 0.0
            }
        # AI: If player is far away, they should move back to the center.
        elif distance_to_center > 15:
            event = {
                "id": event_id_t("MOVE_TO_COORDS"),
                "context": {"target_x": 0.0, "target_y": 0.0}
            }
        # AI: Default action is to do nothing.
        else:
            event = {
                "id": event_id_t("IDLE"),
                "context": {}
            }
            
        samples.append((context_vector, event))
    return samples

# --- Vectorization ---

def _initialise_vectorizers(raw_data: List[Tuple[np.ndarray[Any, Any], Event]]):
    """
    AI: Creates and fits DictVectorizers for each event type based on the data.
    This is identical in logic to Demo3.
    """
    contexts_by_id: dict[event_id_t, list[dict[str, Any]]] = {eid: [] for eid in events_id_list}
    for _, event in raw_data:
        if event['context']:
            contexts_by_id[event['id']].append(event['context'])

    for event_id, contexts in contexts_by_id.items():
        if not contexts:
            event_id_to_vectorizer[event_id] = None
        else:
            vectorizer = DictVectorizer(sparse=False)
            vectorizer.fit(contexts)
            event_id_to_vectorizer[event_id] = vectorizer
            print(f"Vectorizer for {event_id}: features = {vectorizer.get_feature_names_out()}")

def vectorize_data(raw_data: List[Tuple[np.ndarray[Any, Any], Event]], coerce_none_to_zero: bool) -> List[List[Any]]:
    """
    AI: Vectorizes data. Each sequence is [input_vector_t-1, input_vector_t, target_vector_for_t].
    The target vector is structured as [OHE, move_data, attack_data, idle_data], with non-applicable data as None or 0.
    """
    event_ids = [event_to_id[sample[1]['id']] for sample in raw_data]
    one_hot_encoder = OneHotEncoder(categories=[range(NUM_EVENT_TYPES)], sparse_output=False).fit(np.array(event_ids).reshape(-1, 1))
    
    contexts = [sample[0] for sample in raw_data]
    context_scaler = StandardScaler().fit(contexts)

    vectorized_sequences: list[Any] = []
    
    for i in range(1, len(raw_data)):
        state_context_t, event_t = raw_data[i]
        state_context_tm1, event_tm1 = raw_data[i-1]

        # --- Input vector (t-1) ---
        scaled_context_tm1 = context_scaler.transform(state_context_tm1.reshape(1, -1)).flatten()
        event_one_hot_tm1 = one_hot_encoder.transform([[event_to_id[event_tm1['id']]]]).flatten()
        input_vector_tm1 = np.concatenate([scaled_context_tm1, event_one_hot_tm1])
        
        # --- Input vector (t) ---
        scaled_context_t = context_scaler.transform(state_context_t.reshape(1, -1)).flatten()
        event_one_hot_t = one_hot_encoder.transform([[event_to_id[event_t['id']]]]).flatten()
        input_vector_t = np.concatenate([scaled_context_t, event_one_hot_t])

        # --- Target Vector (for time t) ---
        target_class_one_hot = event_one_hot_t.copy()
        
        target_data_vectors: list[np.ndarray[Any, Any]] = [target_class_one_hot]
        
        for event_id_from_list in events_id_list:
            vectorizer = event_id_to_vectorizer.get(event_id_from_list)
            if vectorizer is None:
                continue

            num_features = len(vectorizer.get_feature_names_out())

            if event_t['id'] == event_id_from_list:
                vectorized_context = vectorizer.transform([event_t['context']]).flatten()
                target_data_vectors.append(vectorized_context)
            else:
                # AI: This is where we mark data for non-occurring events as 'None'.
                target_data_vectors.append(np.full(num_features, None, dtype=object))
        
        target_vector = np.concatenate(target_data_vectors)

        if coerce_none_to_zero:
            # AI: THIS IS THE CRITICAL STEP for the 'bad' model's data.
            # When a 'MOVE' event happens, the 'ATTACK' data is [None], which becomes [0.0].
            # The model is now forced to learn an association between the context for 'MOVE'
            # and an output of '0.0' for the 'ATTACK' data. This will directly conflict
            # with learning the legitimate 'ATTACK' event where the target_id is also 0.0.
            target_vector[target_vector == None] = 0.0

        sequence: list[Any] = [input_vector_tm1, input_vector_t, target_vector]
        vectorized_sequences.append(sequence)
        
    return vectorized_sequences

# --- Main Data Generation and Export ---
game_state_series = generate_game_state_series(SEQUENCE_LENGTH)
raw_training_data = generate_training_samples(game_state_series)

# AI: Must initialize vectorizers before vectorizing data.
_initialise_vectorizers(raw_training_data)

TRAINING_DATA_WITH_CONTEXT_VECTORISED = vectorize_data(raw_training_data, coerce_none_to_zero=False)
TRAINING_DATA_WITH_CONTEXT_VECTORISED_COERCION = vectorize_data(raw_training_data, coerce_none_to_zero=True)

# --- Helpers for model4.py ---
def get_vectorizer_output_length(event_id: event_id_t) -> int:
    """Returns the number of regression parameters for a given event type."""
    vectorizer = event_id_to_vectorizer.get(event_id)
    return len(vectorizer.get_feature_names_out()) if vectorizer else 0
    
def get_vector_sizes() -> Tuple[int, int]:
    """Returns the size of the input and data regression vectors."""
    # AI: The input is the game state vector plus the one-hot encoding of the previous event.
    input_size = (2 + NUM_TARGETS) + NUM_EVENT_TYPES
    data_vector_size = sum(get_vectorizer_output_length(eid) for eid in events_id_list)
    return input_size, data_vector_size

# --- Verification Script ---
if __name__ == "__main__":
    print("\n--- Demo 4 Data Generation ---")
    print(f"Generated {len(TRAINING_DATA_WITH_CONTEXT_VECTORISED)} training sequences.")
    
    input_size, data_vec_size = get_vector_sizes()
    print(f"Input vector size: {input_size}")
    print(f"Data regression vector size: {data_vec_size}")
    for eid in events_id_list:
        print(f"  - {eid}: {get_vectorizer_output_length(eid)} params")

    def print_sample_by_type(event_name: event_id_t):
        for i, (state_context, event) in enumerate(raw_training_data):
            if event['id'] == event_name:
                print(f"\n--- Sample for '{event_name}' event ---")
                print(f"  Raw context data: {event['context']}")
                if i > 0:
                    # AI: Note: Indexing vectorized data at [i-1] because the vectorization loop starts at 1
                    vec_idx = i - 1
                    correct_target = TRAINING_DATA_WITH_CONTEXT_VECTORISED[vec_idx][2]
                    coerced_target = TRAINING_DATA_WITH_CONTEXT_VECTORISED_COERCION[vec_idx][2]
                    print(f"  Full Target Vector (Correct): {correct_target}")
                    print(f"  Full Target Vector (Coerced): {coerced_target}")
                return
    
    # AI: Show a MOVE event to demonstrate how the None->0 coercion happens for the ATTACK data.
    print_sample_by_type(event_id_t('MOVE_TO_COORDS'))
    # AI: Show an ATTACK event to show the legitimate target_id=0.
    print_sample_by_type(event_id_t('ATTACK_TARGET'))
