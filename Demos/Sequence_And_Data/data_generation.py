import numpy as np
import pandas as pd
from typing import Any, List

# AI: Define types for clarity
event_type = int  # 0=A, 1=B{False}, 2=B{True}
sequence_state_id = int  # 0-8 representing different [event, event] combinations

# AI: Generate predictable 9x3 transition matrix with dominant transitions
def generate_transition_matrix(dominant_prob: float = 0.9) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Generate a 9x3 transition probability matrix where 9 sequence states transition to 3 next events."""
    matrix = np.zeros((9, 3))  # AI: 9 sequence states -> 3 possible next events
    
    # AI: For each sequence state, choose one dominant next event with ~90% probability
    for i in range(9):
        # AI: Choose a random dominant next event for this sequence state
        dominant_next_event: event_type = np.random.randint(0, 3)
        
        # AI: Set dominant probability
        matrix[i, dominant_next_event] = dominant_prob
        
        # AI: Distribute remaining probability among other events
        remaining_prob = 1.0 - dominant_prob
        other_events = [j for j in range(3) if j != dominant_next_event]
        
        # AI: Generate small random probabilities that sum to remaining_prob
        small_probs = np.random.rand(len(other_events))
        small_probs = small_probs / small_probs.sum() * remaining_prob
        
        # AI: Assign small probabilities to other events
        for j, event in enumerate(other_events):
            matrix[i, event] = small_probs[j]
    
    return matrix

# AI: Map two consecutive events to sequence state ID
def events_to_sequence_state(event1: event_type, event2: event_type) -> sequence_state_id:
    """Convert [event1, event2] to sequence state ID (0-8)."""
    return event1 * 3 + event2

# AI: Map sequence state ID back to two events
def sequence_state_to_events(state_id: sequence_state_id) -> tuple[event_type, event_type]:
    """Convert sequence state ID (0-8) back to [event1, event2]."""
    event1 = state_id // 3
    event2 = state_id % 3
    return event1, event2

# AI: Generate sequences using the transition matrix
def generate_sequence(transition_matrix: np.ndarray[Any, np.dtype[np.float64]], 
                      sequence_length: int,
                      initial_events: tuple[event_type, event_type] = (0, 0)
                      ) -> List[event_type]:
    """Generate event sequences using the 9x3 transition matrix."""
    
    # AI: Start with initial two events
    sequence = list(initial_events)
    
    # AI: Generate remaining events
    for _ in range(sequence_length - 2):
        # AI: Get current sequence state from last two events
        current_state = events_to_sequence_state(sequence[-2], sequence[-1])
        
        # AI: Get transition probabilities for current sequence state
        probs = transition_matrix[current_state]
        
        # AI: Sample next event based on probabilities
        next_event: event_type = np.random.choice(3, p=probs)
        sequence.append(next_event)
    
    return sequence

def print_matrix_info(matrix: np.ndarray[Any, np.dtype[np.float64]]) -> None:
    """Print matrix with sequence state labels and verification."""
    # AI: Sequence state labels based on the outline
    sequence_states = [
        "[A, A]",           # 0: (0,0)
        "[A, B{False}]",    # 1: (0,1)  
        "[A, B{True}]",     # 2: (0,2)
        "[B{False}, A]",    # 3: (1,0)
        "[B{False}, B{False}]", # 4: (1,1)
        "[B{False}, B{True}]",  # 5: (1,2)
        "[B{True}, A]",     # 6: (2,0)
        "[B{True}, B{False}]",  # 7: (2,1)
        "[B{True}, B{True}]"    # 8: (2,2)
    ]
    
    # AI: Next event labels
    next_events = ["A", "B{False}", "B{True}"]
    
    # AI: Create DataFrame for better visualization
    df = pd.DataFrame(matrix, 
                     index=[f"State {i}: {state}" for i, state in enumerate(sequence_states)],
                     columns=[f"→{event}" for event in next_events])
    
    print("9x3 Transition Probability Matrix")
    print("==================================")
    print("Each row represents a sequence state (last 2 events), each column represents next event")
    print("Each row sums to 1.0\n")
    
    print(df.round(3))
    
    print(f"\nRow sums (should all be 1.0):")
    row_sums = matrix.sum(axis=1)
    for i, sum_val in enumerate(row_sums):
        print(f"  Row {i}: {sum_val:.6f}")

# AI: Print generated sequences with event labels
def print_sequence(sequence: List[event_type]) -> None:
    """Print generated event sequences with readable labels."""
    event_labels = ["A", "B{False}", "B{True}"]
    
    print("\nGenerated Event Sequence")
    print("========================")
    print("Event indices:", " -> ".join(map(str, sequence)))
    print("Event labels: ", " -> ".join([event_labels[event] for event in sequence]))

def generate_data_for_sequence_and_data(sequence_length: int = 1000) -> List[event_type]:
    """Generate a random event sequence using sequence-to-event transitions."""
    transition_matrix: np.ndarray[Any, np.dtype[np.float64]] = generate_transition_matrix()
    print_matrix_info(transition_matrix)
    sequence = generate_sequence(transition_matrix, sequence_length=sequence_length)
    print_sequence(sequence)
    return sequence

if __name__ == "__main__":
    # AI: Generate and display the transition matrix
    np.random.seed(42)  # For reproducible results
    generate_data_for_sequence_and_data(sequence_length=20)

    
