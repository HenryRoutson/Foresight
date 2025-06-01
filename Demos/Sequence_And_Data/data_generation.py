import numpy as np
import pandas as pd
from typing import Any, List

# AI: Define types for clarity
event_type = int  # 0=A, 1=B{False}, 2=B{True}
sequence_state_id = int  # 0-8 representing different [event, event] combinations

# AI: Generate deterministic 9x3 transition matrix as specified in outline
def generate_transition_matrix() -> np.ndarray[Any, np.dtype[np.float64]]:
    """Generate the deterministic 9x3 transition matrix as specified in the outline."""
    matrix = np.zeros((9, 3))  # AI: 9 sequence states -> 3 possible next events
    
    # AI: Set transitions exactly as specified in outline:
    # [A, A] -> A (100%)
    matrix[0, 0] = 1.0  # State 0: [A, A] -> A
    
    # [A, B{False}] -> B{False} (100%)  
    matrix[1, 1] = 1.0  # State 1: [A, B{False}] -> B{False}
    
    # [A, B{True}] -> B{True} (100%)
    matrix[2, 2] = 1.0  # State 2: [A, B{True}] -> B{True}
    
    # [B{False}, A] -> A (100%)
    matrix[3, 0] = 1.0  # State 3: [B{False}, A] -> A
    
    # [B{False}, B{False}] -> A (100%)
    matrix[4, 0] = 1.0  # State 4: [B{False}, B{False}] -> A
    
    # [B{False}, B{True}] -> B{False} (100%)
    matrix[5, 1] = 1.0  # State 5: [B{False}, B{True}] -> B{False}
    
    # [B{True}, A] -> B{True} (100%)
    matrix[6, 2] = 1.0  # State 6: [B{True}, A] -> B{True}
    
    # [B{True}, B{False}] -> B{False} (100%)
    matrix[7, 1] = 1.0  # State 7: [B{True}, B{False}] -> B{False}
    
    # [B{True}, B{True}] -> B{True} (100%)
    matrix[8, 2] = 1.0  # State 8: [B{True}, B{True}] -> B{True}
    
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
    """Generate a deterministic event sequence that follows the transition matrix exactly."""
    transition_matrix: np.ndarray[Any, np.dtype[np.float64]] = generate_transition_matrix()
    print_matrix_info(transition_matrix)
    
    # AI: Generate multiple sequences with different starting points and combine them
    # AI: This ensures we get varied patterns while maintaining deterministic rules
    
    all_sequences : List[event_type] = []
    
    # AI: Define all possible starting states to ensure coverage
    starting_states : List[tuple[event_type, event_type]] = [
        (0, 0),  # [A, A] 
        (0, 1),  # [A, B{False}]
        (0, 2),  # [A, B{True}]
        (1, 0),  # [B{False}, A]
        (1, 1),  # [B{False}, B{False}]
        (1, 2),  # [B{False}, B{True}]
        (2, 0),  # [B{True}, A]
        (2, 1),  # [B{True}, B{False}]
        (2, 2),  # [B{True}, B{True}]
    ]
    
    # AI: Generate shorter sequences for each starting state
    seq_per_start : int = max(50, sequence_length // (len(starting_states) * 2))
    
    for start_state in starting_states:
        # AI: Generate sequence following deterministic rules
        sequence = generate_sequence(transition_matrix, seq_per_start, start_state)
        all_sequences.extend(sequence)
        
        # AI: Add some randomness by starting from middle of this sequence
        if len(sequence) > 10:
            mid_point = len(sequence) // 2
            partial_start = (sequence[mid_point], sequence[mid_point + 1])
            partial_sequence = generate_sequence(transition_matrix, seq_per_start // 2, partial_start)
            all_sequences.extend(partial_sequence)
    
    # AI: Trim to exact length and shuffle to mix patterns
    all_sequences = all_sequences[:sequence_length]
    
    # AI: Shuffle in chunks to maintain some local patterns but mix globally
    chunk_size : int = 20
    chunks : List[List[event_type]] = [all_sequences[i:i+chunk_size] for i in range(0, len(all_sequences), chunk_size)]
    np.random.shuffle(chunks)
    sequence: List[event_type]= []
    for chunk in chunks:
        sequence.extend(chunk)
    
    # AI: Final trim
    sequence = sequence[:sequence_length]
    
    print(f"\nGenerated sequence with {len(starting_states)} different starting patterns")
    print(f"Total sequence length: {len(sequence)}")
    
    # AI: Verify the sequence follows deterministic rules
    print("\nVerification: Checking if sequence follows deterministic transition matrix...")
    violations = 0
    labels = ["A", "B{False}", "B{True}"]
    
    for i in range(len(sequence) - 2):
        state = events_to_sequence_state(sequence[i], sequence[i+1])
        actual_next = sequence[i+2]
        
        # AI: Check what the transition matrix says should happen
        expected_probs = transition_matrix[state]
        expected_next = np.argmax(expected_probs)
        
        if actual_next != expected_next:
            violations += 1
            if violations <= 5:  # AI: Show first few violations
                state_desc = f"[{labels[sequence[i]]}, {labels[sequence[i+1]]}]"
                print(f"  Violation {violations}: {state_desc} -> {labels[actual_next]} (expected {labels[expected_next]})")
    
    print(f"Total violations: {violations}/{len(sequence)-2} ({100*violations/(len(sequence)-2):.1f}%)")
    
    if violations == 0:
        print("✓ Perfect compliance with deterministic transition matrix!")
    else:
        print("⚠ Sequence contains violations - data generation needs fixing")
    
    return sequence

if __name__ == "__main__":
    # AI: Generate and display the transition matrix
    np.random.seed(42)  # For reproducible results
    generate_data_for_sequence_and_data(sequence_length=20)

    
