import numpy as np
import pandas as pd
from typing import Any, List

# AI: Generate predictable 9x9 transition matrix with dominant transitions
def generate_transition_matrix(size: int = 9, dominant_prob: float = 0.97) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Generate a transition probability matrix with predictable sequences."""
    matrix = np.zeros((size, size))
    
    # AI: For each state, choose one dominant next state with ~97% probability
    for i in range(size):
        # AI: Choose a random dominant next state for this current state
        dominant_next_state: int = np.random.randint(0, size)
        
        # AI: Set dominant probability
        matrix[i, dominant_next_state] = dominant_prob
        
        # AI: Distribute remaining probability among other states
        remaining_prob = 1.0 - dominant_prob
        other_states = [j for j in range(size) if j != dominant_next_state]
        
        # AI: Generate small random probabilities that sum to remaining_prob
        small_probs = np.random.rand(len(other_states))
        small_probs = small_probs / small_probs.sum() * remaining_prob
        
        # AI: Assign small probabilities to other states
        for j, state in enumerate(other_states):
            matrix[i, state] = small_probs[j]
    
    return matrix

# AI: Generate sequences using the transition matrix
def generate_sequence(transition_matrix: np.ndarray[Any, np.dtype[np.float64]], 
                      sequence_length: int,
                      initial_state: int = 0
                      ) -> List[int]:
    """Generate sequences using the transition matrix."""

    sequence = [initial_state]
    current_state = initial_state
    
    # AI: Generate sequence by sampling from transition probabilities
    for _ in range(sequence_length - 1):
        # AI: Get transition probabilities for current state
        probs = transition_matrix[current_state]
        
        # AI: Sample next state based on probabilities
        next_state = np.random.choice(len(probs), p=probs)
        sequence.append(next_state)
        current_state = next_state
    

    return sequence

def print_matrix_info(matrix: np.ndarray[Any, np.dtype[np.float64]]) -> None:
    """Print matrix with state labels and verification."""
    # AI: State labels based on the example.txt file
    states = [
        "[A, A]",           # 0,0
        "[A, B{False}]",    # 0,1  
        "[A, B{True}]",     # 0,2
        "[B{False}, A]",    # 1,0
        "[B{False}, B{False}]", # 1,1
        "[B{False}, B{True}]",  # 1,2
        "[B{True}, A]",     # 2,0
        "[B{True}, B{False}]",  # 2,1
        "[B{True}, B{True}]"    # 2,2
    ]
    
    # AI: Create DataFrame for better visualization
    df = pd.DataFrame(matrix, 
                     index=[f"State {i}: {state}" for i, state in enumerate(states)],
                     columns=[f"→{i}" for i in range(9)])
    
    print("9x9 Transition Probability Matrix")
    print("=====================================")
    print("Each row represents current state, each column represents next state")
    print("Each row sums to 1.0\n")
    
    print(df.round(3))
    
    print(f"\nRow sums (should all be 1.0):")
    row_sums = matrix.sum(axis=1)
    for i, sum_val in enumerate(row_sums):
        print(f"  Row {i}: {sum_val:.6f}")

# AI: Print generated sequences with state labels
def print_sequence(sequence: List[int]) -> None:
    """Print generated sequences with readable state labels."""
    state_labels = [
        "[A, A]",           
        "[A, B{False}]",    
        "[A, B{True}]",     
        "[B{False}, A]",    
        "[B{False}, B{False}]", 
        "[B{False}, B{True}]",  
        "[B{True}, A]",     
        "[B{True}, B{False}]",  
        "[B{True}, B{True}]"    
    ]
    
    print("\nGenerated Sequence")
    print("===================")
    print("State indices:", " -> ".join(map(str, sequence)))
    print("State labels: ", " -> ".join([state_labels[state] for state in sequence]))




def generate_data_for_sequence_and_data(sequence_length: int = 1000) -> List[int]:
    """Generate a random sequence."""
    transition_matrix: np.ndarray[Any, np.dtype[np.float64]] = generate_transition_matrix()
    print_matrix_info(transition_matrix)
    sequence = generate_sequence(transition_matrix, sequence_length=sequence_length)
    print_sequence(sequence)
    return sequence




if __name__ == "__main__":
    # AI: Generate and display the transition matrix
    np.random.seed(42)  # For reproducible results
    generate_data_for_sequence_and_data(sequence_length=100)

    
