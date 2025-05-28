import numpy as np
import pandas as pd
from typing import Any, List

# AI: Generate predictable 9x9 transition matrix with dominant transitions
def generate_transition_matrix(size: int = 9, dominant_prob: float = 0.97) -> np.ndarray:
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
def generate_sequences(transition_matrix: np.ndarray, 
                      num_sequences: int = 10, 
                      sequence_length: int = 20,
                      initial_state: int = 0) -> List[List[int]]:
    """Generate sequences using the transition matrix."""
    sequences = []
    
    for _ in range(num_sequences):
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
        
        sequences.append(sequence)
    
    return sequences

def print_matrix_info(matrix: np.ndarray) -> None:
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
def print_sequences(sequences: List[List[int]]) -> None:
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
    
    print("\nGenerated Sequences")
    print("===================")
    for i, sequence in enumerate(sequences):
        print(f"\nSequence {i+1}:")
        print("State indices:", " -> ".join(map(str, sequence)))
        print("State labels: ", " -> ".join([state_labels[state] for state in sequence]))

if __name__ == "__main__":
    # AI: Generate and display the transition matrix
    np.random.seed(42)  # For reproducible results
    transition_matrix = generate_transition_matrix()
    print_matrix_info(transition_matrix)
    
    # AI: Save matrix to file
    np.save("transition_matrix.npy", transition_matrix)
    print(f"\nMatrix saved to 'transition_matrix.npy'")
    
    # AI: Generate sequences using the transition matrix
    sequences = generate_sequences(transition_matrix, num_sequences=5, sequence_length=15, initial_state=0)
    print_sequences(sequences)
    
    # AI: Save sequences to file
    np.save("generated_sequences.npy", np.array(sequences, dtype=object))
    print(f"\nSequences saved to 'generated_sequences.npy'") 