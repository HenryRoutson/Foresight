import numpy as np
import pandas as pd
from typing import Any

# AI: Generate random 9x9 transition matrix where each row sums to 1
def generate_transition_matrix(size: int = 9) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Generate a random transition probability matrix."""
    # AI: Create random matrix with positive values
    matrix = np.random.rand(size, size)
    
    # AI: Normalize each row to sum to 1 (row-stochastic matrix)
    row_sums = matrix.sum(axis=1, keepdims=True)
    transition_matrix = matrix / row_sums
    
    return transition_matrix

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

if __name__ == "__main__":
    # AI: Generate and display the transition matrix
    np.random.seed(42)  # For reproducible results
    transition_matrix = generate_transition_matrix()
    print_matrix_info(transition_matrix)
    
    # AI: Save matrix to file
    np.save("transition_matrix.npy", transition_matrix)
    print(f"\nMatrix saved to 'transition_matrix.npy'") 