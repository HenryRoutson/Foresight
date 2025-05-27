# Output Normalization for Unified Event Prediction

## The Constraint Problem

In our unified output vector, different types of data have different mathematical constraints:

```python
output_vector = [
  0.35,  # probability - MUST sum to 1 across all outcomes
  45,    # time_diff_minutes - any positive value
  2,     # num_images - non-negative integer
  0.7,   # severity - bounded between 0-1
  1,     # boolean - 0 or 1
]
```

## Minimal Constraint Enforcement

we apply targeted normalization:

### 1. Probability Normalization (Critical)
```python
def normalize_probabilities(output_vector, prob_indices):
    # AI: Extract probability values from specific positions
    probs = output_vector[prob_indices]
    
    # AI: Apply softmax to ensure they sum to 1
    normalized_probs = torch.softmax(probs, dim=-1)
    
    # AI: Put normalized probabilities back into vector
    output_vector[prob_indices] = normalized_probs
    return output_vector
```

### 2. Other Constraint Types
```python
def apply_data_constraints(output_vector, constraint_metadata):
    for i, constraint_type in enumerate(constraint_metadata):
        if constraint_type == "probability":
            continue  # Already handled by softmax
        elif constraint_type == "non_negative":
            output_vector[i] = F.relu(output_vector[i])
        elif constraint_type == "bounded_0_1": 
            output_vector[i] = torch.sigmoid(output_vector[i])
        elif constraint_type == "integer":
            output_vector[i] = torch.round(output_vector[i])
        elif constraint_type == "boolean":
            output_vector[i] = torch.round(torch.sigmoid(output_vector[i]))
    
    return output_vector
```


## Why This Beats Specialized Heads

- **Shared Learning**: All outcomes learn from each other through shared backbone
