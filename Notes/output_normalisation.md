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


## Why This Beats Specialized Heads

- **Shared Learning**: All outcomes learn from each other through shared backbone
