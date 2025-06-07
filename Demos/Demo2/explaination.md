# Analysis of Backpropagation: Masking vs. Coercion

The core purpose of this demonstration is to illustrate how to correctly handle missing or non-applicable data in a neural network's regression output. Incorrectly handling this can lead to "polluted" error signals during backpropagation, teaching the model unintended and wrong patterns.

We compare two scenarios to show this:

### 1. "WITH CONTEXT" (Correct Method: Masking)

In this scenario, `None` values in the target data are converted to `np.nan`. A mask is then used to exclude these `NaN` values from the loss calculation.

**Example from logs:**
```
--- Loss Details for: WITH CONTEXT (Epoch 0) ---
Sample Target Data (first 10 elements): tensor([nan, nan, nan])
Sample Prediction (first 10 elements):  [ 0.2165  0.1033 -1.1319]
Mask (first 10 elements):               tensor([False, False, False])
```

-   **Target Data**: The `nan` values correctly signify that there is no ground-truth data for these output neurons.
-   **Mask**: The mask becomes `False` for these positions, preventing them from contributing to the loss.
-   **Result**: The model is **not** penalized for its predictions in these positions. The backpropagation algorithm receives error signals *only* from valid data points, leading to a clean and accurate learning process.

### 2. "WITH CONTEXT BUT WITHOUT BACKPROP NONE VALUES" (Incorrect Method: Coercion)

In this scenario, `None` values are incorrectly coerced to `0.0`. This forces the model to treat `0.0` as a real target value.

**Example from logs:**
```
--- Loss Details for: WITH CONTEXT BUT WITHOUT BACKPROP NONE VALUES (Epoch 0) ---
Sample Target Data (first 10 elements): tensor([0., 0., 0.])
Sample Prediction (first 10 elements):  [-0.367  -0.2206 -0.2516]
Mask (first 10 elements):               tensor([True, True, True])
```

-   **Target Data**: Forcing `None` to `0.0` creates an incorrect target. It tells the model the ground truth for these outputs is zero when, in fact, they are irrelevant.
-   **Mask**: Because `0.0` is a valid number, the mask is `True`, incorrectly including these positions in the loss calculation.
-   **Result**: The model is now actively penalized for its predictions on irrelevant data. It receives a **polluted error signal** during backpropagation, forcing it to waste capacity learning to output zeros for data that should be ignored.

### Conclusion

This direct comparison proves the hypothesis. Forcing non-applicable data to a default value like `0.0` introduces a misleading loss signal that corrupts the training process. By converting `None` to `NaN` and using a mask, you ensure that backpropagation is only influenced by genuine error. This is the correct and standard way to handle such cases, leading to a more robust and accurate model.
