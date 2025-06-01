# AI: models.py
# This file will contain the model implementation, training, and evaluation logic
# as per the requirements in models.txt and data_generation_outline.md.

from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Set

import numpy as np
# AI: NDArray import removed as np.ndarray[Any] will be used.
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# AI: Import data from data_generation.py
# Assuming data_generation.py is in the same directory or accessible via PYTHONPATH
try:
    # AI: If run as part of the Demos package
    from .data import TRAINING_DATA_WITH_CONTEXT, TRAINING_DATA_WITHOUT_CONTEXT, DATA_WITH_CONTEXT_SET, DATA_WITHOUT_CONTEXT_SET
except ImportError:
    # AI: Fallback for direct execution if Demos/models.py is run as a script
    # and data_generation.py is in the same directory.
    from Demos.Demo1.data import TRAINING_DATA_WITH_CONTEXT, TRAINING_DATA_WITHOUT_CONTEXT, DATA_WITH_CONTEXT_SET, DATA_WITHOUT_CONTEXT_SET


# AI: Define type aliases for clarity as per user rules
RawSequence = List[str] # A sequence from the input data
Context = Tuple[str, str] # A context of two preceding events
Target = str # The event to be predicted
ProbabilityDistribution = Dict[Target, float] # P(Target | Context)
ModelLearnedProbabilities = Dict[Context, ProbabilityDistribution] # Stores learned probabilities by the model

class SequencePredictor:
    # AI: Model to predict the next item in a sequence given a context of length 2.
    # It learns transition probabilities from the training data (a type of Markov model).

    _model: ModelLearnedProbabilities
    _label_encoder: LabelEncoder
    _labels: List[str] # Sorted list of unique labels for consistent confusion matrix ordering

    def __init__(self) -> None:
        # AI: Initialize the model structure.
        self._model = defaultdict(lambda: defaultdict(float))
        self._label_encoder = LabelEncoder()
        self._labels = []


    def _preprocess_data(self, data: List[RawSequence]) -> List[Tuple[Context, Target]]:
        # AI: Convert raw sequences into (context, target) pairs.
        # Context length is fixed at 2.
        processed_data: List[Tuple[Context, Target]] = []
        for seq in data:
            if len(seq) < 3:
                # AI: Sequence is too short to form a context of 2 and a target.
                # print(f"Warning: Skipping short sequence: {seq}")
                continue
            context: Context = (seq[0], seq[1])
            target: Target = seq[2]
            processed_data.append((context, target))
        return processed_data

    def fit(self, training_data: List[RawSequence], unique_labels: Set[str]) -> None:
        # AI: Train the model. For this statistical model, "training" involves counting occurrences
        # and calculating probabilities for P(Target | Context).
        
        if not unique_labels:
            print("Warning: No unique labels provided. Model may not behave as expected.")
            # AI: Initialize _labels as an empty list if unique_labels is empty,
            # rather than potentially leaving it in an undefined state from a previous call.
            self._labels = []
        else:
            self._labels = sorted(list(unique_labels))
        
        # AI: Fit label encoder with all known unique labels for consistent matrix ordering.
        # Sklearn's confusion matrix and accuracy_score handle string labels directly,
        # but fitting the encoder and using its 'classes_' (or our sorted '_labels')
        # ensures the confusion matrix axes are ordered consistently.
        if self._labels:
            self._label_encoder.fit(self._labels)
        else: # No labels, cannot really train or evaluate meaningfully
            print("Warning: Fitting with no labels. Model will be empty.")
            return


        processed_data = self._preprocess_data(training_data)
        if not processed_data:
            print("Warning: No processable data found for training. Model will be empty.")
            return

        # AI: Count occurrences of (context, target) pairs.
        # AI: Specify the type argument for Counter
        context_target_counts: Dict[Context, Counter[Target]] = defaultdict(Counter)
        for context, target in processed_data:
            context_target_counts[context][target] += 1

        # AI: Convert counts to probabilities.
        for context, target_counts_for_context in context_target_counts.items():
            total_occurrences_for_context = sum(target_counts_for_context.values())
            if total_occurrences_for_context > 0:
                current_context_probs: ProbabilityDistribution = {} # AI: Ensure this is ProbabilityDistribution
                for target_event, count in target_counts_for_context.items():
                    current_context_probs[target_event] = count / total_occurrences_for_context
                self._model[context] = current_context_probs
            # AI: No else needed here; if total_occurrences_for_context is 0, that context shouldn't have counts.


    def predict(self, context_list: List[Context]) -> Tuple[List[Target], List[ProbabilityDistribution]]:
        # AI: Predict the next item for a list of contexts.
        # For deterministic predictions, it picks the most probable target.
        # Also returns the probability distribution for the predicted target.
        
        predictions: List[Target] = []
        all_probabilities: List[ProbabilityDistribution] = []

        for context_tuple in context_list:
            # AI: Initialize dist_to_return here to ensure it's always bound.
            dist_to_return: ProbabilityDistribution = {}

            # AI: Check if this context was learned by the model
            if context_tuple in self._model and self._model[context_tuple]:
                target_probs: ProbabilityDistribution = self._model[context_tuple]
                # AI: Choose the target with the highest probability.
                # If probabilities are tied, max() on dict keys will pick one based on standard key ordering (e.g., lexicographical for strings).
                # The key argument to max should be a function that takes an item from the iterable (a key from target_probs)
                # and returns a value to compare. target_probs.get does exactly this.
                # AI: Ensure target_probs is not empty before calling max, though logic implies it won't be.
                if not target_probs:
                     # This case should ideally not be reached if context_tuple is in self._model and self._model[context_tuple] is non-empty
                    if self._labels: # Fallback if target_probs is unexpectedly empty
                        best_target = self._labels[0]
                        dist_to_return = {label: (1.0/len(self._labels) if self._labels else 0.0) for label in self._labels}
                    else:
                        best_target = "N/A_fallback" # Should not happen with proper training
                        dist_to_return = {}
                else:
                    # AI: Explicitly provide a default to `get` to ensure it always returns a float, satisfying some linters for `max` key.
                    best_target = max(target_probs, key=lambda k: target_probs.get(k, float("-inf"))) 
                    # AI: dist_to_return was moved to be initialized outside the if/else for target_probs
                    # so it needs to be set here if target_probs is valid.
                    dist_to_return = target_probs
                
                predictions.append(best_target)
                all_probabilities.append(dist_to_return)
            else:
                # AI: Context not seen during training or has no learned probabilities.
                # Fallback strategy: predict the first known label with uniform probability (if labels exist).
                # This ensures a prediction is always made.
                if self._labels:
                    num_labels = len(self._labels)
                    # AI: Consistent fallback: pick the first label.
                    best_target = self._labels[0]
                    prob_dist = {label: (1.0/num_labels if num_labels > 0 else 0.0) for label in self._labels}
                    predictions.append(best_target)
                    all_probabilities.append(prob_dist)
                else:
                    # AI: No labels known (e.g., model not trained or trained with no labels).
                    predictions.append("N/A") # Placeholder prediction
                    all_probabilities.append({}) # Empty probability distribution
        
        return predictions, all_probabilities

    def evaluate(self, test_data: List[RawSequence]) -> None:
        # AI: Evaluate the model on test data.
        # Generates confusion matrix and calculates accuracy.
        if not self._labels:
            print("Model not trained or no labels available. Cannot evaluate.")
            return

        contexts_for_prediction: List[Context] = []
        actual_targets: List[Target] = []

        for seq in test_data:
            if len(seq) < 3:
                continue # Skip sequences too short for evaluation
            contexts_for_prediction.append((seq[0], seq[1])) # The context part
            actual_targets.append(seq[2])                    # The target part
        
        if not actual_targets:
            print("No valid test data to evaluate (all sequences too short or empty test_data).")
            # AI: Print an empty confusion matrix representation or skip if preferred
            cm_empty = np.zeros((len(self._labels), len(self._labels)), dtype=int)
            print(f"Labels for Confusion Matrix: {self._labels}")
            print("Confusion Matrix (no data):")
            print(cm_empty)
            print(f"Accuracy: N/A (no data)")
            return

        predicted_targets, _ = self.predict(contexts_for_prediction)
        
        # AI: Ensure actual_targets and predicted_targets are aligned and valid for confusion_matrix.
        # The 'labels' parameter for confusion_matrix ensures consistent row/column order.
        # AI: For sklearn's confusion_matrix, the return is NDArray[np.int_] or similar.
        # Using np.ndarray[Any] to satisfy generic type requirement.
        cm: np.ndarray[Any] = confusion_matrix(actual_targets, predicted_targets, labels=self._labels)
        # AI: accuracy_score returns a float. np.float64 is a specific type of float.
        # Explicitly cast to float to satisfy linter if it expects standard float.
        acc: float = float(accuracy_score(actual_targets, predicted_targets))
        
        print(f"Labels for Confusion Matrix: {self._labels}")
        print("Confusion Matrix:")
        # AI: Print column headers for better readability
        header = "       " + " ".join([f"{label:>5}" for label in self._labels])
        print(header)
        print("     Predicted↓ True→")
        for i, label_true in enumerate(self._labels):
            row_str = f"{label_true:>5} |"
            for val in cm[i]:
                row_str += f"{val:>5} "
            print(row_str)
        
        print(f"Accuracy: {acc:.4f} ({int(acc * len(actual_targets))}/{len(actual_targets)})")


# AI: Main execution block
def main() -> None:
    # AI: Process data with context
    print("Evaluating model WITH context...")
    model_with_context = SequencePredictor()
    # AI: Pass unique labels from the dataset to fit() for consistent matrix ordering
    model_with_context.fit(TRAINING_DATA_WITH_CONTEXT, DATA_WITH_CONTEXT_SET)
    # AI: Evaluate on the same training data, as per the problem's nature (deterministic mapping from outline)
    model_with_context.evaluate(TRAINING_DATA_WITH_CONTEXT)
    print("-" * 40)

    # AI: Process data without context
    print("\nEvaluating model WITHOUT context...")
    model_without_context = SequencePredictor()
    model_without_context.fit(TRAINING_DATA_WITHOUT_CONTEXT, DATA_WITHOUT_CONTEXT_SET)
    model_without_context.evaluate(TRAINING_DATA_WITHOUT_CONTEXT)
    print("-" * 40)

    # AI: Note on theoretical accuracy for "without context":
    # The data_generation_outline.md mentions an expected accuracy of 0.8125 for the "without context" case.
    # This 0.8125 value appears to be calculated by averaging the maximum possible prediction accuracy
    # for each *unique context type*, weighted equally.
    # (e.g., Acc([A,A])=1.0, Acc([A,B])=1.0, Acc([B,A])=0.5, Acc([B,B])=0.75; average = (1+1+0.5+0.75)/4 = 0.8125).
    # The standard `accuracy_score` (and the confusion matrix) calculated here is instance-based:
    # (total correct predictions) / (total instances).
    # For the TRAINING_DATA_WITHOUT_CONTEXT, instance-based accuracy is 7/9 approx 0.7778,
    # because the model makes a single deterministic prediction for each context.
    # E.g., for context (B,A) with P(A)=0.5, P(B)=0.5, if the model predicts 'A',
    # it will be correct for 1 of the 2 (B,A) instances in the data.
    # The confusion matrix and accuracy reported by this script reflect this standard, instance-based calculation.

if __name__ == "__main__":
    main()
