# AI: Demos/Demo4/plotting4.py
# This script visualizes the results from the model comparison experiment in Demo 4.
# It generates plots for learning curves, error distributions, and prediction accuracy
# to clearly illustrate the performance difference between the masked and coerced models.

import os
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from typing import List, Dict, Any, Tuple

def plot_aggregated_learning_curves(
    ax: Axes, 
    all_curves: List[Dict[str, Any]], 
    metric: str, 
    color: str, 
    label: str, 
    linestyle: str = '-'
):
    """
    Plots individual learning curves with transparency and a bold average curve.
    """
    all_metrics: List[List[float]] = []
    max_len = 0

    for curve in all_curves:
        if not curve[metric]: continue
        # Plot individual trial curves
        ax.plot(curve['epochs'], curve[metric], color=color, alpha=0.15, linestyle=linestyle)
        all_metrics.append(curve[metric])
        if len(curve['epochs']) > max_len:
            max_len = len(curve['epochs'])
    
    if not all_metrics: return

    # Pad metrics to the same length for averaging
    padded_metrics = [m + [np.nan] * (max_len - len(m)) for m in all_metrics]
    mean_metric = np.nanmean(np.array(padded_metrics), axis=0)
    
    # Get the corresponding epochs from the longest trial
    longest_curve = max(all_curves, key=lambda c: len(c['epochs']))
    epochs = longest_curve['epochs']

    # Plot average curve
    ax.plot(epochs, mean_metric, color=color, label=label, linewidth=2.5, linestyle=linestyle)

def plot_prediction_distribution(data: Dict[str, Any], graphs_dir: str):
    """
    AI: Plots the distribution of predicted vs. actual values for the 'target_id' feature.
    This plot is crucial for visualizing the spurious correlation learned by the incorrect model.
    """
    correct_evals = data['correct_model'].get('all_evaluations', [])
    incorrect_evals = data['incorrect_model'].get('all_evaluations', [])

    if not correct_evals or not incorrect_evals:
        print("Evaluation data for prediction plot not found. Skipping.")
        return

    # AI: In data4, the 'ATTACK_TARGET' event has one feature: 'target_id'. Its index in the 
    # regression vector is after the 'MOVE_TO_COORDS' features (x, y). So it's at index 2.
    feature_index = 2 
    feature_name = "target_id"

    def collect_points(evaluations: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
        all_preds, all_targets = [], []
        for trial in evaluations:
            for i in range(len(trial['predictions'])):
                pred_vector = np.array(trial['predictions'][i]).flatten()
                target_vector = np.array(trial['targets'][i]).flatten()
                
                if len(target_vector) > feature_index and not np.isnan(target_vector[feature_index]):
                    all_preds.append(pred_vector[feature_index])
                    all_targets.append(target_vector[feature_index])
        return all_preds, all_targets

    c_preds, c_targets = collect_points(correct_evals)
    ic_preds, ic_targets = collect_points(incorrect_evals)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    fig.suptitle(f"Predicted vs. Actual Values for '{feature_name}'", fontsize=16, y=0.98)

    plot_configs = [
        {'ax': axes[0], 'preds': c_preds, 'targets': c_targets, 'color': 'blue', 'title': 'Correct Model (Masking)'},
        {'ax': axes[1], 'preds': ic_preds, 'targets': ic_targets, 'color': 'orange', 'title': 'Incorrect Model (Coercion)'}
    ]

    all_values = c_preds + c_targets + ic_preds + ic_targets
    if not all_values:
        print(f"No valid data for '{feature_name}' found to plot. Skipping.")
        plt.close(fig)
        return
        
    min_val, max_val = min(all_values), max(all_values)
    padding = (max_val - min_val) * 0.1 if (max_val - min_val) > 0 else 1.0
    ax_min, ax_max = min_val - padding, max_val + padding

    for config in plot_configs:
        ax = config['ax']
        if config['preds']:
            label = f"n={len(config['preds'])}"
            ax.scatter(config['targets'], config['preds'], alpha=0.4, color=config['color'], label=label, s=50)
            ax.plot([ax_min, ax_max], [ax_min, ax_max], 'r--', label='Ideal (y=x)')
            ax.axhline(0, color='gray', linestyle=':', linewidth=1)
            ax.axvline(0, color='gray', linestyle=':', linewidth=1)
        ax.set_title(config['title'])
        ax.set_ylabel('Predicted Value')
        ax.set_xlabel('Actual Value')
        ax.grid(True)
        ax.legend()
        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ax_min, ax_max)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(graphs_dir, 'prediction_vs_actual_scatter.png')
    plt.savefig(plot_path)
    print(f"Saved prediction vs actual scatter plot to: {plot_path}")
    plt.close(fig)

def plot_results():
    """
    Plots the results from the .npy file generated by models4.py.
    """
    results_path = os.path.join(os.path.dirname(__file__), 'results.npy')
    graphs_dir = os.path.join(os.path.dirname(__file__), 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)

    if not os.path.exists(results_path):
        print(f"Results file not found at: {results_path}")
        print("Please run models4.py first to generate the results.")
        return

    data = np.load(results_path, allow_pickle=True).item()

    correct_model_data = data['correct_model']
    incorrect_model_data = data['incorrect_model']
    num_trials = data['num_trials']
    
    color_correct = 'blue'
    color_incorrect = 'orange'

    # --- Histogram of Final MSEs ---
    plt.figure(figsize=(12, 7))
    bins = np.histogram(np.hstack((correct_model_data['mse_results'], incorrect_model_data['mse_results'])), bins=20)[1]
    plt.hist(correct_model_data['mse_results'], bins=bins, alpha=0.7, label='Correct Model (Masking)', color=color_correct)
    plt.hist(incorrect_model_data['mse_results'], bins=bins, alpha=0.7, label='Incorrect Model (Coercion)', color=color_incorrect)
    
    stats_text_correct = f"Mean: {correct_model_data['mean_mse']:.4f}\nStd: {correct_model_data['std_mse']:.4f}"
    stats_text_incorrect = f"Mean: {incorrect_model_data['mean_mse']:.4f}\nStd: {incorrect_model_data['std_mse']:.4f}"

    plt.axvline(correct_model_data['mean_mse'], color=color_correct, linestyle='--', linewidth=2)
    plt.axvline(incorrect_model_data['mean_mse'], color=color_incorrect, linestyle='--', linewidth=2)

    plt.text(0.95, 0.95, stats_text_incorrect, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right', color='white',
             bbox=dict(boxstyle='round,pad=0.5', fc=color_incorrect, alpha=0.8))
    plt.text(0.95, 0.75, stats_text_correct, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right', color='white',
             bbox=dict(boxstyle='round,pad=0.5', fc=color_correct, alpha=0.8))

    plt.xlabel('Average Targeted MSE')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Model Errors over {num_trials} Trials')
    plt.legend()
    plt.grid(True, axis='y')
    
    plot_path = os.path.join(graphs_dir, 'error_distribution_histogram.png')
    plt.savefig(plot_path)
    print(f"Saved error distribution plot to: {plot_path}")
    plt.close()

    # --- Aggregated Learning Curves (Loss & MSE) ---
    if 'all_learning_curves' in correct_model_data and correct_model_data['all_learning_curves']:
        # Plot for Hybrid Loss
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), sharex=True)
        fig.suptitle(f'Aggregated Learning Curves over {num_trials} Trials', fontsize=16)

        plot_aggregated_learning_curves(ax1, correct_model_data['all_learning_curves'], 'train_loss', color_correct, 'Correct - Train Loss', linestyle='--')
        plot_aggregated_learning_curves(ax1, correct_model_data['all_learning_curves'], 'val_loss', color_correct, 'Correct - Val Loss')
        plot_aggregated_learning_curves(ax1, incorrect_model_data['all_learning_curves'], 'train_loss', color_incorrect, 'Incorrect - Train Loss', linestyle='--')
        plot_aggregated_learning_curves(ax1, incorrect_model_data['all_learning_curves'], 'val_loss', color_incorrect, 'Incorrect - Val Loss')
        ax1.set_ylabel('Hybrid Loss')
        ax1.legend()
        ax1.grid(True)
        ax1.set_yscale('log')
        ax1.set_title('Hybrid Loss Learning Curves')

        # Plot for Targeted MSE
        plot_aggregated_learning_curves(ax2, correct_model_data['all_learning_curves'], 'train_mse', color_correct, 'Correct - Train MSE', linestyle='--')
        plot_aggregated_learning_curves(ax2, correct_model_data['all_learning_curves'], 'val_mse', color_correct, 'Correct - Val MSE')
        plot_aggregated_learning_curves(ax2, incorrect_model_data['all_learning_curves'], 'train_mse', color_incorrect, 'Incorrect - Train MSE', linestyle='--')
        plot_aggregated_learning_curves(ax2, incorrect_model_data['all_learning_curves'], 'val_mse', color_incorrect, 'Incorrect - Val MSE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Targeted MSE')
        ax2.legend()
        ax2.grid(True)
        ax2.set_yscale('log')
        ax2.set_title('Targeted MSE Learning Curves')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        learning_curve_path = os.path.join(graphs_dir, 'aggregated_learning_curves.png')
        plt.savefig(learning_curve_path)
        print(f"Saved aggregated learning curve plot to: {learning_curve_path}")
        plt.close()

    # --- Prediction vs Actual Scatter Plot ---
    plot_prediction_distribution(data, graphs_dir)

if __name__ == "__main__":
    plot_results()
