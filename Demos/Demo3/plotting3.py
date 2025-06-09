import json
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
    all_metrics : list[Any] = []
    max_len : int = 0

    for curve in all_curves:
        # Plot individual trial curves
        ax.plot(curve['epochs'], curve[metric], color=color, alpha=0.15, linestyle=linestyle)
        all_metrics.append(curve[metric])
        if len(curve['epochs']) > max_len:
            max_len = len(curve['epochs'])
    
    # Pad metrics to the same length for averaging
    padded_metrics : list[Any] = [m + [np.nan] * (max_len - len(m)) for m in all_metrics]
    mean_metric = np.nanmean(np.array(padded_metrics), axis=0)
    
    # Get the corresponding epochs from the longest trial
    longest_curve = max(all_curves, key=lambda c: len(c['epochs']))
    epochs = longest_curve['epochs']

    # Plot average curve
    ax.plot(epochs, mean_metric, color=color, label=label, linewidth=2.5, linestyle=linestyle)

def plot_prediction_distribution(data: Dict[str, Any], graphs_dir: str):
    """
    Plots the distribution of predicted values against their target values
    for each feature index and model type, using four separate subplots.
    """
    correct_evals = data['correct_model'].get('all_evaluations', [])
    incorrect_evals = data['incorrect_model'].get('all_evaluations', [])

    if not correct_evals or not incorrect_evals:
        print("Evaluation data for prediction distribution plot not found. Skipping.")
        return

    # Helper to collect prediction and target points for each feature index
    def collect_points(evaluations: List[Dict[str, Any]]) -> Tuple[List[float], List[float], List[float], List[float]]:
        all_preds_0 : list[Any]
        all_targets_0 : list[Any]
        all_preds_1 : list[Any]
        all_targets_1 : list[Any]

        all_preds_0, all_targets_0 = [], []
        all_preds_1, all_targets_1 = [], []

        for trial in evaluations:
            for i in range(len(trial['predictions'])):
                pred_vector = np.array(trial['predictions'][i]).flatten()
                target_vector = np.array(trial['targets'][i]).flatten()
                
                # Feature Index 0
                if not np.isnan(target_vector[0]):
                    all_preds_0.append(pred_vector[0])
                    all_targets_0.append(target_vector[0])
                
                # Feature Index 1
                if len(target_vector) > 1 and not np.isnan(target_vector[1]):
                    all_preds_1.append(pred_vector[1])
                    all_targets_1.append(target_vector[1])

        return all_preds_0, all_targets_0, all_preds_1, all_targets_1

    c_preds_0, c_targets_0, c_preds_1, c_targets_1 = collect_points(correct_evals)
    ic_preds_0, ic_targets_0, ic_preds_1, ic_targets_1 = collect_points(incorrect_evals)
    
    # Create 4 vertical subplots
    fig, axes = plt.subplots(4, 1, figsize=(10, 24), sharex=True, sharey=True)
    fig.suptitle('Predicted vs. Actual Values for Regression Targets', fontsize=16, y=0.96)

    plot_configs = [
        {'ax': axes[0], 'preds': c_preds_0, 'targets': c_targets_0, 'color': 'blue', 'title': 'Correct Model - Feature 0'},
        {'ax': axes[1], 'preds': ic_preds_0, 'targets': ic_targets_0, 'color': 'orange', 'title': 'Incorrect Model - Feature 0'},
        {'ax': axes[2], 'preds': c_preds_1, 'targets': c_targets_1, 'color': 'blue', 'title': 'Correct Model - Feature 1'},
        {'ax': axes[3], 'preds': ic_preds_1, 'targets': ic_targets_1, 'color': 'orange', 'title': 'Incorrect Model - Feature 1'}
    ]

    all_values = c_preds_0 + c_targets_0 + c_preds_1 + c_targets_1 + ic_preds_0 + ic_targets_0 + ic_preds_1 + ic_targets_1
    if not all_values:
        print("No valid data points found to plot prediction distribution. Skipping.")
        plt.close(fig)
        return
        
    min_val, max_val = min(all_values), max(all_values)
    padding = (max_val - min_val) * 0.1
    ax_min = min_val - padding
    ax_max = max_val + padding

    for config in plot_configs:
        ax = config['ax']
        if config['preds']:
            model_type_label = config['title'].split(' - ')[0]
            label = f"{model_type_label} ({config['color']}, n={len(config['preds'])})"
            ax.scatter(config['targets'], config['preds'], alpha=0.5, color=config['color'], label=label)
            ax.plot([ax_min, ax_max], [ax_min, ax_max], 'r--', label='Ideal (y=x)')
            ax.axhline(0, color='gray', linestyle=':', linewidth=1)
            ax.axvline(0, color='gray', linestyle=':', linewidth=1)
        ax.set_title(config['title'])
        ax.set_ylabel('Predicted Value')
        ax.grid(True)
        ax.legend()
        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ax_min, ax_max)
        ax.set_aspect('equal', adjustable='box')


    axes[3].set_xlabel('Actual Value')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(graphs_dir, 'prediction_vs_actual_scatter.png')
    plt.savefig(plot_path)
    print(f"Saved prediction vs actual scatter plot to: {plot_path}")
    plt.close(fig)

def plot_results():
    """
    Plots the results from the .npy file.
    """
    results_path = os.path.join(os.path.dirname(__file__), 'results.npy')
    graphs_dir = os.path.join(os.path.dirname(__file__), 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)

    if not os.path.exists(results_path):
        print(f"Results file not found at: {results_path}")
        print("Please run models3.py first to generate the results.")
        return

    data = np.load(results_path, allow_pickle=True).item()

    correct_model_data = data['correct_model']
    incorrect_model_data = data['incorrect_model']
    num_trials = data['num_trials']
    
    color_correct = 'blue'
    color_incorrect = 'orange'

    # --- Create and save the histogram ---
    plt.figure(figsize=(12, 7))
    plt.hist(correct_model_data['mse_results'], bins=10, alpha=0.7, label=f'Correct Model (Masking, {color_correct.capitalize()})', color=color_correct)
    plt.hist(incorrect_model_data['mse_results'], bins=10, alpha=0.7, label=f'Incorrect Model (Coercion, {color_incorrect.capitalize()})', color=color_incorrect)
    
    # AI: Add text for mean and std
    stats_text_correct = (f"Mean: {correct_model_data['mean_mse']:.2f}, "
                          f"Std: {correct_model_data['std_mse']:.2f}")
    stats_text_incorrect = (f"Mean: {incorrect_model_data['mean_mse']:.2f}, "
                            f"Std: {incorrect_model_data['std_mse']:.2f}")

    # Position the text on the plot
    plt.text(0.95, 0.95, stats_text_incorrect, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='orange', alpha=0.5))
             
    plt.text(0.95, 0.80, stats_text_correct, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='blue', alpha=0.5))

    plt.xlabel('Average Targeted MSE')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Model Errors over {num_trials} Trials')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(graphs_dir, 'error_distribution_histogram.png')
    plt.savefig(plot_path)
    print(f"Saved error distribution plot to: {plot_path}")
    plt.close()

    # --- Create and save the scatter plot ---
    plt.figure(figsize=(12, 7))
    plt.scatter(correct_model_data['epochs'], correct_model_data['mse_results'], alpha=0.7, label=f'Correct Model (Masking, {color_correct.capitalize()})', color=color_correct)
    plt.scatter(incorrect_model_data['epochs'], incorrect_model_data['mse_results'], alpha=0.7, label=f'Incorrect Model (Coercion, {color_incorrect.capitalize()})', marker='x', color=color_incorrect)
    
    # AI: Add trend lines
    if len(correct_model_data['epochs']) > 1:
        z = np.polyfit(correct_model_data['epochs'], correct_model_data['mse_results'], 1)
        p = np.poly1d(z)
        plt.plot(correct_model_data['epochs'], p(correct_model_data['epochs']), "--", color=color_correct)

    if len(incorrect_model_data['epochs']) > 1:
        z = np.polyfit(incorrect_model_data['epochs'], incorrect_model_data['mse_results'], 1)
        p = np.poly1d(z)
        plt.plot(incorrect_model_data['epochs'], p(incorrect_model_data['epochs']), "--", color=color_incorrect)
        
    plt.xlabel('Epoch at Best Validation Performance')
    plt.ylabel('Average Targeted MSE')
    plt.title(f'Model Error vs. Training Epochs over {num_trials} Trials')
    plt.legend()
    plt.grid(True)
    
    scatter_plot_path = os.path.join(graphs_dir, 'error_vs_epochs_scatter.png')
    plt.savefig(scatter_plot_path)
    print(f"Saved error vs. epochs scatter plot to: {scatter_plot_path}")
    plt.close()

    # --- Create and save the aggregated learning curve plots ---
    if 'all_learning_curves' in correct_model_data and correct_model_data['all_learning_curves']:
        # Plot for Hybrid Loss
        fig, ax = plt.subplots(figsize=(12, 7))
        plot_aggregated_learning_curves(ax, correct_model_data['all_learning_curves'], 'train_loss', color_correct, f'Correct Model - Train Loss ({color_correct.capitalize()})', linestyle='--')
        plot_aggregated_learning_curves(ax, correct_model_data['all_learning_curves'], 'val_loss', color_correct, f'Correct Model - Val Loss ({color_correct.capitalize()})')
        plot_aggregated_learning_curves(ax, incorrect_model_data['all_learning_curves'], 'train_loss', color_incorrect, f'Incorrect Model - Train Loss ({color_incorrect.capitalize()})', linestyle='--')
        plot_aggregated_learning_curves(ax, incorrect_model_data['all_learning_curves'], 'val_loss', color_incorrect, f'Incorrect Model - Val Loss ({color_incorrect.capitalize()})')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Hybrid Loss')
        ax.set_title(f'Aggregated Model Learning Curves over {num_trials} Trials')
        ax.legend()
        ax.grid(True)
        ax.set_yscale('log')
        
        learning_curve_path = os.path.join(graphs_dir, 'loss_learning_curves.png')
        plt.savefig(learning_curve_path)
        print(f"Saved aggregated loss learning curve plot to: {learning_curve_path}")
        plt.close()

        # Plot for Targeted MSE
        fig, ax = plt.subplots(figsize=(12, 7))
        plot_aggregated_learning_curves(ax, correct_model_data['all_learning_curves'], 'train_mse', color_correct, f'Correct Model - Train MSE ({color_correct.capitalize()})', linestyle='--')
        plot_aggregated_learning_curves(ax, correct_model_data['all_learning_curves'], 'val_mse', color_correct, f'Correct Model - Val MSE ({color_correct.capitalize()})')
        plot_aggregated_learning_curves(ax, incorrect_model_data['all_learning_curves'], 'train_mse', color_incorrect, f'Incorrect Model - Train MSE ({color_incorrect.capitalize()})', linestyle='--')
        plot_aggregated_learning_curves(ax, incorrect_model_data['all_learning_curves'], 'val_mse', color_incorrect, f'Incorrect Model - Val MSE ({color_incorrect.capitalize()})')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Targeted MSE')
        ax.set_title(f'Aggregated Model MSE Curves over {num_trials} Trials')
        ax.legend()
        ax.grid(True)
        ax.set_yscale('log')

        mse_curve_path = os.path.join(graphs_dir, 'mse_learning_curves.png')
        plt.savefig(mse_curve_path)
        print(f"Saved aggregated MSE learning curve plot to: {mse_curve_path}")
        plt.close()

    # --- Create and save the prediction distribution plot ---
    plot_prediction_distribution(data, graphs_dir)

if __name__ == "__main__":
    plot_results()
