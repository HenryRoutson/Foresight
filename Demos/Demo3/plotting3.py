import json
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_results():
    """
    Loads experiment results from results.json and generates plots.
    """
    results_path = os.path.join(os.path.dirname(__file__), 'results.json')
    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}")
        print("Please run the experiment script (models3.py) first to generate the results.")
        return

    with open(results_path, 'r') as f:
        data = json.load(f)

    correct_model_data = data['correct_model']
    incorrect_model_data = data['incorrect_model']
    num_trials = data['num_trials']

    # --- Create and save the histogram ---
    plt.figure(figsize=(12, 7))
    plt.hist(correct_model_data['mse_results'], bins=10, alpha=0.7, label='Correct Model (Masking)')
    plt.hist(incorrect_model_data['mse_results'], bins=10, alpha=0.7, label='Incorrect Model (Coercion)')
    
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
    
    plot_path = os.path.join(os.path.dirname(__file__), 'error_distribution_histogram.png')
    plt.savefig(plot_path)
    print(f"Saved error distribution plot to: {plot_path}")
    plt.close()

    # --- Create and save the scatter plot ---
    plt.figure(figsize=(12, 7))
    plt.scatter(correct_model_data['epochs'], correct_model_data['mse_results'], alpha=0.7, label='Correct Model (Masking)')
    plt.scatter(incorrect_model_data['epochs'], incorrect_model_data['mse_results'], alpha=0.7, label='Incorrect Model (Coercion)', marker='x')
    
    # AI: Add trend lines
    if len(correct_model_data['epochs']) > 1:
        z = np.polyfit(correct_model_data['epochs'], correct_model_data['mse_results'], 1)
        p = np.poly1d(z)
        plt.plot(correct_model_data['epochs'], p(correct_model_data['epochs']), "--", color='blue')

    if len(incorrect_model_data['epochs']) > 1:
        z = np.polyfit(incorrect_model_data['epochs'], incorrect_model_data['mse_results'], 1)
        p = np.poly1d(z)
        plt.plot(incorrect_model_data['epochs'], p(incorrect_model_data['epochs']), "--", color='orange')
        
    plt.xlabel('Epoch at Best Validation Performance')
    plt.ylabel('Average Targeted MSE')
    plt.title(f'Model Error vs. Training Epochs over {num_trials} Trials')
    plt.legend()
    plt.grid(True)
    
    scatter_plot_path = os.path.join(os.path.dirname(__file__), 'error_vs_epochs_scatter.png')
    plt.savefig(scatter_plot_path)
    print(f"Saved error vs. epochs scatter plot to: {scatter_plot_path}")
    plt.close()

if __name__ == "__main__":
    plot_results()
