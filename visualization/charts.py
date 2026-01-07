import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from utils.logger import setup_logger

logger = setup_logger("Charts")

def plot_benchmark_comparison(results: dict, output_dir: str = "outputs"):
    """
    Plots a side-by-side bar chart comparing SOTA vs New Models.
    
    Args:
        results (dict): Dictionary where keys are model names and values are dicts of metrics.
                        Example:
                        {
                            "ELECTRA-Base (Paper)": {"F1": 90.64, "Accuracy": 91.0, ...},
                            "Our-ELECTRA": {"F1": 91.2, "Accuracy": 91.5, ...},
                            "LLM-QLoRA": {"F1": 92.5, "Accuracy": 93.0, ...}
                        }
        output_dir (str): Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Flatten data for dataframe
    data = []
    for model, metrics in results.items():
        for metric, value in metrics.items():
            data.append({"Model": model, "Metric": metric, "Score": value})
            
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    chart = sns.barplot(data=df, x="Metric", y="Score", hue="Model", palette="viridis")
    plt.title("Performance Comparison: Paper SOTA vs SentiTR-LLM", fontsize=16)
    plt.ylim(80, 100) # Assuming high scores, adjust as needed
    plt.ylabel("Score (%)")
    plt.xlabel("")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    for container in chart.containers:
        chart.bar_label(container, fmt='%.2f')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark_comparison.png"))
    plt.close()
    logger.info(f"Benchmark plot saved to {output_dir}/benchmark_comparison.png")

def plot_data_distribution(dataset, output_dir: str = "outputs"):
    """
    Plots word length distribution of the dataset to check augmentation effects.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract text from HF dataset
    texts = dataset['train']['text'] # Assuming 'train' split has the full augmented data or typical dist
    
    word_counts = [len(text.split()) for text in texts]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(word_counts, bins=50, kde=True, color="skyblue")
    plt.title("Word Length Distribution (Augmented Dataset)", fontsize=16)
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "word_length_distribution.png"))
    plt.close()
    logger.info(f"Data distribution plot saved to {output_dir}/word_length_distribution.png")
