import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from utils.logger import setup_logger

logger = setup_logger("Analysis")

def plot_confusion_matrix(y_true, y_pred, labels, class_names=None, title="Confusion Matrix", output_dir="outputs"):
    """
    Generates and saves a normalized confusion matrix heatmap.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true') # Normalize by row (true class)
    
    if class_names is None:
        class_names = [str(l) for l in labels]
        
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{title.replace(' ', '_')}.png"))
    plt.close()
    logger.info(f"Confusion matrix saved to {output_dir}")

def analyze_errors(y_true, y_probs, texts, output_dir="outputs", top_k=10):
    """
    Identifies and visualizes the 'hardest' examples where the model was confident but wrong, 
    or low confidence correct predictions.
    
    y_probs: probability vectors (N, C)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    y_pred = np.argmax(y_probs, axis=1)
    confidences = np.max(y_probs, axis=1)
    
    # Create DataFrame for analysis
    df = pd.DataFrame({
        "text": texts,
        "true": y_true,
        "pred": y_pred,
        "confidence": confidences
    })
    
    # 1. High Confidence Errors: Model is sure but wrong
    errors = df[df["true"] != df["pred"]]
    high_conf_errors = errors.sort_values(by="confidence", ascending=False).head(top_k)
    
    if not high_conf_errors.empty:
        high_conf_errors.to_csv(os.path.join(output_dir, "high_confidence_errors.csv"), index=False)
        logger.info(f"High confidence errors saved to {output_dir}/high_confidence_errors.csv")
        
        # Visualize simple table
        fig, ax = plt.subplots(figsize=(12, len(high_conf_errors) * 0.5 + 1))
        ax.axis('tight')
        ax.axis('off')
        table_data = high_conf_errors[["text", "true", "pred", "confidence"]]
        # Truncate text
        table_data["text"] = table_data["text"].apply(lambda x: x[:50] + "..." if len(x) > 50 else x)
        
        table = ax.table(cellText=table_data.values, colLabels=table_data.columns, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        plt.title(f"Top {top_k} High Confidence Errors")
        plt.savefig(os.path.join(output_dir, "error_analysis_table.png"))
        plt.close()
