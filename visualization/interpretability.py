import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from utils.logger import setup_logger

logger = setup_logger("Interpretability")

def explain_with_shap(model, tokenizer, texts, output_dir="outputs"):
    """
    Generates SHAP plots for a subset of texts to explain model predictions.
    Works best with Encoder models (BERT/ELECTRA).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Wrapper for SHAP
    # SHAP expects a function that takes a list of strings and returns a tensor of probabilities
    def f(x):
        inputs = tokenizer(x.tolist(), return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs.cpu().numpy()

    # Define a masker
    masker = shap.maskers.Text(tokenizer)
    
    # Create the explainer
    explainer = shap.Explainer(f, masker, output_names=["Negative", "Neutral", "Positive"])
    
    # Explain
    logger.info(f"Running SHAP on {len(texts)} samples...")
    shap_values = explainer(texts)
    
    # Save plots
    # SHAP plots are tricky to save automatically as standard images, usually they are HTML/JS.
    # We can save the force plot or waterfall plot for the first instance.
    
    # Force Plot for the first example
    # Note: matplotlib=True support is limited in newer SHAP, depends on version.
    # Saving as HTML is safer.
    
    try:
        html = shap.plots.text(shap_values[0], display=False)
        with open(os.path.join(output_dir, "shap_explanation.html"), "w", encoding="utf-8") as file:
             file.write(html)
        logger.info(f"SHAP explanation saved to {output_dir}/shap_explanation.html")
    except Exception as e:
        logger.error(f"Failed to generate/save SHAP plot: {e}")

def visualize_attention(model, tokenizer, text, output_dir="outputs"):
    """
    Visualizes attention weights for high-impact words.
    Requires model to output attentions.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    inputs = tokenizer(text, return_tensors="pt", output_attentions=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get attentions: list of (batch_size, num_heads, sequence_length, sequence_length)
    attentions = outputs.attentions 
    if not attentions:
        logger.warning("No attentions found in model output. Ensure config.output_attentions=True")
        return

    # Take the last layer's attention
    last_layer_attn = attentions[-1][0] # (num_heads, seq_len, seq_len)
    
    # Average across heads
    avg_attn = torch.mean(last_layer_attn, dim=0).cpu().numpy() # (seq_len, seq_len)
    
    # Tokenize manually to line up
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    
    # Plot heatmap for the CLS token (or overall sentence attention)
    # Often we want to see what tokens the CLS token attended to.
    
    cls_attn = avg_attn[0, :] # Attention from [CLS] to all other tokens
    
    plt.figure(figsize=(10, 2))
    sns.heatmap([cls_attn], xticklabels=tokens, yticklabels=['[CLS]'], cmap="viridis", annot=False)
    plt.title(f"Attention Map (Last Layer Avg Head): {text[:30]}...")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "attention_map.png"))
    plt.close()
    logger.info(f"Attention map saved to {output_dir}/attention_map.png")
