import argparse
import sys
from data.preprocessor import load_and_preprocess_data
from training.fine_tuner import train_encoder_model, train_llm_model
from utils.logger import setup_logger

logger = setup_logger("Main")

def main():
    parser = argparse.ArgumentParser(description="SentiTR-LLM: Turkish Sentiment Analysis")
    
    # Task selection
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate"], help="Mode: train or evaluate")
    parser.add_argument("--model_type", type=str, default="encoder", choices=["encoder", "llm"], help="Type of model to use")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="dbmdz/bert-base-turkish-cased", help="HuggingFace model name")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization for LLMs")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    
    # Data configuration
    parser.add_argument("--test_size", type=float, default=0.1, help="Test set size")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation set size")
    
    # Visualization flag
    parser.add_argument("--visualize", action="store_true", help="Generate advanced visualizations after training/eval")
    
    args = parser.parse_args()
    
    logger.info(f"Starting SentiTR-LLM with args: {args}")
    
    # Load Data
    try:
        dataset = load_and_preprocess_data(test_size=args.test_size, val_size=args.val_size)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
        
    trainer = None
    if args.mode == "train":
        if args.model_type == "encoder":
            trainer, tokenized_dataset = train_encoder_model(args, dataset)
        elif args.model_type == "llm":
            trainer, tokenized_dataset = train_llm_model(args, dataset)
            
        logger.info("Training finished.")
        
    elif args.mode == "evaluate":
        # Placeholder for evaluation-only mode loading saved model
        logger.info("Evaluation mode not fully implemented in main.py yet. Use train mode to train and eval.")
    
    if args.visualize:
        logger.info("Generating visualizations...")
        from visualization.charts import plot_data_distribution, plot_benchmark_comparison
        from visualization.analysis import plot_confusion_matrix, analyze_errors
        from visualization.interpretability import explain_with_shap, visualize_attention
        import numpy as np
        
        # 1. Data Distribution
        plot_data_distribution(dataset)
        
        # 2. Benchmark Comparison (Mock data for demo if not full benchmark run)
        # In a real run, we would aggregate results from multiple runs
        mock_results = {
            "Paper SOTA (ELECTRA)": {"F1": 90.64, "Accuracy": 91.0},
            "SentiTR-LLM (Current)": {"F1": 0.0, "Accuracy": 0.0} # Placeholder
        }
        
        # Determine current model metrics if available
        if trainer:
             # Predict on test set
             output = trainer.predict(tokenized_dataset["test"])
             metrics = output.metrics
             # extract f1 and accuracy
             mock_results["SentiTR-LLM (Current)"]["F1"] = metrics.get("test_f1", 0) * 100
             mock_results["SentiTR-LLM (Current)"]["Accuracy"] = metrics.get("test_accuracy", 0) * 100
             
             # 3. Confusion Matrix
             y_pred = np.argmax(output.predictions, axis=1) if not isinstance(output.predictions, tuple) else np.argmax(output.predictions[0], axis=1)
             y_true = output.label_ids
             plot_confusion_matrix(y_true, y_pred, labels=[0, 1, 2], class_names=["Negative", "Neutral", "Positive"])
             
             # 4. Error Analysis
             # Need probabilities
             logits = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
             import torch
             probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
             analyze_errors(y_true, probs, dataset["test"]["text"])

             # 5. Interpretability (Subset)
             if args.model_type == "encoder":
                 # SHAP
                 explain_with_shap(trainer.model, trainer.tokenizer, dataset["test"]["text"][:5])
             elif args.model_type == "llm":
                 # Attention Map
                 visualize_attention(trainer.model, trainer.tokenizer, dataset["test"]["text"][0])

        plot_benchmark_comparison(mock_results)
        logger.info("Visualizations completed.")
        
if __name__ == "__main__":
    main()
