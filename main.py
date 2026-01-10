import argparse
import sys
from data.preprocessor import load_and_preprocess_data
from training.fine_tuner import train_encoder_model, train_llm_model
from utils.logger import setup_logger

logger = setup_logger("Main")

def run_ensemble(args, dataset):
    """Run ensemble prediction combining Encoder and LLM models."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import Trainer, TrainingArguments
    from models.model_loader import load_encoder_from_checkpoint, load_llm_from_checkpoint
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    logger.info("=== Starting Ensemble Mode ===")
    
    # Load models from checkpoints
    logger.info(f"Loading Encoder from: {args.encoder_checkpoint}")
    encoder_model, encoder_tokenizer = load_encoder_from_checkpoint(args.encoder_checkpoint)
    
    logger.info(f"Loading LLM from: {args.llm_checkpoint}")
    llm_model, llm_tokenizer = load_llm_from_checkpoint(args.llm_checkpoint)
    
    # Tokenize test set for both models
    def tokenize_encoder(examples):
        return encoder_tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    
    def tokenize_llm(examples):
        return llm_tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    
    test_data_encoder = dataset["test"].map(tokenize_encoder, batched=True)
    test_data_llm = dataset["test"].map(tokenize_llm, batched=True)
    
    # Create trainers for prediction
    training_args = TrainingArguments(
        output_dir="./tmp_ensemble",
        per_device_eval_batch_size=args.batch_size,
        report_to="none"
    )
    
    # Get predictions from Encoder
    logger.info("Getting Encoder predictions...")
    encoder_trainer = Trainer(
        model=encoder_model,
        args=training_args,
        processing_class=encoder_tokenizer
    )
    encoder_output = encoder_trainer.predict(test_data_encoder)
    encoder_logits = encoder_output.predictions
    if isinstance(encoder_logits, tuple):
        encoder_logits = encoder_logits[0]
    encoder_probs = F.softmax(torch.tensor(encoder_logits), dim=-1).numpy()
    logger.info(f"Encoder standalone accuracy: {accuracy_score(encoder_output.label_ids, np.argmax(encoder_logits, axis=1)):.4f}")
    
    # Get predictions from LLM
    logger.info("Getting LLM predictions...")
    llm_trainer = Trainer(
        model=llm_model,
        args=training_args,
        processing_class=llm_tokenizer
    )
    llm_output = llm_trainer.predict(test_data_llm)
    llm_logits = llm_output.predictions
    if isinstance(llm_logits, tuple):
        llm_logits = llm_logits[0]
    llm_probs = F.softmax(torch.tensor(llm_logits), dim=-1).numpy()
    logger.info(f"LLM standalone accuracy: {accuracy_score(llm_output.label_ids, np.argmax(llm_logits, axis=1)):.4f}")
    
    # Weighted ensemble
    encoder_weight = args.encoder_weight
    llm_weight = args.llm_weight
    logger.info(f"Ensemble weights - Encoder: {encoder_weight}, LLM: {llm_weight}")
    
    ensemble_probs = encoder_weight * encoder_probs + llm_weight * llm_probs
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    y_true = encoder_output.label_ids
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, ensemble_preds)
    f1 = f1_score(y_true, ensemble_preds, average='weighted')
    precision = precision_score(y_true, ensemble_preds, average='weighted')
    recall = recall_score(y_true, ensemble_preds, average='weighted')
    
    logger.info("=" * 50)
    logger.info("ENSEMBLE RESULTS")
    logger.info("=" * 50)
    logger.info(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info("=" * 50)
    
    # Compare with SOTA
    sota_f1 = 90.64
    if f1 * 100 > sota_f1:
        logger.info(f"🎉 SOTA BEATEN! Ensemble F1 ({f1*100:.2f}%) > Paper SOTA ({sota_f1}%)")
    else:
        logger.info(f"📊 Ensemble F1: {f1*100:.2f}% vs Paper SOTA: {sota_f1}%")
    
    return ensemble_preds, y_true, ensemble_probs


def main():
    parser = argparse.ArgumentParser(description="SentiTR-LLM: Turkish Sentiment Analysis")
    
    # Task selection
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate", "ensemble"], help="Mode: train, evaluate, or ensemble")
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
    
    # Ensemble configuration
    parser.add_argument("--encoder_checkpoint", type=str, default="checkpoints/bert-base-turkish-cased", help="Path to encoder checkpoint")
    parser.add_argument("--llm_checkpoint", type=str, default="checkpoints/turkish-gpt2-large", help="Path to LLM checkpoint")
    parser.add_argument("--encoder_weight", type=float, default=0.6, help="Weight for encoder model in ensemble")
    parser.add_argument("--llm_weight", type=float, default=0.4, help="Weight for LLM model in ensemble")
    
    args = parser.parse_args()
    
    logger.info(f"Starting SentiTR-LLM with args: {args}")
    
    # Load Data
    try:
        dataset = load_and_preprocess_data(test_size=args.test_size, val_size=args.val_size)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
        
    trainer = None
    tokenized_dataset = None
    ensemble_results = None
    
    if args.mode == "train":
        if args.model_type == "encoder":
            trainer, tokenized_dataset = train_encoder_model(args, dataset)
        elif args.model_type == "llm":
            trainer, tokenized_dataset = train_llm_model(args, dataset)
            
        logger.info("Training finished.")
        
    elif args.mode == "evaluate":
        logger.info("Evaluation mode not fully implemented in main.py yet. Use train mode to train and eval.")
    
    elif args.mode == "ensemble":
        ensemble_preds, y_true, ensemble_probs = run_ensemble(args, dataset)
        ensemble_results = (ensemble_preds, y_true, ensemble_probs)
    
    if args.visualize:
        logger.info("Generating visualizations...")
        from visualization.charts import plot_data_distribution, plot_benchmark_comparison
        from visualization.analysis import plot_confusion_matrix, analyze_errors
        from visualization.interpretability import explain_with_shap, visualize_attention
        import numpy as np
        
        # 1. Data Distribution
        plot_data_distribution(dataset)
        
        # 2. Benchmark Comparison
        mock_results = {
            "Paper SOTA (ELECTRA)": {"F1": 90.64, "Accuracy": 91.0},
            "SentiTR-LLM (Current)": {"F1": 0.0, "Accuracy": 0.0}
        }
        
        if args.mode == "ensemble" and ensemble_results:
            ensemble_preds, y_true, ensemble_probs = ensemble_results
            from sklearn.metrics import accuracy_score, f1_score
            mock_results["SentiTR-LLM (Current)"]["F1"] = f1_score(y_true, ensemble_preds, average='weighted') * 100
            mock_results["SentiTR-LLM (Current)"]["Accuracy"] = accuracy_score(y_true, ensemble_preds) * 100
            
            # Confusion Matrix for ensemble
            plot_confusion_matrix(y_true, ensemble_preds, labels=[0, 1, 2], class_names=["Negative", "Neutral", "Positive"])
            
            # Error Analysis
            analyze_errors(y_true, ensemble_probs, dataset["test"]["text"])
            
        elif trainer and tokenized_dataset:
            output = trainer.predict(tokenized_dataset["test"])
            metrics = output.metrics
            mock_results["SentiTR-LLM (Current)"]["F1"] = metrics.get("test_f1", 0) * 100
            mock_results["SentiTR-LLM (Current)"]["Accuracy"] = metrics.get("test_accuracy", 0) * 100
             
            y_pred = np.argmax(output.predictions, axis=1) if not isinstance(output.predictions, tuple) else np.argmax(output.predictions[0], axis=1)
            y_true = output.label_ids
            plot_confusion_matrix(y_true, y_pred, labels=[0, 1, 2], class_names=["Negative", "Neutral", "Positive"])
             
            logits = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
            import torch
            probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
            analyze_errors(y_true, probs, dataset["test"]["text"])

            if args.model_type == "encoder":
                explain_with_shap(trainer.model, trainer.tokenizer, dataset["test"]["text"][:5])
            elif args.model_type == "llm":
                visualize_attention(trainer.model, trainer.tokenizer, dataset["test"]["text"][0])

        plot_benchmark_comparison(mock_results)
        logger.info("Visualizations completed.")
        
if __name__ == "__main__":
    main()

