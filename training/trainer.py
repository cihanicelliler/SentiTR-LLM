import torch
from transformers import Trainer, TrainingArguments
from evaluation.metrics import compute_metrics
from utils.logger import setup_logger
import os

logger = setup_logger("Trainer")

class SentiTRTrainer:
    """
    Wrapper around Hugging Face Trainer for consistent training setup.
    """
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        output_dir: str = "checkpoints",
        batch_size: int = 16,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        evaluation_strategy: str = "epoch",
        save_strategy: str = "epoch",
        fp16: bool = False, # Set True if GPU supports it
    ):
        self.output_dir = output_dir
        self.model = model
        self.tokenizer = tokenizer
        
        self.args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500, # Default warmup
            weight_decay=weight_decay,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=100,
            eval_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            learning_rate=learning_rate,
            fp16=fp16,
            save_total_limit=2,
            remove_unused_columns=True # Important: remove 'text' column so collator doesn't fail
        )
        
        self.trainer = Trainer(
            model=model,
            args=self.args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        
    def train(self):
        logger.info("Starting training...")
        train_result = self.trainer.train()
        metrics = train_result.metrics
        self.trainer.save_model()
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()
        logger.info(f"Training completed. Metrics: {metrics}")
        return metrics
    
    def evaluate(self):
        logger.info("Starting evaluation...")
        metrics = self.trainer.evaluate()
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)
        logger.info(f"Evaluation completed. Metrics: {metrics}")
        return metrics

    def predict(self, dataset):
        return self.trainer.predict(dataset)
