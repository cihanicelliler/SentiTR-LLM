"""
Model Loader Utilities for loading trained checkpoints.
FIXED: Don't use merge_and_unload() as it doesn't properly handle modules_to_save.
Instead, return PeftModel directly for inference.
"""
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from peft import PeftModel, PeftConfig
from utils.logger import setup_logger

logger = setup_logger("ModelLoader")


def load_encoder_from_checkpoint(checkpoint_path: str):
    """
    Load an Encoder model (BERT/ELECTRA) from checkpoint.
    Returns PeftModel directly (no merge_and_unload) to preserve modules_to_save weights.
    """
    logger.info(f"Loading Encoder from: {checkpoint_path}")
    
    adapter_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
    
    if os.path.exists(adapter_path):
        # Load PEFT adapter
        try:
            peft_config = PeftConfig.from_pretrained(checkpoint_path)
            logger.info(f"Loading base model from: {peft_config.base_model_name_or_path}")
            base_model = AutoModelForSequenceClassification.from_pretrained(
                peft_config.base_model_name_or_path,
                num_labels=3
            )
            model = PeftModel.from_pretrained(base_model, checkpoint_path)
            # DON'T merge_and_unload - it doesn't handle modules_to_save properly
            # Just return PeftModel directly
            logger.info("Loaded PEFT adapter (keeping as PeftModel for inference)")
        except Exception as e:
            logger.error(f"Failed to load PEFT model: {e}")
            raise
    else:
        # Try direct load
        try:
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
            logger.info("Loaded full model checkpoint")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    return model, tokenizer


def load_llm_from_checkpoint(checkpoint_path: str):
    """
    Load an LLM model (GPT-2/LLaMA) from checkpoint.
    """
    logger.info(f"Loading LLM from: {checkpoint_path}")
    
    adapter_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
    
    if os.path.exists(adapter_path):
        # PEFT adapter exists
        try:
            peft_config = PeftConfig.from_pretrained(checkpoint_path)
            logger.info(f"Loading base model from: {peft_config.base_model_name_or_path}")
            base_model = AutoModelForSequenceClassification.from_pretrained(
                peft_config.base_model_name_or_path,
                num_labels=3,
                device_map="auto"
            )
            model = PeftModel.from_pretrained(base_model, checkpoint_path)
            # DON'T merge_and_unload - keep PeftModel for inference
            logger.info("Loaded PEFT adapter (keeping as PeftModel for inference)")
        except Exception as e:
            logger.error(f"Failed to load PEFT model: {e}")
            raise
    else:
        # Try direct load  
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint_path,
                device_map="auto"
            )
            logger.info("Loaded full model checkpoint")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def get_device():
    """Get the appropriate device for inference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
