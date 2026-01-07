import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)

class SentimentLLM(torch.nn.Module):
    """
    Wrapper for LLM-based Sentiment Analysis using LoRA/QLoRA.
    """
    def __init__(self, model_name: str, num_labels: int = 3, use_4bit: bool = True):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Quantization Config
        bnb_config = None
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right" # Fix for fp16

        # Load Model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Prepare for int8/4 training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        # Target modules depend on architecture (Llama, Bloom, etc.)
        # Defaulting to common Llama targets
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, # or SEQ_CLS if using classification head directly
            inference_mode=False, 
            r=8, 
            lora_alpha=32, 
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"] # Adjust based on specific model
        )
        
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
        # Note: For Sequence Classification with LLMs, typically we generate text 
        # OR we add a classification head. The prompt asks for "Sequence Classification".
        # Standard PEFT approach often keeps CausalLM and generates labels, OR uses AutoModelForSequenceClassification.
        # But QLoRA usually applied to base models. 
        # If we strictly want Classification Head + LoRA, we should've used AutoModelForSequenceClassification.
        # Let's adjust to support SequenceClassification if possible, 
        # BUT many large models (7B) are easier loaded as CausalLM.
        # For simplicity and "SOTA" standard usually implies generation for LLMs or
        # using header. Attempting AutoModelForSequenceClassification with QLoRA is cleaner for direct F1 comparison.
        
    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    # helper to save
    def save_pretrained(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

# Alternate implementation for Strict Sequence Classification (preferred for direct comparison)
class SentimentLLMClassifier(torch.nn.Module):
    def __init__(self, model_name: str, num_labels: int = 3, use_4bit: bool = True):
        super().__init__()
        
        bnb_config = None
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16 # bfloat16 if ampere
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        self.model = prepare_model_for_kbit_training(self.model)
        
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]
        )
        
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def forward(self, **kwargs):
        return self.model(**kwargs)

