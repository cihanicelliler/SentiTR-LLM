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
        
        # Auto-detect target modules based on model architecture
        # BERT/RoBERTa/ELECTRA: query, value
        # LLaMA/Mistral/GPT: q_proj, v_proj
        target_modules = self._detect_target_modules()
        
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=target_modules
        )
        
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def _detect_target_modules(self):
        """Auto-detect LoRA target modules based on model architecture."""
        # Get all module names from the model
        module_names = [name for name, _ in self.model.named_modules()]
        module_names_str = " ".join(module_names)
        
        # BERT/RoBERTa/ELECTRA style (encoder models)
        if "query" in module_names_str and "value" in module_names_str:
            return ["query", "value"]
        # LLaMA/Mistral/GPT-NeoX style (decoder models)
        elif "q_proj" in module_names_str and "v_proj" in module_names_str:
            return ["q_proj", "v_proj"]
        # GPT-2/GPT-J style
        elif "c_attn" in module_names_str:
            return ["c_attn"]
        # Fallback: try common patterns
        else:
            # Search for attention-related modules
            for name in module_names:
                if "query" in name.lower():
                    return ["query", "value"]
                if "q_proj" in name.lower():
                    return ["q_proj", "v_proj"]
            # Ultimate fallback
            return ["query", "value"]

    def forward(self, **kwargs):
        return self.model(**kwargs)

