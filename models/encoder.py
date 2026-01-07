import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

class SentimentEncoder(nn.Module):
    """
    Wrapper for Encoder-based models (BERT, ELECTRA) for Sentiment Analysis.
    """
    def __init__(self, model_name: str, num_labels: int = 3):
        super(SentimentEncoder, self).__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    def save_pretrained(self, save_directory):
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        
    @classmethod
    def from_pretrained(cls, load_directory):
        # Logic to reload a saved custom wrapper might need config handling
        # For simplicity, we can reload properties from the model config
        config = AutoConfig.from_pretrained(load_directory)
        return cls(config._name_or_path, num_labels=config.num_labels)

if __name__ == "__main__":
    # Test initialization
    try:
        model = SentimentEncoder("dbmdz/electra-base-turkish-cased-discriminator")
        print("Encoder initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize encoder: {e}")
