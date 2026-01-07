import numpy as np
import torch
import torch.nn.functional as F
from utils.logger import setup_logger

logger = setup_logger("Ensemble")

class WeightedEnsemble:
    """
    Weighted Ensemble of multiple models.
    """
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights else [1.0/len(models)] * len(models)
        
        assert len(self.models) == len(self.weights), "Number of weights must match number of models."

    def predict(self, dataset):
        """
        Generates weighted average predictions.
        Assumes models are HF Trainer compatible or return logits.
        This is a simplified version; in production, you'd run inference batch-wise.
        """
        all_logits = []
        
        for model in self.models:
            # Placeholder: In a real scenario, you'd use a Trainer or a Predictor loop
            # to get logits for the whole dataset.
            # For this script, we assume 'model' is a trainer or predictor object
            # capable of .predict(dataset) returning (logits, labels, metrics)
            output = model.predict(dataset)
            logits = output.predictions
            
            # If tuple (logits, hidden_states), take logits
            if isinstance(logits, tuple):
                logits = logits[0]
                
            probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
            all_logits.append(probs)
            
        # Weighted sum of probabilities
        weighted_probs = np.zeros_like(all_logits[0])
        for i, probs in enumerate(all_logits):
            weighted_probs += self.weights[i] * probs
            
        final_preds = np.argmax(weighted_probs, axis=1)
        return final_preds

if __name__ == "__main__":
    logger.info("Ensemble module ready.")
