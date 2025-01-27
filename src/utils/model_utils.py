from typing import List, Dict, Optional
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def get_model_predictions(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str]
) -> np.ndarray:
    """Get model predictions for a list of texts"""
    predictions = []
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions.append(probs[0].numpy())
            
    return np.array(predictions)

def calculate_uncertainty_metrics(
    predictions: np.ndarray,
    num_samples: int = 10
) -> Dict[str, List[float]]:
    """Calculate various uncertainty metrics from predictions"""
    entropy = -np.sum(predictions * np.log(predictions + 1e-10), axis=1)
    confidence = np.max(predictions, axis=1)
    
    return {
        "predictive_entropy": entropy.tolist(),
        "confidence": confidence.tolist(),
        "mutual_information": (entropy * (1 - confidence)).tolist()
    } 