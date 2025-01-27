from typing import Optional, List, Dict
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ..config.visualization_config import VisualizationConfig
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class BaseVisualizer:
    """Base class for all visualizers"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=2,
            ignore_mismatched_sizes=True,
            attn_implementation="eager"
        ) 