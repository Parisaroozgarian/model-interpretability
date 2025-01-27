from typing import List, Dict, Optional
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from .base import BaseVisualizer

class AdvancedVisualizer(BaseVisualizer):
    """Advanced visualization techniques"""
    
    def plot_embedding_space(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None
    ) -> go.Figure:
        """Create 2D visualization of embedding space"""
        embeddings = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt")
                outputs = self.model(**inputs)
                embedding = outputs.logits.numpy().mean(axis=0)
                embeddings.append(embedding)
                
        embeddings = np.array(embeddings)
        
        fig = go.Figure(data=[
            go.Scatter(
                x=embeddings[:, 0],
                y=embeddings[:, 1],
                mode='markers+text',
                text=texts,
                marker=dict(
                    size=10,
                    color=labels if labels is not None else 'blue'
                )
            )
        ])
        
        fig.update_layout(
            title="2D Embedding Space",
            width=self.config.width,
            height=self.config.height
        )
        
        return fig 