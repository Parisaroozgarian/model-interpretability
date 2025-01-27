from typing import List, Dict, Optional
import plotly.graph_objects as go
import numpy as np
import umap
from .base import BaseVisualizer
import torch

class UncertaintyVisualizer(BaseVisualizer):
    """Visualizer for uncertainty analysis"""
    
    def plot_uncertainty_landscape(
        self,
        texts: List[str],
        uncertainties: Dict[str, List[float]]
    ) -> go.Figure:
        """Create 3D visualization of uncertainty landscape"""
        # Get embeddings and ensure proper shape
        text_embeddings = self._get_text_embeddings(texts)
        
        if text_embeddings.ndim == 1:
            text_embeddings = text_embeddings.reshape(-1, 1)
        
        if text_embeddings.shape[1] == 1:
            text_embeddings = np.hstack([text_embeddings, np.zeros_like(text_embeddings)])
            
        # Adjust n_neighbors based on dataset size
        n_neighbors = min(30, len(texts) - 1)
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors)
        reduced_embeddings = reducer.fit_transform(text_embeddings)
        
        # Create 3D scatter plot
        uncertainty_values = np.array(uncertainties["predictive_entropy"])
        
        fig = go.Figure(data=[
            go.Scatter3d(
                x=reduced_embeddings[:, 0],
                y=reduced_embeddings[:, 1],
                z=uncertainty_values,
                mode='markers',
                marker=dict(
                    size=10,
                    color=uncertainty_values,
                    colorscale=self.config.colorscale
                )
            )
        ])
        
        fig.update_layout(
            title="Uncertainty Landscape",
            scene=dict(
                xaxis_title="Embedding Dimension 1",
                yaxis_title="Embedding Dimension 2",
                zaxis_title="Uncertainty"
            ),
            width=self.config.width,
            height=self.config.height
        )
        
        return fig 

    def _get_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for input texts"""
        embeddings = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                outputs = self.model(**inputs)
                embedding = outputs.logits.numpy().mean(axis=0)
                embeddings.append(embedding)
        return np.array(embeddings) 