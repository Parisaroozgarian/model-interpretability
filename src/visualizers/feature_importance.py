from typing import List, Dict, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from .base import BaseVisualizer

class FeatureImportanceVisualizer(BaseVisualizer):
    """Visualizer for feature importance analysis"""
    
    def plot_token_importance(
        self,
        tokens: List[str],
        importance_scores: List[float],
        highlight_top_k: int = 5
    ) -> go.Figure:
        """Visualize token importance scores"""
        # Sort tokens by importance
        sorted_indices = np.argsort(importance_scores)
        top_k_indices = sorted_indices[-highlight_top_k:]
        
        # Create bar plot
        fig = go.Figure()
        
        # Add bars
        fig.add_trace(
            go.Bar(
                x=[tokens[i] for i in range(len(tokens))],
                y=[importance_scores[i] for i in range(len(tokens))],
                marker_color=['rgba(55, 128, 191, 0.7)' if i not in top_k_indices 
                            else 'rgba(219, 64, 82, 0.7)'
                            for i in range(len(tokens))]
            )
        )
        
        fig.update_layout(
            title="Token Importance Scores",
            xaxis_title="Tokens",
            yaxis_title="Importance Score",
            width=self.config.width,
            height=self.config.height
        )
        
        return fig 