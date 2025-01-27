from typing import List, Dict, Optional, Tuple
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .base import BaseVisualizer

class DeepModelAnalyzer(BaseVisualizer):
    """Deep analysis of model behavior and decision-making process"""
    
    def analyze_layer_dynamics(
        self,
        text: str,
        layer_range: Optional[Tuple[int, int]] = None
    ) -> go.Figure:
        """Analyze how representations evolve through layers"""
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs, output_hidden_states=True, output_attentions=True)
        hidden_states = outputs.hidden_states
        
        # Get layer-wise statistics
        layer_stats = []
        for layer_idx in range(len(hidden_states)):
            layer_output = hidden_states[layer_idx][0].detach().numpy()
            layer_stats.append({
                'mean': np.mean(layer_output),
                'std': np.std(layer_output),
                'norm': np.linalg.norm(layer_output),
                'sparsity': np.mean(layer_output == 0)
            })
            
        # Create multi-metric visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Mean Activation", "Standard Deviation", 
                          "L2 Norm", "Sparsity"]
        )
        
        x = list(range(len(layer_stats)))
        metrics = ['mean', 'std', 'norm', 'sparsity']
        positions = [(1,1), (1,2), (2,1), (2,2)]
        
        for metric, pos in zip(metrics, positions):
            fig.add_trace(
                go.Scatter(
                    x=x, 
                    y=[s[metric] for s in layer_stats],
                    name=metric.capitalize(),
                    mode="lines+markers"
                ),
                row=pos[0], col=pos[1]
            )
            
        fig.update_layout(
            title="Layer-wise Representation Dynamics",
            height=800,
            showlegend=True
        )
        
        return fig 