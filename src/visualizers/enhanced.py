from typing import List, Dict, Optional
import plotly.graph_objects as go
from .base import BaseVisualizer

class EnhancedVisualizer(BaseVisualizer):
    """Enhanced visualization capabilities"""
    
    def plot_attention_flow(
        self,
        text: str,
        layer_idx: int = -1
    ) -> go.Figure:
        """Visualize attention flow in the model"""
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs, output_attentions=True)
        attention = outputs.attentions[layer_idx][0].mean(dim=0).detach().numpy()
        
        fig = go.Figure(data=[
            go.Heatmap(
                z=attention,
                x=self.tokenizer.tokenize(text),
                y=self.tokenizer.tokenize(text)
            )
        ])
        
        fig.update_layout(
            title=f"Attention Pattern - Layer {layer_idx}",
            width=self.config.width,
            height=self.config.height
        )
        
        return fig 