from dataclasses import dataclass
from typing import Optional

@dataclass
class VisualizationConfig:
    """Configuration for visualization settings"""
    width: int = 800
    height: int = 600
    theme: str = "plotly"
    colorscale: str = "Viridis"
    show_legend: bool = True

@dataclass
class AdvancedVisualizationConfig(VisualizationConfig):
    """Enhanced configuration with more visualization options"""
    plot_style: str = "darkgrid"
    color_palette: str = "husl"
    font_scale: float = 1.2
    interactive: bool = True 