from src.config.visualization_config import VisualizationConfig
from src.visualizers.feature_importance import FeatureImportanceVisualizer
from src.visualizers.uncertainty import UncertaintyVisualizer
import numpy as np

def main():
    # Initialize configuration
    config = VisualizationConfig()
    
    # Create visualizers
    feature_viz = FeatureImportanceVisualizer(config)
    uncertainty_viz = UncertaintyVisualizer(config)
    
    # Sample data
    tokens = ["This", "is", "a", "sample", "text"]
    importance_scores = np.random.random(5)
    uncertainties = {
        "predictive_entropy": np.random.random(10),
        "mutual_information": np.random.random(10)
    }
    
    # Create visualizations
    importance_fig = feature_viz.plot_token_importance(
        tokens=tokens,
        importance_scores=importance_scores,
        highlight_top_k=2
    )
    importance_fig.show()
    
    uncertainty_fig = uncertainty_viz.plot_uncertainty_landscape(
        texts=tokens,
        uncertainties=uncertainties
    )
    uncertainty_fig.show()

if __name__ == "__main__":
    main() 