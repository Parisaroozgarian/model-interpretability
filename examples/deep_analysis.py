from src.config.visualization_config import AdvancedVisualizationConfig
from src.visualizers.deep_analysis import DeepModelAnalyzer
from src.utils.model_utils import get_model_predictions, calculate_uncertainty_metrics

def main():
    # Initialize configuration and analyzer
    config = AdvancedVisualizationConfig()
    analyzer = DeepModelAnalyzer(config)
    
    # Analyze layer dynamics
    text = "This is an example of deep model analysis"
    dynamics_fig = analyzer.analyze_layer_dynamics(text)
    dynamics_fig.show()

if __name__ == "__main__":
    main() 