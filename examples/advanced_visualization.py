from src.config.visualization_config import AdvancedVisualizationConfig
from src.visualizers.advanced import AdvancedVisualizer
from src.visualizers.enhanced import EnhancedVisualizer

def main():
    # Initialize configuration
    config = AdvancedVisualizationConfig()
    
    # Create visualizers
    advanced_viz = AdvancedVisualizer(config)
    enhanced_viz = EnhancedVisualizer(config)
    
    # Example texts
    texts = [
        "This is a positive example",
        "This is a negative example",
        "This is a neutral example"
    ]
    labels = [1, 0, 0.5]
    
    # Create 3D embedding visualization
    embedding_fig = advanced_viz.plot_embedding_space(
        texts=texts,
        labels=labels
    )
    embedding_fig.show()
    
    # Create attention flow visualization
    attention_fig = enhanced_viz.plot_attention_flow(
        text=texts[0],
        layer_idx=-1
    )
    attention_fig.show()

if __name__ == "__main__":
    main() 