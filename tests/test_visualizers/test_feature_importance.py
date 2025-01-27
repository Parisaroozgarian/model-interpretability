import unittest
import numpy as np
from src.config.visualization_config import VisualizationConfig
from src.visualizers.feature_importance import FeatureImportanceVisualizer

class TestFeatureImportanceVisualizer(unittest.TestCase):
    def setUp(self):
        self.config = VisualizationConfig()
        self.visualizer = FeatureImportanceVisualizer(self.config)
        
    def test_plot_token_importance(self):
        tokens = ["This", "is", "a", "test", "sentence"]
        importance_scores = np.random.random(5)
        
        fig = self.visualizer.plot_token_importance(
            tokens=tokens,
            importance_scores=importance_scores,
            highlight_top_k=2
        )
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 1)  # One bar trace
        self.assertEqual(len(fig.data[0].x), len(tokens))

if __name__ == '__main__':
    unittest.main() 