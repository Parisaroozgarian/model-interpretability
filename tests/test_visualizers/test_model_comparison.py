import unittest
import numpy as np
from src.config.visualization_config import VisualizationConfig
from src.visualizers.model_comparison import ModelComparisonVisualizer

class TestModelComparisonVisualizer(unittest.TestCase):
    def setUp(self):
        self.config = VisualizationConfig()
        self.visualizer = ModelComparisonVisualizer(self.config)
        
    def test_plot_embedding_comparison(self):
        embeddings1 = np.random.random((5, 10))
        embeddings2 = np.random.random((5, 10))
        labels = ["text1", "text2", "text3", "text4", "text5"]
        
        fig = self.visualizer.plot_embedding_comparison(
            embeddings1=embeddings1,
            embeddings2=embeddings2,
            labels=labels
        )
        
        self.assertIsNotNone(fig)
        self.assertEqual(fig.layout.width, self.config.width)
        self.assertEqual(fig.layout.height, self.config.height)

if __name__ == '__main__':
    unittest.main() 