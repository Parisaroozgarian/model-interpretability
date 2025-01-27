import unittest
import torch
from src.config.visualization_config import AdvancedVisualizationConfig
from src.visualizers.deep_analysis import DeepModelAnalyzer

class TestDeepModelAnalyzer(unittest.TestCase):
    def setUp(self):
        self.config = AdvancedVisualizationConfig()
        self.analyzer = DeepModelAnalyzer(self.config)
        
    def test_analyze_layer_dynamics(self):
        text = "This is a test sentence"
        
        fig = self.analyzer.analyze_layer_dynamics(text)
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 4)  # Four metrics
        self.assertTrue(all(trace.mode == "lines+markers" for trace in fig.data))

if __name__ == '__main__':
    unittest.main() 