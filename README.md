# 🔍 Model Interpretability Framework

A deep learning model visualization and interpretation framework that helps understand model behavior through interactive visualizations.

[![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://badge.fury.io/py/model-interpretability.svg)](https://badge.fury.io/py/model-interpretability)
[![Documentation Status](https://readthedocs.org/projects/model-interpretability/badge/?version=latest)](https://model-interpretability.readthedocs.io/en/latest/?badge=latest)

> 🎯 **Deep Learning Model Analysis Made Easy**: Visualize and understand your model's behavior, attention patterns, and decision boundaries with interactive visualizations.

**Keywords**: `deep-learning` `model-interpretability` `visualization` `machine-learning` `attention-visualization` `pytorch` `model-analysis` `uncertainty-quantification`

## 🎯 Overview

This framework provides tools to analyze and visualize:
- 🧠 Model attention patterns
- 🎯 Token importance scores
- 📊 Uncertainty landscapes
- 🌐 Embedding spaces
- ⚡ Layer dynamics
- 🔄 Decision boundaries

## ✨ Key Features

### 📊 1. Basic Visualizations
- 📈 Token importance bar charts
- 🔥 Attention heatmaps
- 📉 Confidence scores

### 🔬 2. Advanced Analysis
- 🌌 2D embedding space visualization
- 🔄 Cross-model comparisons
- 📱 Interactive plots with Plotly

### 🚀 3. Deep Model Analysis
- 🔄 Layer-wise representation dynamics
- 📊 Multi-metric visualization
- ⚡ Attention flow patterns

### 🎯 4. Uncertainty Analysis
- 🌐 3D uncertainty landscapes
- 📈 Predictive entropy visualization
- 📉 Confidence evolution plots

## 🚀 Quick Start

1. Install the package:
```bash
git clone https://github.com/Parisaroozgarian/model-interpretability.git
cd model-interpretability
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

2. Basic usage:
```python
from src.config.visualization_config import VisualizationConfig
from src.visualizers.feature_importance import FeatureImportanceVisualizer

# Initialize
config = VisualizationConfig()
viz = FeatureImportanceVisualizer(config)

# Create visualization
fig = viz.plot_token_importance(
    tokens=["This", "is", "a", "test"],
    importance_scores=[0.3, 0.2, 0.1, 0.4]
)
fig.show()
```

## 📚 Examples

Check the `examples/` directory for more usage examples:
- 📊 `basic_visualization.py`: Simple token importance and uncertainty plots
- 🔬 `advanced_visualization.py`: 2D embedding space and attention patterns
- 🚀 `deep_analysis.py`: Layer dynamics and model behavior analysis

## 📁 Project Structure
```
model-interpretability/
├── src/                 # Source code
│   ├── config/         # ⚙️ Configuration settings
│   │   └── visualization_config.py
│   ├── visualizers/    # 📊 Visualization modules
│   │   ├── base.py
│   │   ├── feature_importance.py
│   │   ├── uncertainty.py
│   │   ├── advanced.py
│   │   └── deep_analysis.py
│   └── utils/          # 🛠️ Utility functions
├── examples/           # 📚 Usage examples
├── tests/             # 🧪 Test suite
└── docs/              # 📖 Documentation
```

## 🛠️ Dependencies
```
torch>=1.9.0
transformers>=4.5.0
plotly>=5.1.0
numpy>=1.19.5
seaborn>=0.11.2
tensorflow-hub>=0.12.0
nltk>=3.6.3
umap-learn>=0.5.1
scikit-learn>=0.24.2
networkx>=2.6.2
```

## 👨‍💻 Development

Run tests:
```bash
python -m pytest tests/
```

Code coverage:
```bash
pytest tests/ --cov=src/ --cov-report=xml
```

## 👩‍💻 Author

[Parisa Roozgarian](https://github.com/Parisaroozgarian)


## 🙏 Acknowledgments

- 🤗 BERT model from Hugging Face Transformers
- 📊 Plotly for interactive visualizations
- 🌐 UMAP for dimensionality reduction

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details