# 🤖 Model Interpretability Framework

A deep learning model visualization and interpretation framework that helps understand model behavior through interactive visualizations.

[![Stars](https://img.shields.io/github/stars/Parisaroozgarian/model-interpretability?style=for-the-badge&logo=github&color=yellow)](https://github.com/Parisaroozgarian/model-interpretability/stargazers)
[![PyPI Version](https://img.shields.io/pypi/v/model-interpretability?style=for-the-badge&logo=pypi&logoColor=white&color=blue)](https://pypi.org/project/model-interpretability/)
[![Python](https://img.shields.io/pypi/pyversions/model-interpretability?style=for-the-badge&logo=python&logoColor=white&color=green)](https://www.python.org/)
[![License](https://img.shields.io/github/license/Parisaroozgarian/model-interpretability?style=for-the-badge&logo=opensourceinitiative&logoColor=white&color=purple)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/Parisaroozgarian/model-interpretability/python-tests.yml?style=for-the-badge&logo=github-actions&logoColor=white&label=tests)](https://github.com/Parisaroozgarian/model-interpretability/actions)
[![Code style](https://img.shields.io/badge/code%20style-black-black?style=for-the-badge&logo=python&logoColor=white)](https://github.com/psf/black)


## 🌟 Overview

This framework provides tools to analyze and visualize:
- 🧮 Model attention patterns
- 📊 Token importance scores
- 🎯 Uncertainty landscapes
- 🔮 Embedding spaces
- ⚡ Layer dynamics
- 🎲 Decision boundaries

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