from setuptools import setup, find_packages

setup(
    name="model-interpretability",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "plotly",
        "numpy",
        "seaborn",
        "tensorflow-hub",
        "nltk",
        "umap-learn",
        "scikit-learn",
        "networkx"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A framework for analyzing and visualizing language model behavior",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/model-interpretability",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 