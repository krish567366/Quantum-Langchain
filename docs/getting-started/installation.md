# Getting Started with QuantumLangChain

Welcome to QuantumLangChain! This guide will help you get up and running with the world's most advanced hybrid quantum-classical AI framework.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Optional: GPU support for enhanced performance

### Install from PyPI (Recommended)

```bash
pip install quantumlangchain
```

### Install from Source

```bash
git clone https://github.com/krishna-bajpai/quantumlangchain.git
cd quantumlangchain
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/krishna-bajpai/quantumlangchain.git
cd quantumlangchain
pip install -e ".[dev]"
```

## Dependencies

QuantumLangChain automatically installs the core dependencies:

### Quantum Computing Backends

- **Qiskit** - IBM's quantum computing framework
- **PennyLane** - Xanadu's quantum machine learning library
- **Amazon Braket SDK** - AWS quantum computing service

### AI/ML Libraries

- **LangChain** - Base chain and agent framework
- **transformers** - Hugging Face transformers
- **torch** - PyTorch for neural networks
- **numpy** - Numerical computing

### Vector Stores

- **ChromaDB** - Vector database for embeddings
- **FAISS** - Facebook AI Similarity Search
- **Sentence Transformers** - Text embeddings

### Optional Dependencies

For enhanced functionality, install optional packages:

```bash
# Quantum hardware support
pip install qiskit-ibm-runtime qiskit-aer

# Advanced ML capabilities
pip install datasets gymnasium

# Visualization
pip install matplotlib plotly

# Development tools
pip install pytest black flake8 mypy
```

## Quick Verification

Test your installation:

```python
import quantumlangchain as qlc

# Check version
print(f"QuantumLangChain version: {qlc.__version__}")

# Test quantum backend
from quantumlangchain.backends import QiskitBackend
backend = QiskitBackend()
print(f"Quantum backend ready: {backend.get_backend_info()}")
```

## Next Steps

- 📚 [Quick Start Guide](quick-start.md) - Your first quantum AI chain
- 🏗️ [Architecture Overview](../architecture.md) - Understanding the framework
- 🧪 [Examples](../examples/) - Comprehensive examples
- 📖 [API Reference](../api/) - Detailed API documentation

## Getting Help

- 📖 [Documentation](https://krishna-bajpai.github.io/quantumlangchain/)
- 🐛 [Issues](https://github.com/krishna-bajpai/quantumlangchain/issues)
- 💬 [Discussions](https://github.com/krishna-bajpai/quantumlangchain/discussions)
- 📧 [Email Support](mailto:bajpaikrishna715@gmail.com)

## System Requirements

### Minimum Requirements

- **CPU**: 2+ cores
- **RAM**: 4GB
- **Storage**: 1GB free space
- **Python**: 3.8+

### Recommended Specifications

- **CPU**: 4+ cores (Intel/AMD x64 or Apple Silicon)
- **RAM**: 8GB+ (16GB for large quantum simulations)
- **GPU**: CUDA-compatible for accelerated computing
- **Storage**: 5GB+ SSD storage
- **Network**: Stable internet for quantum cloud access

### Supported Platforms

- ✅ **Linux** (Ubuntu 20.04+, CentOS 8+)
- ✅ **macOS** (10.15+, including Apple Silicon)
- ✅ **Windows** (10/11, WSL2 recommended)
- ✅ **Docker** containers
- ✅ **Google Colab** and Jupyter environments

## License

QuantumLangChain is open source software licensed under the [MIT License](https://github.com/krishna-bajpai/quantumlangchain/blob/main/LICENSE).
