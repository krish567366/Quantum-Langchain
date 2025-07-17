# 🔧 Core API Reference

🔐 **Licensed Component** - Contact: [bajpaikrishna715@gmail.com](mailto:bajpaikrishna715@gmail.com) for licensing

## Core Classes and Functions

### QLChain

The main interface for quantum-enhanced language processing.

```python
class QLChain:
    def __init__(
        self,
        backend: str = "qiskit",
        quantum_dim: int = 4,
        classical_dim: int = 512,
        **kwargs
    ):
        """Initialize QLChain with quantum backend."""
        
    async def arun(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute quantum-enhanced reasoning."""
        
    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """Synchronous version of arun."""
```

### QuantumMemory

Quantum-enhanced memory system.

```python
class QuantumMemory:
    def __init__(
        self,
        classical_dim: int = 512,
        quantum_dim: int = 4,
        **kwargs
    ):
        """Initialize quantum memory."""
        
    async def store(self, key: str, value: Any, **metadata) -> bool:
        """Store information in quantum memory."""
        
    async def retrieve(self, query: str, **params) -> List[MemoryItem]:
        """Retrieve information using quantum search."""
```

## 🔐 License Requirements

All core API features require valid licensing. Contact [bajpaikrishna715@gmail.com](mailto:bajpaikrishna715@gmail.com) for licensing.
