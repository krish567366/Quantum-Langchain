# 🔍 Retrievers API Reference

🔐 **Licensed Component** - Contact: [bajpaikrishna715@gmail.com](mailto:bajpaikrishna715@gmail.com) for licensing

## Retriever Classes

### QuantumRetriever

Quantum-enhanced document retrieval.

```python
class QuantumRetriever(LicensedComponent):
    """Quantum document retriever."""
    
    async def retrieve(self, query: str, **kwargs) -> List[Document]:
        """Quantum-enhanced retrieval."""
        
    async def quantum_search(self, query: str, **kwargs) -> List[Document]:
        """Pure quantum search."""
```

### VectorStoreRetriever

Retriever for vector stores.

```python
class VectorStoreRetriever(QuantumRetriever):
    """Vector store quantum retriever."""
    
    async def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Quantum similarity search."""
```

### MultiRetriever

Multiple retriever coordination.

```python
class MultiRetriever(QuantumRetriever):
    """Multi-source quantum retriever."""
    
    async def ensemble_retrieve(self, query: str, **kwargs) -> List[Document]:
        """Ensemble retrieval with quantum fusion."""
```

## 🔐 License Requirements

Retriever API features require appropriate licensing tiers. Contact [bajpaikrishna715@gmail.com](mailto:bajpaikrishna715@gmail.com) for licensing.
