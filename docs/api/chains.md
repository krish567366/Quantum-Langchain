# ⛓️ Chains API Reference

🔐 **Licensed Component** - Contact: [bajpaikrishna715@gmail.com](mailto:bajpaikrishna715@gmail.com) for licensing

## Chain Classes

### QLChain

Main quantum chain implementation.

```python
class QLChain(LicensedComponent):
    """Quantum-enhanced language chain."""
    
    async def arun(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute quantum reasoning."""
```

### ConversationalQuantumChain

Conversational chain with quantum memory.

```python
class ConversationalQuantumChain(QLChain):
    """Conversational quantum chain."""
    
    async def conversation_turn(self, input: str, session_id: str) -> str:
        """Process conversation turn."""
```

### RAGQuantumChain

Retrieval-augmented generation with quantum enhancement.

```python
class RAGQuantumChain(QLChain):
    """Quantum RAG chain."""
    
    async def quantum_retrieval(self, query: str, **kwargs) -> List[Document]:
        """Quantum-enhanced retrieval."""
```

## 🔐 License Requirements

Chain API features require appropriate licensing tiers. Contact [bajpaikrishna715@gmail.com](mailto:bajpaikrishna715@gmail.com) for licensing.
