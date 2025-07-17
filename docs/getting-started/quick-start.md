# Quick Start Guide

Get up and running with QuantumLangChain in just a few minutes! This guide will walk you through creating your first quantum-enhanced AI application.

## Your First Quantum Chain

Let's start with a simple quantum-enhanced reasoning chain:

```python
import asyncio
from quantumlangchain import QLChain, QuantumMemory, QiskitBackend

async def main():
    # Initialize quantum backend
    backend = QiskitBackend()
    
    # Create quantum memory
    memory = QuantumMemory(
        classical_dim=128,
        quantum_dim=4,
        backend=backend
    )
    
    # Create quantum chain
    chain = QLChain(
        memory=memory,
        backend=backend,
        config={
            "parallel_branches": 2,
            "circuit_injection_enabled": True
        }
    )
    
    # Initialize components
    await memory.initialize()
    await chain.initialize()
    
    # Run quantum-enhanced reasoning
    result = await chain.arun(
        "What are the implications of quantum entanglement for AI?"
    )
    
    print("Quantum AI Response:", result)
    
    # Get execution statistics
    stats = chain.get_execution_stats()
    print(f"Quantum state: {chain.quantum_state}")
    print(f"Executions: {stats['total_executions']}")

# Run the example
asyncio.run(main())
```

## Multi-Agent Quantum Collaboration

Create entangled agents that work together:

```python
import asyncio
from quantumlangchain import EntangledAgents, QiskitBackend

async def collaborative_problem_solving():
    backend = QiskitBackend()
    
    # Define agent roles
    agent_configs = [
        {
            "name": "researcher",
            "description": "Research and analysis specialist",
            "capabilities": ["search", "analyze", "synthesize"],
            "priority": 0.8
        },
        {
            "name": "validator",
            "description": "Solution validation expert", 
            "capabilities": ["verify", "test", "critique"],
            "priority": 0.9
        },
        {
            "name": "optimizer",
            "description": "Solution optimization specialist",
            "capabilities": ["optimize", "refine", "enhance"],
            "priority": 0.7
        }
    ]
    
    # Create entangled agent system
    agents = EntangledAgents(
        agent_configs=agent_configs,
        backend=backend
    )
    
    await agents.initialize()
    
    # Collaborative problem solving
    problem = """
    Design a quantum algorithm for optimizing neural network 
    architectures while maintaining interpretability.
    """
    
    solution = await agents.collaborative_solve(
        problem=problem,
        max_iterations=3,
        enable_interference=True
    )
    
    print("Collaborative Solution:")
    print(solution["final_solution"])
    print(f"\nAgent Contributions: {len(solution['agent_contributions'])}")
    print(f"Quantum Effects: {solution['quantum_effects']}")

asyncio.run(collaborative_problem_solving())
```

## Quantum-Enhanced Vector Search

Use quantum algorithms for superior document retrieval:

```python
import asyncio
from quantumlangchain import QuantumRetriever, HybridChromaDB, QiskitBackend

async def quantum_search_demo():
    backend = QiskitBackend()
    
    # Create quantum vector store
    vectorstore = HybridChromaDB(
        collection_name="quantum_docs",
        config={
            "similarity_threshold": 0.7,
            "quantum_boost_factor": 1.5
        }
    )
    
    await vectorstore.initialize()
    
    # Add some quantum-enhanced documents
    documents = [
        "Quantum entanglement enables instantaneous correlation between particles",
        "Machine learning algorithms can benefit from quantum superposition", 
        "Quantum computing offers exponential speedup for certain problems",
        "Neural networks and quantum circuits share mathematical similarities",
        "Quantum error correction is essential for fault-tolerant computing"
    ]
    
    doc_ids = await vectorstore.add_documents(
        documents=documents,
        quantum_enhanced=True
    )
    
    # Create entanglement between related documents
    related_docs = doc_ids[:3]  # First 3 documents
    entanglement_id = await vectorstore.entangle_documents(related_docs)
    
    # Create quantum retriever
    retriever = QuantumRetriever(
        vectorstore=vectorstore,
        backend=backend,
        config={
            "search_type": "amplitude_amplification",
            "k": 3
        }
    )
    
    await retriever.initialize()
    
    # Quantum-enhanced search
    query = "quantum machine learning applications"
    results = await retriever.aretrieve(query, quantum_enhanced=True)
    
    print("Quantum Search Results:")
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc.page_content}")
        print(f"   Score: {doc.metadata.get('score', 'N/A')}")
        print(f"   Quantum Enhanced: {doc.metadata.get('quantum_enhanced', False)}")

asyncio.run(quantum_search_demo())
```

## Quantum Memory Systems

Implement reversible, entangled memory:

```python
import asyncio
from quantumlangchain import QuantumMemory, QiskitBackend

async def quantum_memory_demo():
    backend = QiskitBackend()
    
    # Create quantum memory system
    memory = QuantumMemory(
        classical_dim=256,
        quantum_dim=6,
        backend=backend
    )
    
    await memory.initialize()
    
    # Store quantum-enhanced memories
    await memory.store("concept_1", "Quantum superposition principle", quantum_enhanced=True)
    await memory.store("concept_2", "Entanglement correlations", quantum_enhanced=True)
    await memory.store("concept_3", "Wave function collapse", quantum_enhanced=True)
    
    # Create memory entanglement
    concepts = ["concept_1", "concept_2", "concept_3"]
    entanglement_id = await memory.entangle_memories(concepts)
    
    # Create memory snapshot
    snapshot_id = await memory.create_memory_snapshot()
    
    # Demonstrate quantum memory retrieval
    result = await memory.retrieve("concept_1", quantum_search=True)
    print(f"Retrieved: {result}")
    
    # Add new memory after snapshot
    await memory.store("concept_4", "Quantum decoherence effects", quantum_enhanced=True)
    
    # Show current stats
    stats = await memory.get_stats()
    print(f"\nMemory Statistics:")
    print(f"Total entries: {stats['total_entries']}")
    print(f"Quantum enhanced: {stats['quantum_enhanced_entries']}")
    print(f"Entangled entries: {stats['entangled_entries']}")
    
    # Restore previous snapshot
    await memory.restore_memory_snapshot(snapshot_id)
    
    # Verify restoration
    stats_after = await memory.get_stats()
    print(f"\nAfter restoration:")
    print(f"Total entries: {stats_after['total_entries']}")

asyncio.run(quantum_memory_demo())
```

## Quantum Tool Execution

Chain tools with quantum enhancement:

```python
import asyncio
from quantumlangchain import QuantumToolExecutor

async def quantum_tools_demo():
    executor = QuantumToolExecutor()
    await executor.initialize()
    
    # Register quantum-enhanced tools
    def analyze_data(data):
        return f"Analysis of {data}: Pattern detected with 87% confidence"
    
    def optimize_parameters(params):
        return f"Optimized parameters: {params} -> improved by 23%"
    
    def validate_results(results):
        return f"Validation: {results} passes all quantum coherence tests"
    
    # Register tools
    executor.register_tool("analyze", analyze_data, "Data analysis tool", quantum_enhanced=True)
    executor.register_tool("optimize", optimize_parameters, "Parameter optimizer", quantum_enhanced=True)
    executor.register_tool("validate", validate_results, "Result validator", quantum_enhanced=True)
    
    # Create tool chain
    executor.create_tool_chain("analysis_pipeline", ["analyze", "optimize", "validate"])
    
    # Execute quantum superposition tools (parallel execution)
    tool_configs = [
        {"name": "analyze", "args": ["quantum_dataset_1"], "kwargs": {}},
        {"name": "analyze", "args": ["quantum_dataset_2"], "kwargs": {}},
        {"name": "analyze", "args": ["quantum_dataset_3"], "kwargs": {}}
    ]
    
    # Run tools in quantum superposition
    result = await executor.execute_quantum_superposition_tools(
        tool_configs=tool_configs,
        measurement_function=lambda results: max(results, key=lambda r: r.execution_time if r.success else 0)
    )
    
    print("Quantum Tool Result:", result.result)
    print("Quantum Effects:", result.metadata)
    
    # Get tool statistics
    stats = executor.get_tool_statistics()
    print(f"\nTool Statistics:")
    print(f"Total executions: {stats['total_executions']}")
    print(f"Quantum enhanced: {stats['quantum_enhanced_executions']}")

asyncio.run(quantum_tools_demo())
```

## Configuration and Backends

### Backend Selection

Choose your quantum backend:

```python
from quantumlangchain.backends import QiskitBackend, PennyLaneBackend, BraketBackend

# IBM Qiskit (default)
qiskit_backend = QiskitBackend()

# Xanadu PennyLane
pennylane_backend = PennyLaneBackend(device="default.qubit", shots=1000)

# Amazon Braket
braket_backend = BraketBackend(device="local:braket/local_simulator")
```

### Global Configuration

```python
from quantumlangchain.core.base import QuantumConfig

# Configure quantum parameters
config = QuantumConfig(
    num_qubits=8,
    circuit_depth=20,
    decoherence_threshold=0.1,
    backend_type="qiskit",
    enable_error_correction=True
)

# Use with any quantum component
chain = QLChain(config=config)
```

## Next Steps

Now that you've seen the basics, explore more advanced features:

- 🏗️ [Architecture Guide](../architecture.md) - Deep dive into framework design
- 🧪 [Advanced Examples](../examples/) - Complex quantum AI applications  
- 🔧 [API Reference](../api/) - Complete API documentation
- 🚀 [Deployment Guide](deployment.md) - Production deployment strategies
- 🎯 [Best Practices](best-practices.md) - Optimization and performance tips

## Common Patterns

### Error Handling

```python
try:
    result = await chain.arun(query)
except QuantumDecoherenceError:
    # Handle quantum decoherence
    await chain.reset_quantum_state()
    result = await chain.arun(query)
except Exception as e:
    print(f"Error: {e}")
```

### Performance Monitoring

```python
# Monitor quantum coherence
if chain.decoherence_level > 0.8:
    await chain.reset_quantum_state()

# Track execution stats
stats = chain.get_execution_stats()
print(f"Success rate: {stats['success_rate']}")
```

### Resource Management

```python
# Cleanup quantum resources
async with QLChain(memory=memory, backend=backend) as chain:
    result = await chain.arun(query)
# Chain automatically cleaned up
```
