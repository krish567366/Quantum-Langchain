# 📚 QuantumLangChain Examples

## 🎯 Basic Examples

### Hello Quantum World

```python
import asyncio
from quantumlangchain import QLChain

async def hello_quantum():
    """Your first quantum-enhanced AI chain."""
    chain = QLChain()
    await chain.initialize()
    
    result = await chain.arun("What is quantum superposition?")
    print(f"🌊 Quantum Answer: {result}")

asyncio.run(hello_quantum())
```

### Quantum Memory Demo

```python
from quantumlangchain import QuantumMemory

async def memory_example():
    """Demonstrate quantum-enhanced memory storage and retrieval."""
    memory = QuantumMemory(classical_dim=256, quantum_dim=4)
    await memory.initialize()
    
    # Store quantum concepts
    concepts = [
        "Quantum entanglement creates non-local correlations",
        "Superposition allows qubits to exist in multiple states",
        "Decoherence causes quantum states to lose coherence",
        "Quantum interference enables computational speedups"
    ]
    
    for i, concept in enumerate(concepts):
        await memory.store(f"concept_{i}", concept)
    
    # Quantum-enhanced retrieval
    query = "quantum computational advantages"
    results = await memory.retrieve(query, top_k=2)
    
    print(f"🔍 Query: {query}")
    for result in results:
        print(f"📊 Score: {result['score']:.3f}")
        print(f"📝 Content: {result['content']}")
        print("---")

asyncio.run(memory_example())
```

## 🤖 Multi-Agent Examples

### Collaborative Problem Solving

```python
from quantumlangchain import EntangledAgents, SharedQuantumMemory

async def agent_collaboration():
    """Demonstrate quantum-entangled multi-agent collaboration."""
    
    # Create shared quantum memory
    shared_memory = SharedQuantumMemory(
        agents=3,
        entanglement_depth=2
    )
    
    # Initialize entangled agents
    agents = EntangledAgents(
        agent_count=3,
        shared_memory=shared_memory,
        specializations=["theory", "implementation", "optimization"]
    )
    
    await agents.initialize()
    
    # Collaborative task
    problem = "Design a quantum algorithm for solving traveling salesman problem"
    
    solution = await agents.collaborative_solve(
        problem,
        collaboration_type="consensus",
        max_rounds=5
    )
    
    print(f"🎯 Problem: {problem}")
    print(f"🤝 Solution: {solution['result']}")
    print(f"👥 Agent Contributions:")
    
    for agent_id, contribution in solution['contributions'].items():
        print(f"  Agent {agent_id}: {contribution}")
    
    print(f"🔗 Entanglement Score: {solution['entanglement_score']:.3f}")

asyncio.run(agent_collaboration())
```

### Competitive Agent System

```python
async def competitive_agents():
    """Demonstrate competitive quantum agents with interference."""
    
    agents = EntangledAgents(
        agent_count=2,
        competition_mode=True,
        interference_weight=0.5
    )
    
    await agents.initialize()
    
    # Competitive optimization
    problem = "Find the optimal hyperparameters for a neural network"
    
    solutions = await agents.competitive_solve(
        problem,
        rounds=3,
        selection_pressure=0.7
    )
    
    print("🏆 Competitive Solutions:")
    for i, solution in enumerate(solutions):
        print(f"Solution {i+1}: {solution['result']}")
        print(f"Fitness: {solution['fitness']:.3f}")
        print("---")

asyncio.run(competitive_agents())
```

## 🔍 Advanced Retrieval Examples

### Quantum-Enhanced RAG

```python
from quantumlangchain import QuantumRetriever, HybridChromaDB, QLChain

async def quantum_rag_system():
    """Build a quantum-enhanced Retrieval-Augmented Generation system."""
    
    # Setup hybrid vector store
    vectorstore = HybridChromaDB(
        collection_name="quantum_knowledge",
        classical_embeddings=True,
        quantum_embeddings=True,
        entanglement_degree=2
    )
    
    # Add knowledge base
    documents = [
        "Quantum computers use qubits instead of classical bits",
        "Shor's algorithm can factor large numbers exponentially faster",
        "Grover's algorithm provides quadratic speedup for search",
        "Quantum error correction is essential for fault-tolerant computing",
        "Variational quantum eigensolver finds ground state energies"
    ]
    
    await vectorstore.bulk_add(documents)
    
    # Quantum retriever with Grover enhancement
    retriever = QuantumRetriever(
        vectorstore=vectorstore,
        grover_iterations=2,
        quantum_speedup=True,
        top_k=3
    )
    
    # Quantum chain with retrieval
    chain = QLChain(
        retriever=retriever,
        augment_with_retrieval=True
    )
    
    await chain.initialize()
    
    # Query with quantum enhancement
    query = "How do quantum algorithms achieve computational speedups?"
    result = await chain.arun(query)
    
    print(f"🔍 Query: {query}")
    print(f"🧠 Enhanced Answer: {result['answer']}")
    print(f"📚 Retrieved Context: {result['context']}")
    print(f"⚡ Quantum Speedup: {result['quantum_speedup']:.2f}x")

asyncio.run(quantum_rag_system())
```

## 🧪 Experimental Features

### Quantum Circuit Integration

```python
from quantumlangchain import QLChain, QuantumCircuitInjector
from qiskit import QuantumCircuit

async def circuit_injection_example():
    """Demonstrate custom quantum circuit injection."""
    
    # Create custom quantum circuit
    def create_custom_circuit(n_qubits):
        circuit = QuantumCircuit(n_qubits)
        
        # Create entangled state
        circuit.h(0)
        for i in range(n_qubits - 1):
            circuit.cx(i, i + 1)
            
        # Add rotation gates
        for i in range(n_qubits):
            circuit.ry(0.5, i)
            
        return circuit
    
    # Circuit injector
    injector = QuantumCircuitInjector(
        circuit_generator=create_custom_circuit,
        injection_points=["reasoning", "memory_access"]
    )
    
    # Chain with custom circuits
    chain = QLChain(
        circuit_injector=injector,
        quantum_enhancement_level="high"
    )
    
    await chain.initialize()
    
    result = await chain.arun(
        "Analyze the quantum advantages in machine learning",
        inject_circuits=True
    )
    
    print(f"🔬 Circuit-Enhanced Result: {result}")

asyncio.run(circuit_injection_example())
```

### Timeline Manipulation

```python
async def timeline_example():
    """Demonstrate quantum timeline manipulation and rollback."""
    
    memory = QuantumMemory(
        classical_dim=128,
        quantum_dim=4,
        timeline_tracking=True
    )
    
    await memory.initialize()
    
    # Initial state
    await memory.store("fact1", "Initial knowledge state")
    checkpoint1 = await memory.create_checkpoint()
    
    # Add more information
    await memory.store("fact2", "Additional knowledge added")
    await memory.store("fact3", "More complex information")
    checkpoint2 = await memory.create_checkpoint()
    
    # Risky operation that might need rollback
    await memory.store("risky_fact", "Potentially incorrect information")
    
    # Evaluate and rollback if needed
    quality_score = await memory.evaluate_coherence()
    
    if quality_score < 0.7:
        print("🔄 Rolling back to previous checkpoint...")
        await memory.rollback_to_checkpoint(checkpoint2)
    
    # Timeline analysis
    timeline = await memory.get_timeline()
    print("📈 Memory Timeline:")
    for event in timeline:
        print(f"  {event['timestamp']}: {event['action']}")

asyncio.run(timeline_example())
```

## 🎨 Creative Applications

### Quantum Poetry Generator

```python
async def quantum_poetry():
    """Generate poetry using quantum superposition of ideas."""
    
    chain = QLChain(
        creative_mode=True,
        superposition_layers=3
    )
    
    await chain.initialize()
    
    # Quantum-inspired poetry
    poem = await chain.arun(
        "Write a haiku about quantum entanglement",
        creative_enhancement=True,
        explore_superposition=True
    )
    
    print("🎭 Quantum Haiku:")
    print(poem)

asyncio.run(quantum_poetry())
```

### Quantum Brainstorming

```python
async def quantum_brainstorm():
    """Use quantum parallelism for idea generation."""
    
    agents = EntangledAgents(
        agent_count=5,
        diversity_enhancement=True,
        idea_entanglement=True
    )
    
    await agents.initialize()
    
    # Parallel idea generation
    ideas = await agents.parallel_brainstorm(
        "Innovative applications of quantum computing in healthcare",
        idea_count=10,
        diversity_weight=0.8
    )
    
    print("💡 Quantum-Generated Ideas:")
    for i, idea in enumerate(ideas, 1):
        print(f"{i}. {idea['content']}")
        print(f"   Novelty: {idea['novelty_score']:.3f}")
        print(f"   Feasibility: {idea['feasibility_score']:.3f}")
        print()

asyncio.run(quantum_brainstorm())
```

## 🔬 Research Applications

### Quantum Literature Review

```python
async def literature_review():
    """Conduct quantum-enhanced literature analysis."""
    
    # Setup quantum retrieval system
    vectorstore = HybridChromaDB("research_papers")
    retriever = QuantumRetriever(
        vectorstore=vectorstore,
        semantic_clustering=True,
        quantum_similarity=True
    )
    
    # Analysis chain
    chain = QLChain(
        retriever=retriever,
        analytical_mode=True,
        citation_tracking=True
    )
    
    await chain.initialize()
    
    # Research query
    query = "Recent advances in quantum machine learning algorithms"
    
    analysis = await chain.research_synthesis(
        query,
        depth="comprehensive",
        timeline="2020-2025",
        include_citations=True
    )
    
    print(f"📊 Research Analysis: {query}")
    print(f"📚 Papers Analyzed: {analysis['paper_count']}")
    print(f"🔍 Key Insights: {analysis['insights']}")
    print(f"📈 Trends: {analysis['trends']}")
    print(f"🔮 Future Directions: {analysis['future_work']}")

asyncio.run(literature_review())
```

## 🏗️ Production Examples

### Quantum API Service

```python
from fastapi import FastAPI
from quantumlangchain import QLChain, QuantumMemory

app = FastAPI(title="Quantum AI API")

# Global quantum components
quantum_chain = None
quantum_memory = None

@app.on_event("startup")
async def startup():
    global quantum_chain, quantum_memory
    
    quantum_memory = QuantumMemory(classical_dim=512, quantum_dim=8)
    await quantum_memory.initialize()
    
    quantum_chain = QLChain(memory=quantum_memory)
    await quantum_chain.initialize()

@app.post("/quantum-reasoning")
async def quantum_reasoning(query: str):
    """Quantum-enhanced reasoning endpoint."""
    result = await quantum_chain.arun(query)
    return {
        "query": query,
        "answer": result["answer"],
        "quantum_state": result["quantum_state"],
        "confidence": result["confidence"]
    }

@app.post("/quantum-memory/store")
async def store_memory(key: str, content: str):
    """Store information in quantum memory."""
    await quantum_memory.store(key, content)
    return {"status": "stored", "key": key}

@app.get("/quantum-memory/retrieve")
async def retrieve_memory(query: str, top_k: int = 5):
    """Retrieve from quantum memory."""
    results = await quantum_memory.retrieve(query, top_k=top_k)
    return {"query": query, "results": results}

# Run with: uvicorn main:app --reload
```

### Quantum Batch Processing

```python
async def batch_processing_example():
    """Process large batches of queries efficiently."""
    
    chain = QLChain(
        batch_mode=True,
        parallel_processing=True,
        optimization_level="high"
    )
    
    await chain.initialize()
    
    # Large batch of queries
    queries = [
        "Explain quantum entanglement",
        "What is superposition?",
        "How does quantum computing work?",
        "Applications of quantum algorithms",
        "Quantum vs classical advantages"
    ] * 100  # 500 queries total
    
    # Batch processing with progress tracking
    async for batch_result in chain.batch_process_stream(
        queries,
        batch_size=50,
        show_progress=True
    ):
        print(f"Processed batch: {batch_result['batch_id']}")
        print(f"Completion rate: {batch_result['completion_rate']:.2f}")

asyncio.run(batch_processing_example())
```

These examples showcase the full range of QuantumLangChain capabilities, from basic usage to advanced research and production applications. Each example is designed to be educational and practically useful.

## 🚀 Running the Examples

1. **Install QuantumLangChain**: `pip install quantumlangchain[all]`
2. **Choose an example**: Copy the code you want to try
3. **Run it**: `python your_example.py`
4. **Experiment**: Modify parameters and see the results

Happy quantum computing! 🌊⚛️
