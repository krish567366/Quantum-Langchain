# Examples

Complete collection of practical examples demonstrating QuantumLangChain capabilities.

## Basic Examples

### Simple Quantum Chain

```python
import asyncio
from quantumlangchain import QLChain, QuantumMemory, QuantumConfig
from quantumlangchain.backends import QiskitBackend

async def simple_quantum_chain():
    """Basic quantum-enhanced chain example."""
    # Setup
    config = QuantumConfig(num_qubits=4, backend_type="qiskit")
    backend = QiskitBackend(config=config)
    memory = QuantumMemory(classical_dim=768, quantum_dim=4, backend=backend)
    
    # Create chain
    chain = QLChain(memory=memory, backend=backend)
    
    # Execute query
    result = await chain.arun(
        "What is quantum superposition?",
        quantum_enhanced=True
    )
    
    print(f"Answer: {result['output']}")
    print(f"Quantum coherence: {result['quantum_coherence']:.3f}")

# Run example
asyncio.run(simple_quantum_chain())
```

### Quantum Memory Storage

```python
async def quantum_memory_example():
    """Demonstrate quantum memory capabilities."""
    from quantumlangchain import QuantumMemory
    
    memory = QuantumMemory(classical_dim=512, quantum_dim=6)
    
    # Store facts with quantum enhancement
    facts = [
        ("quantum_computing", "Quantum computing uses quantum mechanical phenomena like superposition and entanglement."),
        ("machine_learning", "Machine learning algorithms learn patterns from data to make predictions."),
        ("quantum_ai", "Quantum AI combines quantum computing with artificial intelligence for enhanced capabilities.")
    ]
    
    for key, value in facts:
        await memory.store(key, value, quantum_enhanced=True)
    
    # Create entangled memories
    entanglement_id = await memory.entangle_memories([
        "quantum_computing", "machine_learning", "quantum_ai"
    ])
    print(f"Created entanglement: {entanglement_id}")
    
    # Search with quantum enhancement
    results = await memory.similarity_search(
        "How does quantum computing enhance AI?",
        top_k=3
    )
    
    for result in results:
        print(f"Key: {result['key']}")
        print(f"Similarity: {result['similarity']:.3f}")
        print(f"Content: {result['content'][:100]}...")
        print()

asyncio.run(quantum_memory_example())
```

## Multi-Agent Examples

### Collaborative Problem Solving

```python
from quantumlangchain import EntangledAgents, AgentRole

async def collaborative_research():
    """Multi-agent collaborative research example."""
    
    # Define specialized agents
    agent_configs = [
        {
            'agent_id': 'data_scientist',
            'role': AgentRole.ANALYST,
            'capabilities': ['statistical_analysis', 'data_visualization', 'pattern_recognition'],
            'specialization': 'data_analysis',
            'quantum_weight': 1.0
        },
        {
            'agent_id': 'domain_expert',
            'role': AgentRole.RESEARCHER,
            'capabilities': ['domain_knowledge', 'literature_review', 'hypothesis_generation'],
            'specialization': 'subject_matter_expertise',
            'quantum_weight': 0.9
        },
        {
            'agent_id': 'creative_thinker',
            'role': AgentRole.CREATIVE,
            'capabilities': ['brainstorming', 'alternative_perspectives', 'innovative_solutions'],
            'specialization': 'creative_problem_solving',
            'quantum_weight': 0.8
        }
    ]
    
    agents = EntangledAgents(agent_configs=agent_configs)
    
    # Research problem
    problem = """
    Analyze the potential applications of quantum machine learning 
    in drug discovery, considering both current limitations and 
    future possibilities. Provide actionable recommendations.
    """
    
    # Collaborative solution
    result = await agents.collaborative_solve(
        problem=problem,
        max_iterations=3,
        enable_interference=True
    )
    
    print("Collaborative Research Results:")
    print(f"Final Recommendation: {result['solution']}")
    print(f"Confidence Level: {result['confidence']:.2%}")
    print(f"Consensus Score: {result['consensus_score']:.3f}")
    
    print("\nAgent Contributions:")
    for agent_id, contribution in result['contributions'].items():
        print(f"\n{agent_id.replace('_', ' ').title()}:")
        print(f"  Key Insights: {contribution['insights']}")
        print(f"  Confidence: {contribution['confidence']:.2%}")
        print(f"  Quantum Coherence: {contribution['quantum_coherence']:.3f}")

asyncio.run(collaborative_research())
```

### Parallel Agent Analysis

```python
async def parallel_market_analysis():
    """Parallel agent analysis example."""
    
    # Market analysis agents
    market_agents = [
        {
            'agent_id': 'technical_analyst',
            'role': AgentRole.ANALYST,
            'specialization': 'technical_analysis',
            'quantum_weight': 1.0
        },
        {
            'agent_id': 'fundamental_analyst',
            'role': AgentRole.RESEARCHER,
            'specialization': 'fundamental_analysis',
            'quantum_weight': 0.9
        },
        {
            'agent_id': 'sentiment_analyst',
            'role': AgentRole.ANALYST,
            'specialization': 'sentiment_analysis',
            'quantum_weight': 0.8
        }
    ]
    
    agents = EntangledAgents(agent_configs=market_agents)
    
    # Market analysis task
    analysis_task = "Analyze the quantum computing market outlook for 2024-2025"
    
    # Run agents in parallel
    results = await agents.run_parallel_agents(
        agent_ids=['technical_analyst', 'fundamental_analyst', 'sentiment_analyst'],
        problem=analysis_task
    )
    
    print("Parallel Market Analysis:")
    for result in results:
        print(f"\n{result['agent_id'].replace('_', ' ').title()}:")
        print(f"  Analysis: {result['analysis'][:200]}...")
        print(f"  Key Metrics: {result['metrics']}")
        print(f"  Recommendation: {result['recommendation']}")
        print(f"  Confidence: {result['confidence']:.2%}")

asyncio.run(parallel_market_analysis())
```

## Vector Store Examples

### Advanced ChromaDB Integration

```python
from quantumlangchain.vectorstores import HybridChromaDB
import numpy as np

async def advanced_chromadb_example():
    """Advanced ChromaDB with quantum enhancements."""
    
    # Initialize hybrid vector store
    vectorstore = HybridChromaDB(
        collection_name="quantum_research_papers",
        persist_directory="./research_db",
        config={
            'quantum_enhanced': True,
            'entanglement_enabled': True,
            'coherence_threshold': 0.8,
            'similarity_threshold': 0.7
        }
    )
    
    # Research paper abstracts
    research_papers = [
        {
            'content': "Quantum machine learning algorithms demonstrate exponential speedup for certain classification tasks by leveraging quantum superposition and entanglement.",
            'metadata': {
                'title': 'Quantum ML Speedup',
                'year': 2023,
                'field': 'quantum_ml',
                'impact_score': 8.5,
                'authors': ['Smith, J.', 'Johnson, A.']
            }
        },
        {
            'content': "Variational quantum eigensolvers provide near-term quantum advantage for molecular simulation problems in drug discovery applications.",
            'metadata': {
                'title': 'VQE Drug Discovery',
                'year': 2023,
                'field': 'quantum_chemistry',
                'impact_score': 9.2,
                'authors': ['Brown, K.', 'Davis, L.']
            }
        },
        {
            'content': "Quantum natural language processing models show promise for context-aware text analysis using quantum attention mechanisms.",
            'metadata': {
                'title': 'Quantum NLP',
                'year': 2024,
                'field': 'quantum_nlp',
                'impact_score': 7.8,
                'authors': ['Wilson, M.', 'Taylor, R.']
            }
        }
    ]
    
    # Add documents with quantum enhancement
    doc_ids = []
    for paper in research_papers:
        doc_id = await vectorstore.add_documents(
            documents=[paper['content']],
            metadatas=[paper['metadata']],
            quantum_enhanced=True
        )
        doc_ids.extend(doc_id)
    
    # Create thematic entanglements
    ml_papers = [doc_ids[0], doc_ids[2]]  # ML and NLP papers
    entanglement_id = await vectorstore.entangle_documents(
        doc_ids=ml_papers,
        entanglement_strength=0.9
    )
    
    print(f"Created thematic entanglement: {entanglement_id}")
    
    # Quantum similarity search with filters
    results = await vectorstore.quantum_similarity_search(
        query="quantum algorithms for natural language understanding",
        k=3,
        quantum_algorithm="amplitude_amplification"
    )
    
    print("\nQuantum Search Results:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"\nResult {i} (Score: {score:.3f}):")
        print(f"Title: {doc.metadata['title']}")
        print(f"Field: {doc.metadata['field']}")
        print(f"Impact Score: {doc.metadata['impact_score']}")
        print(f"Quantum Coherence: {doc.quantum_coherence:.3f}")
        print(f"Content Preview: {doc.page_content[:100]}...")

asyncio.run(advanced_chromadb_example())
```

### FAISS Quantum Algorithms

```python
from quantumlangchain.vectorstores import QuantumFAISS
import numpy as np

async def faiss_quantum_algorithms():
    """Demonstrate FAISS quantum search algorithms."""
    
    # Initialize quantum FAISS
    vectorstore = QuantumFAISS(
        dimension=384,  # Sentence transformer dimension
        index_type="IVFFlat",
        metric="cosine",
        nlist=50,
        config={
            'quantum_enhancement': True,
            'grovers_enabled': True,
            'amplitude_amplification': True,
            'coherence_preservation': True
        }
    )
    
    # Generate embeddings for technical documents
    np.random.seed(42)
    num_documents = 1000
    embeddings = np.random.rand(num_documents, 384).astype(np.float32)
    
    # Create metadata with categories
    categories = ['quantum_computing', 'machine_learning', 'data_science', 'physics', 'mathematics']
    metadatas = []
    for i in range(num_documents):
        metadatas.append({
            'doc_id': f"doc_{i:04d}",
            'category': categories[i % len(categories)],
            'importance': np.random.uniform(0.1, 1.0),
            'publication_year': 2020 + (i % 4),
            'citation_count': int(np.random.exponential(50))
        })
    
    # Add vectors to index
    doc_ids = [f"doc_{i:04d}" for i in range(num_documents)]
    await vectorstore.add_vectors(
        vectors=embeddings,
        ids=doc_ids,
        metadatas=metadatas,
        quantum_enhanced=True
    )
    
    print(f"Added {num_documents} documents to quantum FAISS index")
    
    # Define search conditions
    def high_impact_condition(metadata):
        """Target high-impact recent papers."""
        return (metadata.get('importance', 0) > 0.7 and 
                metadata.get('publication_year', 0) >= 2022)
    
    def quantum_papers_oracle(metadata):
        """Oracle for quantum computing papers."""
        return metadata.get('category') == 'quantum_computing'
    
    # Query vector
    query_vector = np.random.rand(384).astype(np.float32)
    
    # 1. Amplitude Amplification Search
    print("\n1. Amplitude Amplification Search (High Impact Papers):")
    aa_results = await vectorstore.amplitude_amplification_search(
        query_vector=query_vector,
        target_condition=high_impact_condition,
        k=5,
        iterations=3
    )
    
    for doc_id, score, metadata in aa_results:
        print(f"  {doc_id}: Score={score:.3f}, "
              f"Importance={metadata['importance']:.2f}, "
              f"Year={metadata['publication_year']}")
    
    # 2. Grover's Algorithm Search
    print("\n2. Grover's Algorithm Search (Quantum Computing Papers):")
    grover_results = await vectorstore.grovers_search(
        oracle_function=quantum_papers_oracle,
        k=5,
        max_iterations=10
    )
    
    for doc_id, score, metadata in grover_results:
        print(f"  {doc_id}: Score={score:.3f}, "
              f"Category={metadata['category']}, "
              f"Citations={metadata['citation_count']}")
    
    # 3. Hybrid Quantum-Classical Search
    print("\n3. Hybrid Search (Classical + Quantum Enhancement):")
    hybrid_results = await vectorstore.search(
        query_vector=query_vector,
        k=10,
        quantum_enhanced=True,
        filter_metadata={'category': 'machine_learning'}
    )
    
    for doc_id, score, metadata in hybrid_results[:5]:
        print(f"  {doc_id}: Score={score:.3f}, "
              f"Category={metadata['category']}, "
              f"Importance={metadata['importance']:.2f}")

asyncio.run(faiss_quantum_algorithms())
```

## Tool Integration Examples

### Quantum Tool Orchestration

```python
from quantumlangchain.tools import QuantumToolExecutor
import json
import requests
from typing import Dict, Any

async def quantum_tool_orchestration():
    """Advanced tool orchestration with quantum enhancement."""
    
    executor = QuantumToolExecutor()
    
    # Define research tools
    def arxiv_search(query: str, max_results: int = 5) -> Dict[str, Any]:
        """Simulate arXiv paper search."""
        papers = [
            {
                'title': f'Quantum {query} Research Paper {i+1}',
                'authors': ['Author A', 'Author B'],
                'abstract': f'Abstract discussing {query} with quantum applications...',
                'arxiv_id': f'2024.{1000+i:04d}',
                'categories': ['quant-ph', 'cs.AI'],
                'published': f'2024-0{(i%9)+1}-15'
            }
            for i in range(max_results)
        ]
        return {'papers': papers, 'total_found': max_results}
    
    def extract_key_concepts(text: str) -> Dict[str, Any]:
        """Extract key concepts from text."""
        # Simulate concept extraction
        concepts = ['quantum computing', 'machine learning', 'algorithms', 'optimization']
        return {
            'concepts': concepts[:3],  # Top 3 concepts
            'confidence_scores': [0.9, 0.8, 0.7],
            'extracted_from': text[:50] + '...'
        }
    
    def generate_summary(papers: list, concepts: list) -> Dict[str, Any]:
        """Generate research summary."""
        return {
            'summary': f'Analysis of {len(papers)} papers covering {", ".join(concepts)}',
            'key_findings': [
                'Quantum algorithms show promise',
                'Implementation challenges remain',
                'Future research directions identified'
            ],
            'confidence': 0.85
        }
    
    def citation_analysis(papers: list) -> Dict[str, Any]:
        """Analyze citation patterns."""
        return {
            'total_papers': len(papers),
            'citation_network': {'nodes': len(papers), 'edges': len(papers) * 2},
            'influential_papers': papers[:2] if papers else [],
            'research_trends': ['increasing', 'quantum_advantage', 'practical_applications']
        }
    
    # Register tools with quantum capabilities
    tools = [
        ('arxiv_search', arxiv_search, 'Search academic papers', True, True, False),
        ('extract_concepts', extract_key_concepts, 'Extract key concepts', True, True, True),
        ('generate_summary', generate_summary, 'Generate research summary', False, False, True),
        ('citation_analysis', citation_analysis, 'Analyze citations', True, True, False)
    ]
    
    for name, func, desc, quantum, parallel, entangle in tools:
        executor.register_tool(
            name=name,
            function=func,
            description=desc,
            quantum_enhanced=quantum,
            parallel_execution=parallel,
            entanglement_enabled=entangle
        )
    
    # Create research workflow
    executor.create_tool_chain(
        chain_name="quantum_research_workflow",
        tool_names=["arxiv_search", "extract_concepts", "generate_summary", "citation_analysis"]
    )
    
    # Execute research workflow
    print("Executing Quantum Research Workflow...")
    
    workflow_results = await executor.execute_tool_chain(
        chain_name="quantum_research_workflow",
        initial_input="quantum machine learning optimization",
        propagate_results=True
    )
    
    print("\nWorkflow Results:")
    for i, result in enumerate(workflow_results, 1):
        print(f"\nStep {i}: {result.tool_name}")
        print(f"  Success: {result.success}")
        print(f"  Execution Time: {result.execution_time:.3f}s")
        print(f"  Quantum Enhanced: {result.quantum_enhanced}")
        if result.entanglement_id:
            print(f"  Entanglement ID: {result.entanglement_id}")
        
        # Print relevant result data
        if result.tool_name == "arxiv_search":
            papers = result.result['papers']
            print(f"  Found {len(papers)} papers")
        elif result.tool_name == "extract_concepts":
            concepts = result.result['concepts']
            print(f"  Key concepts: {', '.join(concepts)}")
        elif result.tool_name == "generate_summary":
            summary = result.result['summary']
            print(f"  Summary: {summary}")
        elif result.tool_name == "citation_analysis":
            analysis = result.result
            print(f"  Citation network: {analysis['citation_network']}")

asyncio.run(quantum_tool_orchestration())
```

### Parallel Tool Execution

```python
async def parallel_tool_execution():
    """Demonstrate parallel quantum tool execution."""
    
    executor = QuantumToolExecutor()
    
    # Define analysis tools
    def data_analysis(data: str) -> Dict[str, Any]:
        """Analyze data statistically."""
        return {
            'mean': 42.5,
            'std_dev': 12.3,
            'outliers': 2,
            'distribution': 'normal',
            'confidence': 0.95
        }
    
    def sentiment_analysis(text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        return {
            'sentiment': 'positive',
            'confidence': 0.87,
            'emotions': ['optimism', 'excitement', 'curiosity'],
            'intensity': 0.72
        }
    
    def trend_analysis(data: str) -> Dict[str, Any]:
        """Analyze trends in data."""
        return {
            'trend': 'increasing',
            'slope': 0.15,
            'r_squared': 0.89,
            'forecast': 'continued_growth',
            'confidence_interval': [0.12, 0.18]
        }
    
    # Register tools
    for name, func in [('data_analysis', data_analysis), 
                       ('sentiment_analysis', sentiment_analysis),
                       ('trend_analysis', trend_analysis)]:
        executor.register_tool(
            name=name,
            function=func,
            description=f"Perform {name.replace('_', ' ')}",
            quantum_enhanced=True,
            parallel_execution=True,
            entanglement_enabled=True
        )
    
    # Parallel execution configuration
    parallel_configs = [
        {
            'name': 'data_analysis',
            'args': ['Quantum computing performance metrics over 5 years'],
            'kwargs': {}
        },
        {
            'name': 'sentiment_analysis', 
            'args': ['Quantum computing is revolutionizing AI and machine learning!'],
            'kwargs': {}
        },
        {
            'name': 'trend_analysis',
            'args': ['Quantum computing market growth data'],
            'kwargs': {}
        }
    ]
    
    # Execute tools in parallel with entanglement
    print("Executing parallel tools with quantum entanglement...")
    
    parallel_results = await executor.execute_parallel_tools(
        tool_configs=parallel_configs,
        entangle_results=True
    )
    
    print("\nParallel Execution Results:")
    total_time = sum(r.execution_time for r in parallel_results)
    max_time = max(r.execution_time for r in parallel_results)
    
    print(f"Total execution time: {total_time:.3f}s")
    print(f"Parallel execution time: {max_time:.3f}s")
    print(f"Speedup factor: {total_time/max_time:.2f}x")
    
    for result in parallel_results:
        print(f"\n{result.tool_name}:")
        print(f"  Result: {result.result}")
        print(f"  Quantum Enhanced: {result.quantum_enhanced}")
        print(f"  Entanglement ID: {result.entanglement_id}")

asyncio.run(parallel_tool_execution())
```

## Advanced Examples

### Quantum Context Management

```python
from quantumlangchain.context import QuantumContextManager, ContextScope

async def advanced_context_management():
    """Advanced quantum context management example."""
    
    context_manager = QuantumContextManager()
    
    # Create specialized context windows
    windows = {
        'research': context_manager.create_context_window(
            window_id="research_context",
            max_size=30,
            coherence_threshold=0.9
        ),
        'analysis': context_manager.create_context_window(
            window_id="analysis_context", 
            max_size=20,
            coherence_threshold=0.8
        ),
        'synthesis': context_manager.create_context_window(
            window_id="synthesis_context",
            max_size=15,
            coherence_threshold=0.85
        )
    }
    
    # Set hierarchical context
    contexts = [
        (ContextScope.GLOBAL, "project_domain", "quantum_computing_research"),
        (ContextScope.SESSION, "current_phase", "literature_review"),
        (ContextScope.CONVERSATION, "focus_area", "quantum_algorithms"),
        (ContextScope.TURN, "specific_topic", "variational_quantum_eigensolvers")
    ]
    
    for scope, key, value in contexts:
        await context_manager.set_context(
            scope=scope,
            key=key,
            value=value,
            quantum_enhanced=True
        )
    
    # Create multi-level entanglements
    research_contexts = [
        (ContextScope.SESSION, "current_phase"),
        (ContextScope.CONVERSATION, "focus_area"),
        (ContextScope.TURN, "specific_topic")
    ]
    
    research_entanglement = await context_manager.entangle_contexts(
        context_keys=research_contexts,
        entanglement_strength=0.9
    )
    
    print(f"Created research entanglement: {research_entanglement}")
    
    # Demonstrate context evolution
    evolution_steps = [
        ("literature_review", "quantum_algorithms", "variational_quantum_eigensolvers"),
        ("experiment_design", "quantum_circuits", "ansatz_optimization"),
        ("data_analysis", "performance_metrics", "convergence_analysis"),
        ("synthesis", "conclusions", "future_directions")
    ]
    
    for phase, area, topic in evolution_steps:
        # Update context
        await context_manager.set_context(
            ContextScope.SESSION, "current_phase", phase, quantum_enhanced=True
        )
        await context_manager.set_context(
            ContextScope.CONVERSATION, "focus_area", area, quantum_enhanced=True
        )
        await context_manager.set_context(
            ContextScope.TURN, "specific_topic", topic, quantum_enhanced=True
        )
        
        # Create snapshot
        snapshot_id = await context_manager.create_snapshot(
            scope=ContextScope.SESSION,
            include_windows=True
        )
        
        print(f"\nPhase: {phase}")
        print(f"  Focus: {area} -> {topic}")
        print(f"  Snapshot: {snapshot_id}")
        
        # Retrieve quantum-enhanced context
        current_context = await context_manager.get_context(
            scope=ContextScope.CONVERSATION,
            key="focus_area",
            quantum_search=True
        )
        print(f"  Current focus (quantum): {current_context}")

asyncio.run(advanced_context_management())
```

### Complex Prompt Orchestration

```python
from quantumlangchain.prompts import QPromptChain, PromptType

async def complex_prompt_orchestration():
    """Complex prompt orchestration with quantum selection."""
    
    prompt_chain = QPromptChain()
    
    # Create adaptive prompt system
    prompt_configs = [
        # System prompts
        {
            'content': "You are an expert quantum computing researcher with deep knowledge of algorithms and applications.",
            'type': PromptType.SYSTEM,
            'weight': 1.0,
            'conditions': {'role': 'researcher'},
            'metadata': {'expertise': 'high', 'domain': 'quantum_computing'}
        },
        {
            'content': "You are a practical quantum software engineer focused on implementation and optimization.",
            'type': PromptType.SYSTEM,
            'weight': 0.9,
            'conditions': {'role': 'engineer'},
            'metadata': {'expertise': 'high', 'domain': 'quantum_software'}
        },
        
        # Analysis prompts
        {
            'content': "Analyze {topic} from a theoretical quantum computing perspective, focusing on mathematical foundations and algorithmic complexity.",
            'type': PromptType.USER,
            'weight': 1.0,
            'conditions': {'analysis_type': 'theoretical'},
            'metadata': {'style': 'academic', 'depth': 'deep'}
        },
        {
            'content': "Examine {topic} from a practical implementation standpoint, considering current hardware limitations and near-term applications.",
            'type': PromptType.USER,
            'weight': 0.8,
            'conditions': {'analysis_type': 'practical'},
            'metadata': {'style': 'applied', 'depth': 'moderate'}
        },
        {
            'content': "Explore {topic} creatively, considering unconventional approaches and future possibilities beyond current constraints.",
            'type': PromptType.USER,
            'weight': 0.7,
            'conditions': {'analysis_type': 'creative'},
            'metadata': {'style': 'innovative', 'depth': 'exploratory'}
        },
        
        # Synthesis prompts
        {
            'content': "Synthesize the analysis into actionable insights, highlighting key implications and recommended next steps.",
            'type': PromptType.USER,
            'weight': 0.9,
            'conditions': {'phase': 'synthesis'},
            'metadata': {'output': 'actionable', 'format': 'structured'}
        }
    ]
    
    # Add prompts and create groups
    prompt_ids = []
    for config in prompt_configs:
        prompt_id = prompt_chain.add_prompt(
            content=config['content'],
            prompt_type=config['type'],
            quantum_weight=config['weight'],
            conditions=config['conditions'],
            metadata=config['metadata']
        )
        prompt_ids.append(prompt_id)
    
    # Create prompt groups
    system_prompts = prompt_ids[:2]
    analysis_prompts = prompt_ids[2:5]
    synthesis_prompts = prompt_ids[5:]
    
    prompt_chain.create_superposition_group(
        group_name="system_role",
        prompt_ids=system_prompts,
        selection_method="quantum_interference"
    )
    
    prompt_chain.create_superposition_group(
        group_name="analysis_approach",
        prompt_ids=analysis_prompts,
        selection_method="amplitude_amplification"
    )
    
    # Create entanglements between related prompts
    theoretical_practical_entanglement = prompt_chain.entangle_prompts(
        prompt_ids=[prompt_ids[2], prompt_ids[3]],  # theoretical + practical
        entanglement_strength=0.8
    )
    
    # Create adaptive prompt chains
    research_chain = prompt_chain.create_prompt_chain(
        chain_name="quantum_research_analysis",
        prompt_ids=["system_role", "analysis_approach"] + synthesis_prompts,
        allow_quantum_selection=True
    )
    
    # Execute with different contexts
    contexts = [
        {
            'role': 'researcher',
            'analysis_type': 'theoretical',
            'phase': 'analysis',
            'user_expertise': 'expert',
            'time_constraint': 'none'
        },
        {
            'role': 'engineer', 
            'analysis_type': 'practical',
            'phase': 'synthesis',
            'user_expertise': 'intermediate',
            'time_constraint': 'moderate'
        },
        {
            'role': 'researcher',
            'analysis_type': 'creative',
            'phase': 'analysis',
            'user_expertise': 'expert', 
            'time_constraint': 'low'
        }
    ]
    
    topic = "quantum error correction in NISQ devices"
    
    print("Adaptive Quantum Prompt Orchestration:")
    print(f"Topic: {topic}\n")
    
    for i, context in enumerate(contexts, 1):
        variables = {'topic': topic}
        
        result = await prompt_chain.execute_prompt_chain(
            chain_name="quantum_research_analysis",
            context=context,
            variables=variables
        )
        
        print(f"Context {i}: {context['role']} - {context['analysis_type']}")
        print(f"Selected Prompts: {result.selected_prompts}")
        print(f"Quantum Coherence: {result.quantum_coherence:.3f}")
        print(f"Selection Confidence: {result.selection_confidence:.3f}")
        print(f"Final Prompt Preview: {result.final_prompt[:100]}...")
        print(f"Entanglement Effects: {len(result.entanglement_influences)} detected")
        print()

asyncio.run(complex_prompt_orchestration())
```

## Real-World Application Examples

### Quantum-Enhanced RAG System

```python
async def quantum_rag_system():
    """Complete quantum-enhanced RAG system example."""
    
    from quantumlangchain import QuantumLangChain, QuantumConfig
    from quantumlangchain.vectorstores import HybridChromaDB
    from quantumlangchain.retrievers import QuantumRetriever
    
    # Initialize quantum RAG system
    config = QuantumConfig(
        num_qubits=8,
        backend_type="qiskit",
        decoherence_threshold=0.1,
        shots=2048
    )
    
    qlc = QuantumLangChain(config=config)
    
    # Setup quantum vector store
    vectorstore = HybridChromaDB(
        collection_name="quantum_knowledge_base",
        config={'quantum_enhanced': True, 'entanglement_enabled': True}
    )
    
    # Knowledge base documents
    knowledge_base = [
        {
            'content': "Variational Quantum Eigensolvers (VQE) are hybrid quantum-classical algorithms designed to find ground state energies of molecular systems.",
            'metadata': {'type': 'algorithm', 'domain': 'quantum_chemistry', 'complexity': 'intermediate'}
        },
        {
            'content': "Quantum Approximate Optimization Algorithm (QAOA) tackles combinatorial optimization problems using quantum gates and classical optimization.",
            'metadata': {'type': 'algorithm', 'domain': 'optimization', 'complexity': 'advanced'}
        },
        {
            'content': "Quantum machine learning leverages quantum superposition and entanglement to potentially achieve exponential speedups in certain learning tasks.",
            'metadata': {'type': 'application', 'domain': 'machine_learning', 'complexity': 'expert'}
        },
        {
            'content': "Noisy Intermediate-Scale Quantum (NISQ) devices represent the current era of quantum computing with limited qubits and high error rates.",
            'metadata': {'type': 'hardware', 'domain': 'quantum_devices', 'complexity': 'beginner'}
        }
    ]
    
    # Populate knowledge base
    doc_ids = []
    for doc in knowledge_base:
        ids = await vectorstore.add_documents(
            documents=[doc['content']],
            metadatas=[doc['metadata']],
            quantum_enhanced=True
        )
        doc_ids.extend(ids)
    
    # Create domain entanglements
    algorithm_docs = [doc_ids[0], doc_ids[1]]  # VQE and QAOA
    await vectorstore.entangle_documents(
        doc_ids=algorithm_docs,
        entanglement_strength=0.9
    )
    
    # Setup quantum retriever
    retriever = QuantumRetriever(
        vectorstore=vectorstore,
        config={
            'quantum_enhanced': True,
            'retrieval_algorithm': 'amplitude_amplification',
            'max_retrievals': 3
        }
    )
    
    # Create quantum RAG chain
    rag_chain = qlc.create_rag_chain(
        retriever=retriever,
        config={
            'enable_quantum_reasoning': True,
            'context_entanglement': True,
            'adaptive_retrieval': True
        }
    )
    
    # Test queries
    queries = [
        "How do variational quantum algorithms work for optimization problems?",
        "What are the challenges with current quantum devices for machine learning?",
        "Explain the connection between QAOA and quantum chemistry applications."
    ]
    
    print("Quantum-Enhanced RAG System Results:")
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        
        # Execute quantum RAG
        result = await rag_chain.arun(
            query=query,
            quantum_enhanced=True,
            enable_entanglement=True
        )
        
        print(f"Answer: {result['answer']}")
        print(f"Sources: {len(result['source_documents'])} documents")
        print(f"Quantum Coherence: {result['quantum_coherence']:.3f}")
        print(f"Retrieval Confidence: {result['retrieval_confidence']:.3f}")
        
        # Show retrieved documents
        for j, doc in enumerate(result['source_documents']):
            print(f"  Source {j+1}: {doc.metadata['type']} ({doc.metadata['domain']})")

asyncio.run(quantum_rag_system())
```

This comprehensive examples collection demonstrates the full capabilities of QuantumLangChain across all major use cases, from basic quantum chains to complex real-world applications.
