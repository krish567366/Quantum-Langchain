# API Reference

Complete API documentation for QuantumLangChain components.

## Core Components

### QuantumBase

Base class for all quantum-enhanced components.

```python
class QuantumBase:
    """Abstract base class providing quantum state management."""
```

#### Methods

##### `__init__(config: Optional[QuantumConfig] = None)`

Initialize quantum component with optional configuration.

**Parameters:**

- `config`: Quantum configuration settings

##### `async initialize() -> None`

Initialize quantum state and resources.

##### `async reset_quantum_state() -> None`

Reset component to coherent quantum state.

##### `create_entanglement(other: QuantumBase, strength: float = 0.8) -> str`

Create quantum entanglement with another component.

**Parameters:**

- `other`: Target component for entanglement
- `strength`: Entanglement strength (0.0 to 1.0)

**Returns:**

- Entanglement ID string

##### `update_decoherence(delta: float) -> None`

Update decoherence level.

**Parameters:**

- `delta`: Decoherence increment

#### Properties

- `quantum_state: QuantumState` - Current quantum state
- `decoherence_level: float` - Decoherence level (0.0 to 1.0)
- `entanglement_registry: Dict[str, Any]` - Registry of entanglements

---

## Chains

### QLChain

Quantum-enhanced reasoning chain with superposition and entanglement.

```python
class QLChain(QuantumBase):
    """Quantum-ready chain with parallel execution branches."""
```

#### Methods

##### `__init__(memory: QuantumMemory, backend: QuantumBackend, config: Optional[Dict] = None)`

Initialize quantum chain.

**Parameters:**

- `memory`: Quantum memory system
- `backend`: Quantum computing backend
- `config`: Chain configuration

##### `async arun(input_text: str, **kwargs) -> Dict[str, Any]`

Execute chain with quantum enhancement.

**Parameters:**

- `input_text`: Input query or prompt
- `**kwargs`: Additional execution parameters

**Returns:**

- Execution result dictionary

##### `async abatch(inputs: List[str], **kwargs) -> List[Dict[str, Any]]`

Batch process multiple inputs.

**Parameters:**

- `inputs`: List of input texts
- `**kwargs`: Execution parameters

**Returns:**

- List of results

##### `async astream(input_text: str, **kwargs) -> AsyncIterator[Dict[str, Any]]`

Stream chain execution results.

**Parameters:**

- `input_text`: Input query
- `**kwargs`: Execution parameters

**Yields:**

- Incremental results

##### `get_execution_stats() -> Dict[str, Any]`

Get chain execution statistics.

**Returns:**

- Statistics dictionary

---

## Memory Systems

### QuantumMemory

Quantum-enhanced memory with entanglement and reversibility.

```python
class QuantumMemory(QuantumBase):
    """Reversible, entangled memory system."""
```

#### Methods

##### `__init__(classical_dim: int, quantum_dim: int, backend: Optional[QuantumBackend] = None)`

Initialize quantum memory.

**Parameters:**

- `classical_dim`: Classical embedding dimension
- `quantum_dim`: Quantum register size
- `backend`: Quantum backend

##### `async store(key: str, value: Any, quantum_enhanced: bool = False) -> None`

Store memory entry.

**Parameters:**

- `key`: Memory key
- `value`: Value to store
- `quantum_enhanced`: Enable quantum enhancement

##### `async retrieve(key: str, quantum_search: bool = False) -> Any`

Retrieve memory entry.

**Parameters:**

- `key`: Memory key
- `quantum_search`: Use quantum search

**Returns:**

- Retrieved value or None

##### `async similarity_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]`

Search memory by similarity.

**Parameters:**

- `query`: Search query
- `top_k`: Number of results

**Returns:**

- List of similar entries

##### `async entangle_memories(keys: List[str]) -> str`

Create entanglement between memory entries.

**Parameters:**

- `keys`: List of memory keys

**Returns:**

- Entanglement ID

##### `async create_memory_snapshot() -> str`

Create reversible memory snapshot.

**Returns:**

- Snapshot ID

##### `async restore_memory_snapshot(snapshot_id: str) -> bool`

Restore memory from snapshot.

**Parameters:**

- `snapshot_id`: Snapshot identifier

**Returns:**

- Success status

##### `async get_stats() -> Dict[str, Any]`

Get memory statistics.

**Returns:**

- Statistics dictionary

---

## Agents

### EntangledAgents

Multi-agent system with quantum collaboration.

```python
class EntangledAgents(QuantumBase):
    """Entangled agent system for collaborative problem solving."""
```

#### Methods

##### `__init__(agent_configs: List[Dict[str, Any]], backend: Optional[QuantumBackend] = None)`

Initialize entangled agent system.

**Parameters:**

- `agent_configs`: List of agent configurations
- `backend`: Quantum backend

##### `async collaborative_solve(problem: str, max_iterations: int = 5, enable_interference: bool = True) -> Dict[str, Any]`

Solve problem collaboratively.

**Parameters:**

- `problem`: Problem description
- `max_iterations`: Maximum solving iterations
- `enable_interference`: Enable quantum interference

**Returns:**

- Solution with collaboration details

##### `async run_single_agent(agent_id: str, problem: str) -> Dict[str, Any]`

Run single agent on problem.

**Parameters:**

- `agent_id`: Agent identifier
- `problem`: Problem description

**Returns:**

- Agent solution

##### `async run_parallel_agents(agent_ids: List[str], problem: str) -> List[Dict[str, Any]]`

Run multiple agents in parallel.

**Parameters:**

- `agent_ids`: List of agent identifiers
- `problem`: Problem description

**Returns:**

- List of agent solutions

##### `async propagate_belief(belief: BeliefState) -> List[BeliefState]`

Propagate belief across agents.

**Parameters:**

- `belief`: Initial belief state

**Returns:**

- List of propagated beliefs

##### `add_agent_role(role: AgentRole) -> None`

Add new agent role.

**Parameters:**

- `role`: Agent role configuration

##### `update_agent_role(agent_id: str, **kwargs) -> None`

Update existing agent role.

**Parameters:**

- `agent_id`: Agent identifier
- `**kwargs`: Role updates

##### `get_performance_stats() -> Dict[str, Any]`

Get agent performance statistics.

**Returns:**

- Performance statistics

---

## Retrievers

### QuantumRetriever

Quantum-enhanced document retrieval system.

```python
class QuantumRetriever(QuantumBase):
    """Quantum-enhanced semantic retrieval with Grover's algorithm."""
```

#### Methods

##### `__init__(vectorstore: Any, backend: Optional[QuantumBackend] = None, config: Optional[Dict] = None)`

Initialize quantum retriever.

**Parameters:**

- `vectorstore`: Vector store backend
- `backend`: Quantum backend
- `config`: Retriever configuration

##### `async aretrieve(query: str, quantum_enhanced: bool = True, **kwargs) -> List[Document]`

Retrieve relevant documents.

**Parameters:**

- `query`: Search query
- `quantum_enhanced`: Use quantum enhancement
- `**kwargs`: Additional parameters

**Returns:**

- List of relevant documents

##### `async asimilarity_search(query: str, k: int = 5, algorithm: str = "amplitude_amplification") -> List[Tuple[Document, float]]`

Perform quantum similarity search.

**Parameters:**

- `query`: Search query
- `k`: Number of results
- `algorithm`: Quantum algorithm to use

**Returns:**

- List of documents with scores

##### `get_retrieval_stats() -> Dict[str, Any]`

Get retrieval statistics.

**Returns:**

- Retrieval statistics

---

## Vector Stores

### HybridChromaDB

Quantum-enhanced ChromaDB vector store.

```python
class HybridChromaDB(QuantumBase):
    """Hybrid ChromaDB with quantum similarity search."""
```

#### Methods

##### `__init__(collection_name: str = "quantum_documents", persist_directory: Optional[str] = None, embedding_function: Optional[Any] = None, config: Optional[Dict] = None)`

Initialize hybrid ChromaDB.

**Parameters:**

- `collection_name`: Collection name
- `persist_directory`: Persistence directory
- `embedding_function`: Embedding function
- `config`: Configuration

##### `async add_documents(documents: List[str], metadatas: Optional[List[Dict]] = None, ids: Optional[List[str]] = None, embeddings: Optional[List[List[float]]] = None, quantum_enhanced: bool = False) -> List[str]`

Add documents to collection.

**Parameters:**

- `documents`: List of document texts
- `metadatas`: Document metadata
- `ids`: Document IDs
- `embeddings`: Pre-computed embeddings
- `quantum_enhanced`: Enable quantum enhancement

**Returns:**

- List of document IDs

##### `async similarity_search(query: str, k: int = 5, filter: Optional[Dict] = None, quantum_enhanced: bool = False) -> List[Tuple[QuantumDocument, float]]`

Search for similar documents.

**Parameters:**

- `query`: Search query
- `k`: Number of results
- `filter`: Metadata filter
- `quantum_enhanced`: Use quantum enhancement

**Returns:**

- List of documents with similarity scores

##### `async entangle_documents(doc_ids: List[str], entanglement_strength: float = 0.8) -> str`

Create document entanglement.

**Parameters:**

- `doc_ids`: Document IDs to entangle
- `entanglement_strength`: Entanglement strength

**Returns:**

- Entanglement ID

##### `async quantum_similarity_search(query: str, k: int = 5, quantum_algorithm: str = "amplitude_amplification") -> List[Tuple[QuantumDocument, float]]`

Quantum-enhanced similarity search.

**Parameters:**

- `query`: Search query
- `k`: Number of results
- `quantum_algorithm`: Quantum algorithm

**Returns:**

- Enhanced search results

### QuantumFAISS

Quantum-enhanced FAISS vector store.

```python
class QuantumFAISS(QuantumBase):
    """FAISS vector store with quantum amplitude amplification."""
```

#### Methods

##### `__init__(dimension: int, index_type: str = "IVFFlat", metric: str = "L2", nlist: int = 100, persist_path: Optional[str] = None, config: Optional[Dict] = None)`

Initialize quantum FAISS.

**Parameters:**

- `dimension`: Vector dimension
- `index_type`: FAISS index type
- `metric`: Distance metric
- `nlist`: Number of clusters
- `persist_path`: Persistence path
- `config`: Configuration

##### `async add_vectors(vectors: Union[np.ndarray, List[List[float]]], ids: Optional[List[str]] = None, metadatas: Optional[List[Dict]] = None, quantum_enhanced: bool = False) -> List[str]`

Add vectors to index.

**Parameters:**

- `vectors`: Vector data
- `ids`: Vector IDs
- `metadatas`: Vector metadata
- `quantum_enhanced`: Enable quantum enhancement

**Returns:**

- List of vector IDs

##### `async search(query_vector: Union[np.ndarray, List[float]], k: int = 5, quantum_enhanced: bool = False, filter_metadata: Optional[Dict] = None) -> List[Tuple[str, float, Dict]]`

Search for similar vectors.

**Parameters:**

- `query_vector`: Query vector
- `k`: Number of results
- `quantum_enhanced`: Use quantum enhancement
- `filter_metadata`: Metadata filter

**Returns:**

- Search results with scores

##### `async amplitude_amplification_search(query_vector: Union[np.ndarray, List[float]], target_condition: Callable, k: int = 5, iterations: int = 3) -> List[Tuple[str, float, Dict]]`

Amplitude amplification search.

**Parameters:**

- `query_vector`: Query vector
- `target_condition`: Target condition function
- `k`: Number of results
- `iterations`: Amplification iterations

**Returns:**

- Amplified search results

##### `async grovers_search(oracle_function: Callable, k: int = 5, max_iterations: int = 10) -> List[Tuple[str, float, Dict]]`

Grover's algorithm search.

**Parameters:**

- `oracle_function`: Oracle function
- `k`: Number of results
- `max_iterations`: Maximum iterations

**Returns:**

- Grover's search results

---

## Tools

### QuantumToolExecutor

Quantum-enhanced tool execution system.

```python
class QuantumToolExecutor(QuantumBase):
    """Quantum tool execution with entangled tool chaining."""
```

#### Methods

##### `register_tool(name: str, function: Callable, description: str, quantum_enhanced: bool = False, parallel_execution: bool = False, entanglement_enabled: bool = False) -> None`

Register a tool.

**Parameters:**

- `name`: Tool name
- `function`: Tool function
- `description`: Tool description
- `quantum_enhanced`: Enable quantum enhancement
- `parallel_execution`: Allow parallel execution
- `entanglement_enabled`: Enable entanglement

##### `async execute_tool(tool_name: str, *args, quantum_enhanced: Optional[bool] = None, **kwargs) -> ToolResult`

Execute a single tool.

**Parameters:**

- `tool_name`: Name of tool to execute
- `*args`: Tool arguments
- `quantum_enhanced`: Override quantum enhancement
- `**kwargs`: Tool keyword arguments

**Returns:**

- Tool execution result

##### `async execute_parallel_tools(tool_configs: List[Dict[str, Any]], entangle_results: bool = False) -> List[ToolResult]`

Execute multiple tools in parallel.

**Parameters:**

- `tool_configs`: Tool configuration list
- `entangle_results`: Entangle results

**Returns:**

- List of tool results

##### `async execute_quantum_superposition_tools(tool_configs: List[Dict[str, Any]], measurement_function: Optional[Callable] = None) -> ToolResult`

Execute tools in quantum superposition.

**Parameters:**

- `tool_configs`: Tool configurations
- `measurement_function`: Result measurement function

**Returns:**

- Measured tool result

##### `create_tool_chain(chain_name: str, tool_names: List[str]) -> None`

Create a tool execution chain.

**Parameters:**

- `chain_name`: Chain name
- `tool_names`: List of tool names

##### `async execute_tool_chain(chain_name: str, initial_input: Any = None, propagate_results: bool = True) -> List[ToolResult]`

Execute a tool chain.

**Parameters:**

- `chain_name`: Chain to execute
- `initial_input`: Initial input data
- `propagate_results`: Propagate results between tools

**Returns:**

- List of chain results

---

## Context Management

### QuantumContextManager

Quantum-enhanced context management with temporal snapshots.

```python
class QuantumContextManager(QuantumBase):
    """Quantum context management with coherent state tracking."""
```

#### Methods

##### `create_context_window(window_id: str, max_size: int = 100, coherence_threshold: float = 0.8) -> ContextWindow`

Create a context window.

**Parameters:**

- `window_id`: Window identifier
- `max_size`: Maximum window size
- `coherence_threshold`: Coherence threshold

**Returns:**

- Context window instance

##### `async set_context(scope: ContextScope, key: str, value: Any, quantum_enhanced: bool = False, window_id: Optional[str] = None) -> None`

Set context value.

**Parameters:**

- `scope`: Context scope
- `key`: Context key
- `value`: Context value
- `quantum_enhanced`: Enable quantum enhancement
- `window_id`: Target window ID

##### `async get_context(scope: ContextScope, key: str, default: Any = None, quantum_search: bool = False) -> Any`

Get context value.

**Parameters:**

- `scope`: Context scope
- `key`: Context key
- `default`: Default value
- `quantum_search`: Use quantum search

**Returns:**

- Context value

##### `async create_snapshot(scope: ContextScope = ContextScope.SESSION, include_windows: bool = True) -> str`

Create context snapshot.

**Parameters:**

- `scope`: Snapshot scope
- `include_windows`: Include context windows

**Returns:**

- Snapshot ID

##### `async restore_snapshot(snapshot_id: str) -> bool`

Restore from snapshot.

**Parameters:**

- `snapshot_id`: Snapshot identifier

**Returns:**

- Success status

##### `async entangle_contexts(context_keys: List[Tuple[ContextScope, str]], entanglement_strength: float = 0.8) -> str`

Entangle context items.

**Parameters:**

- `context_keys`: List of context keys with scopes
- `entanglement_strength`: Entanglement strength

**Returns:**

- Entanglement ID

---

## Prompts

### QPromptChain

Quantum-enhanced prompt chaining system.

```python
class QPromptChain(QuantumBase):
    """Quantum prompt chaining with superposition-based selection."""
```

#### Methods

##### `add_prompt(content: str, prompt_type: PromptType = PromptType.USER, quantum_weight: float = 1.0, conditions: Optional[Dict] = None, metadata: Optional[Dict] = None) -> str`

Add a prompt to the collection.

**Parameters:**

- `content`: Prompt content
- `prompt_type`: Type of prompt
- `quantum_weight`: Quantum selection weight
- `conditions`: Conditional logic
- `metadata`: Prompt metadata

**Returns:**

- Prompt ID

##### `create_prompt_chain(chain_name: str, prompt_ids: List[str], allow_quantum_selection: bool = True) -> None`

Create a prompt chain.

**Parameters:**

- `chain_name`: Chain name
- `prompt_ids`: List of prompt IDs
- `allow_quantum_selection`: Enable quantum selection

##### `create_superposition_group(group_name: str, prompt_ids: List[str], selection_method: str = "quantum_interference") -> None`

Create superposition group.

**Parameters:**

- `group_name`: Group name
- `prompt_ids`: Prompt IDs
- `selection_method`: Selection algorithm

##### `async execute_prompt_chain(chain_name: str, context: Dict[str, Any], variables: Optional[Dict[str, Any]] = None) -> PromptChainResult`

Execute prompt chain.

**Parameters:**

- `chain_name`: Chain to execute
- `context`: Execution context
- `variables`: Template variables

**Returns:**

- Chain execution result

##### `entangle_prompts(prompt_ids: List[str], entanglement_strength: float = 0.8) -> str`

Entangle prompts.

**Parameters:**

- `prompt_ids`: Prompt IDs to entangle
- `entanglement_strength`: Entanglement strength

**Returns:**

- Entanglement ID

---

## Backends

### QuantumBackend

Abstract base class for quantum backends.

```python
class QuantumBackend:
    """Abstract interface for quantum computing backends."""
```

#### Methods

##### `async execute_circuit(circuit: Any, shots: int = 1000) -> Dict[str, Any]`

Execute quantum circuit.

**Parameters:**

- `circuit`: Quantum circuit
- `shots`: Number of measurements

**Returns:**

- Execution results

##### `async create_entangling_circuit(qubits: List[int]) -> Any`

Create entangling circuit.

**Parameters:**

- `qubits`: Qubit indices

**Returns:**

- Quantum circuit

##### `get_backend_info() -> Dict[str, Any]`

Get backend information.

**Returns:**

- Backend capabilities and info

### QiskitBackend

IBM Qiskit quantum backend.

```python
class QiskitBackend(QuantumBackend):
    """Qiskit quantum computing backend."""
```

### PennyLaneBackend

Xanadu PennyLane quantum backend.

```python
class PennyLaneBackend(QuantumBackend):
    """PennyLane quantum machine learning backend."""
```

### BraketBackend

Amazon Braket quantum backend.

```python
class BraketBackend(QuantumBackend):
    """Amazon Braket quantum computing backend."""
```

---

## Configuration

### QuantumConfig

Configuration class for quantum parameters.

```python
@dataclass
class QuantumConfig:
    """Configuration for quantum components."""
    num_qubits: int = 4
    circuit_depth: int = 10
    decoherence_threshold: float = 0.1
    backend_type: str = "qiskit"
    shots: int = 1000
    optimization_level: int = 1
    enable_error_correction: bool = False
```

---

## Data Classes

### Document

Document class for retrieved content.

```python
@dataclass
class Document:
    """Document with content and metadata."""
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### ToolResult

Result from tool execution.

```python
@dataclass
class ToolResult:
    """Tool execution result."""
    tool_name: str
    result: Any
    success: bool
    execution_time: float
    quantum_enhanced: bool = False
    entanglement_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## Enums

### QuantumState

Quantum state enumeration.

```python
class QuantumState(Enum):
    COHERENT = "coherent"
    SUPERPOSITION = "superposition"
    ENTANGLED = "entangled"
    COLLAPSED = "collapsed"
    DECOHERENT = "decoherent"
```

### ContextScope

Context scope levels.

```python
class ContextScope(Enum):
    GLOBAL = "global"
    SESSION = "session"
    CONVERSATION = "conversation"
    TURN = "turn"
    QUANTUM_STATE = "quantum_state"
```

### PromptType

Prompt type enumeration.

```python
class PromptType(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    QUANTUM_SUPERPOSITION = "quantum_superposition"
    ENTANGLED = "entangled"
    CONDITIONAL = "conditional"
```
