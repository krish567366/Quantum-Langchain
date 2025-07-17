"""
Pytest configuration file for QuantumLangChain test suite.
"""

import pytest
import asyncio
import warnings
from pathlib import Path


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "quantum: marks tests that require quantum backends"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU acceleration"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark slow tests
        if "slow" in item.nodeid or "integration" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Mark quantum tests
        if any(keyword in item.nodeid.lower() for keyword in ["quantum", "backend", "circuit"]):
            item.add_marker(pytest.mark.quantum)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config():
    """Test configuration settings."""
    return {
        "test_quantum_backends": ["qiskit"],  # Only test available backends
        "max_test_qubits": 8,
        "test_timeout": 30,
        "mock_expensive_operations": True,
    }


@pytest.fixture
def mock_quantum_backend():
    """Mock quantum backend for testing without real quantum hardware."""
    from unittest.mock import Mock, AsyncMock
    
    backend = Mock()
    backend.execute_circuit = AsyncMock(return_value={
        "success": True,
        "counts": {"00": 50, "11": 50},
        "probabilities": [0.5, 0.0, 0.0, 0.5],
        "shots": 100,
        "circuit_depth": 2,
        "execution_time": 0.1
    })
    backend.create_entangling_circuit = AsyncMock()
    backend.get_backend_info = Mock(return_value={
        "backend_name": "mock_simulator",
        "provider": "mock",
        "num_qubits": 32,
        "quantum_volume": 64
    })
    return backend


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing vector operations."""
    import numpy as np
    return {
        "query": np.random.rand(128).astype(np.float32),
        "docs": [np.random.rand(128).astype(np.float32) for _ in range(10)],
        "metadata": [{"id": i, "text": f"Document {i}"} for i in range(10)]
    }


@pytest.fixture
def temp_test_dir(tmp_path):
    """Create temporary directory for test files."""
    test_dir = tmp_path / "quantumlangchain_tests"
    test_dir.mkdir()
    return test_dir


# Suppress specific warnings during tests
@pytest.fixture(autouse=True)
def suppress_warnings():
    """Automatically suppress common warnings during tests."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    warnings.filterwarnings("ignore", message=".*gymnasium.*")
    warnings.filterwarnings("ignore", message=".*qiskit.*")


# Skip tests that require specific dependencies
def pytest_runtest_setup(item):
    """Skip tests based on missing dependencies."""
    # Skip quantum tests if no quantum backends available
    if item.get_closest_marker("quantum"):
        try:
            import qiskit
        except ImportError:
            pytest.skip("Quantum backend not available")
    
    # Skip GPU tests if no CUDA available
    if item.get_closest_marker("gpu"):
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")
        except ImportError:
            pytest.skip("PyTorch not available")


# Custom pytest hooks for better reporting
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add custom terminal summary."""
    if exitstatus == 0:
        terminalreporter.write_line(
            "🎉 All QuantumLangChain tests passed! Quantum coherence maintained.", 
            green=True
        )
    else:
        terminalreporter.write_line(
            "⚠️  Some tests failed. Check quantum decoherence levels.", 
            red=True
        )
