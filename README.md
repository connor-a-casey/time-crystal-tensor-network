<div align="center">
    <h1>
        <img src="assets/header.jpg">
    </h1>
</div>


# Discrete Time Crystals for Quantum Memories: A Tensor-Network Approach
![Paper Status](https://img.shields.io/badge/paper-published-yellow)
![Tests](https://img.shields.io/badge/tests-39%20passing-brightgreen)
![Code Size](https://img.shields.io/github/languages/code-size/connor-a-casey/time-crystal-tensor-network)
![Repo Size](https://img.shields.io/github/repo-size/connor-a-casey/time-crystal-tensor-network)
[![Parameters Documented](https://img.shields.io/badge/parameters-documented-blue)](config.txt)

## Abstract

Quantum memories are essential components in applications ranging from quantum computing to quantum communication networks. However, their practical utility is constrained by short coherence times, motivating the search for new physical systems that can inherently protect stored information. Discrete time crystals (DTCs)—periodically driven many-body systems exhibiting stable subharmonic oscillations that break discrete time-translation symmetry—offer a promising approach, as they are theoretically able to shield encoded information from local perturbations, making them compelling candidates for next-generation, passively protected quantum memories.

In this work, we employ a tensor-network framework that models a quantum memory as a DTC. We employ the time-evolving block-decimation (TEBD) algorithm to perform both real- and imaginary-time evolution of a matrix-product-state (MPS) representation, thereby efficiently capturing the large many-body Hilbert space while tracking entanglement growth, sub-harmonic spectral responses, and memory-fidelity metrics over experimentally relevant timescales. By sweeping the drive strength, interaction range, and disorder, we map the phase diagram, pinpoint regimes that sustain time-crystalline order, and set the stage to model their coherence lifetimes.

## 🔬 Overview

Our tensor-network framework provides:
- **Time-Evolving Block Decimation (TEBD)** for efficient many-body quantum simulation
- **Matrix Product State (MPS)** representation capturing large Hilbert spaces
- **Real and imaginary-time evolution** algorithms for ground state preparation and dynamics
- **Phase diagram mapping** across drive strength, interaction range, and disorder parameters
- **Spectral analysis tools** for identifying time-crystalline signatures


## 📁 Repository Structure

```
time-crystal-tensor-network/
├── src/
│   ├── core/                    # Core tensor network algorithms
│   │   ├── tensor_utils.py      # MPS/MPO operations and utilities
│   │   └── observables.py       # Measurement and correlation functions
│   ├── models/                  # Physical system implementations
│   │   └── kicked_ising.py      # Floquet kicked-Ising model
│   └── dynamics/                # Time evolution algorithms
│       └── tebd_evolution.py    # Time-evolving block decimation
├── figures/                     # Generated time crystal figures
│   ├── (initially empty)
├── tests/                       # Comprehensive test suite
│   ├── test_basic_functionality.py    # Core functionality tests
│   ├── test_physics_validation.py     # Physics correctness tests
│   ├── test_performance.py            # Performance benchmarks
│   └── run_tests.py                   # Test runner with reporting
├── assets/                      # Project assets
│   └── header.jpg               # README header image
├── main.py                      # Generate all figures
├── config.txt                   # Simulation parameters & citations
├── run_all_tests.sh             # Easy test runner script
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- NumPy, SciPy, matplotlib
- TensorFlow or PyTorch (for tensor operations)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/ccasey/time-crystal-tensor-network.git
cd time-crystal-tensor-network

# Install dependencies
pip install -r requirements.txt

# Verify installation with tests
./run_all_tests.sh

# Generate time crystal figures
python main.py
```

This will generate four key figures demonstrating discrete time crystal physics:
- `perfect_time_crystal.png/pdf` - Clean period-doubling oscillations
- `disordered_time_crystal.png/pdf` - DTC behavior under disorder
- `time_crystal_with_dephasing.png/pdf` - Open-system effects
- `multisite_time_crystal_dynamics.png/pdf` - Individual site dynamics

### Testing Your Installation
```bash
# Run comprehensive test suite to verify everything works
./run_all_tests.sh
```

## 🧪 Testing & Validation

This repository includes a comprehensive test suite with 39 tests across three modules to ensure code reliability and physical correctness. The test runner provides the following:

### Running Tests
```bash
# Run complete test suite (all 39 tests)
./run_all_tests.sh

# Or run the comprehensive test runner directly
python tests/run_tests.py

# The test runner automatically:
# - Checks dependencies (numpy, matplotlib, tenpy, etc.)
# - Validates core module imports
# - Runs all test modules with timing
# - Generates detailed success/failure reports
```

### Test Categories

**Basic Functionality (21 tests)**
- Core tensor network operations (Pauli matrices, MPS states)
- Model initialization and Floquet evolution
- Observable calculations (magnetization, Loschmidt echo)
- DTC detection and phase analysis
- Physics validation and conservation laws
- Integration workflows and error handling

**Physics Validation (9 tests)**
- Discrete time crystal signatures and period-doubling
- Many-body localization regime behavior
- Initial state dependence and physical consistency
- Tensor network properties (norm conservation, hermiticity)
- Parameter dependence (disorder, frequency, system size)

**Performance Benchmarks (9 tests)**
- Floquet step execution time scaling
- Memory usage monitoring and bond dimension scaling
- Phase point calculation performance
- Scalability limits and maximum system sizes
- Concurrent operations and memory optimization

### Advanced Usage
```bash
# Run with detailed output and reporting
python tests/run_tests.py --verbose --output test_report.txt

# For individual test modules (if needed)
python tests/test_basic_functionality.py    # Core functionality (21 tests)
python tests/test_physics_validation.py     # Physics correctness (9 tests)
python tests/test_performance.py            # Performance analysis (9 tests)

# Run tests with increased verbosity
python tests/run_tests.py -vv
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this framework in your research, please cite our work:

### BibTeX Format
```bibtex
@inproceedings{casey2025dtc_code,
  author = {Casey, Connor},
  title = {Discrete Time Crystals for Quantum Memories: A Tensor-Network Approach},
  url = {https://github.com/ccasey/time-crystal-tensor-network},
  year = {2025},
}
```

## 📧 Contact

**Connor Casey**  
📧 [cacasey@umass.edu](mailto:cacasey@umass.edu)  
🏛️ College of Information and Computer Sciences & Department of Physics  
🎓 University of Massachusetts Amherst, USA