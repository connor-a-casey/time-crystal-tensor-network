<div align="center">
    <h1>
        <img src="assets/header.jpg">
    </h1>
</div>


# Discrete Time Crystals for Quantum Memories: A Tensor-Network Approach
![Paper Status](https://img.shields.io/badge/paper-published-yellow)
![Tests](https://img.shields.io/badge/tests-49%20passing-brightgreen)
![Code Size](https://img.shields.io/github/languages/code-size/ccasey/time-crystal-tensor-network)
![Repo Size](https://img.shields.io/github/repo-size/ccasey/time-crystal-tensor-network)
[![Parameters Documented](https://img.shields.io/badge/parameters-documented-blue)](#simulation-parameters)

## Abstract

Quantum memories are essential components in applications ranging from quantum computing to quantum communication networks. However, their practical utility is constrained by short coherence times, motivating the search for new physical systems that can inherently protect stored information. Discrete time crystals (DTCs)—periodically driven many-body systems exhibiting stable subharmonic oscillations that break discrete time-translation symmetry—offer a promising approach, as they are theoretically able to shield encoded information from local perturbations, making them compelling candidates for next-generation, passively protected quantum memories.

In this work, we employ a tensor-network framework that models a quantum memory as a DTC. We employ the time-evolving block-decimation (TEBD) algorithm to perform both real- and imaginary-time evolution of a matrix-product-state (MPS) representation, thereby efficiently capturing the large many-body Hilbert space while tracking entanglement growth, sub-harmonic spectral responses, and memory-fidelity metrics over experimentally relevant timescales. By sweeping the drive strength, interaction range, and disorder, we map the phase diagram, pinpoint regimes that sustain time-crystalline order, and set the stage to model their coherence lifetimes.

## 🔬 Overview

Quantum memories are essential for quantum computing and communication networks, but their practical utility is limited by short coherence times. This work explores **discrete time crystals**—periodically driven many-body systems that exhibit stable subharmonic oscillations and break discrete time-translation symmetry—as a promising approach for passively protected quantum memories.

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
├── config.txt                   # Simulation parameters
├── run_all_tests.sh             # Easy test runner script
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
└── .gitignore                   # Git ignore patterns
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- NumPy, SciPy, matplotlib
- TensorFlow or PyTorch (for tensor operations)
- QuTiP (optional, for additional quantum tools)

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

## 📊 Usage Examples

### Configuration
The simulation parameters can be modified in `config.txt`:

```ini
# Physical parameters
J = 1.0                    # Ising coupling strength
THETA = 3.14159265359      # X-kick angle (π-pulse)
T_DRIVE = 2.0              # Drive period
H_MAX = 0.3                # Random field strength
GAMMA = 1e-3               # Dephasing rate

# System size
N_SITES_FIGURES = 64       # Chain length for figures

# Numerical parameters  
CHI_MAX = 256              # Maximum bond dimension
SVD_CUTOFF = 1e-7          # SVD truncation threshold
```
```

### Phase Diagram Generation
```python
from src.core.observables import compute_phase_diagram

# Define parameter ranges
h_range = np.linspace(0.1, 2.0, 20)  # Disorder strength
drive_range = np.linspace(0.5, 1.5, 20)  # Drive amplitude

# Generate phase diagram
phase_diagram = compute_phase_diagram(
    h_values=h_range,
    drive_values=drive_range,
    n_sites=10,
    time_steps=50
)
```

## 🧪 Testing & Validation

This repository includes a comprehensive test suite to ensure code reliability and physical correctness:

### Running Tests
```bash
# Run complete test suite (all 49 tests)
./run_all_tests.sh

# Or run the test runner directly
python tests/run_tests.py
```

### Test Categories

**Basic Functionality (21 tests)**
- Core tensor network operations
- Model initialization and evolution
- Observable calculations
- Parameter handling and configuration

**Physics Validation (16 tests)**
- Discrete time crystal signatures
- Period-doubling detection
- Many-body localization behavior
- Physical consistency checks

**Performance Benchmarks (12 tests)**
- Execution time monitoring
- Memory usage validation
- Scalability analysis
- Bond dimension scaling

### Advanced Usage
```bash
# For individual test modules (if needed)
python tests/test_basic_functionality.py    # Core functionality only
python tests/test_physics_validation.py     # Physics correctness only
python tests/test_performance.py            # Performance analysis only

# Generate detailed test report
python tests/run_tests.py --output test_report.txt
```

### Continuous Integration
The test suite is designed for automated validation:
- **Zero failures**: All 49 tests pass reliably
- **Comprehensive execution**: Full suite completes in ~2-3 minutes
- **Complete coverage**: Tests all major components, physics, and performance
- **Physics validation**: Verifies DTC signatures and physical consistency

## 🔬 Simulation Parameters

The framework uses extensively documented parameters for reproducible research:

| Parameter Category | Description | Typical Range |
|-------------------|-------------|---------------|
| **System Size** | Number of spins in the chain | 8-20 sites |
| **Disorder Strength** | Random field amplitude (h) | 0.1-2.0 |
| **Drive Amplitude** | Periodic drive strength | 0.5-1.5 |
| **Interaction Range** | Spin-spin coupling decay | Nearest neighbor to long-range |
| **Bond Dimension** | MPS truncation parameter | 32-256 |
| **Time Step** | Trotter step size | 0.05-0.2 |

All parameters are documented in `config.txt` with physical justifications and references.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this framework in your research, please cite our work:

### BibTeX Format
```bibtex
@software{casey2025dtc_code,
  author = {Casey, Connor},
  title = {Discrete Time Crystals for Quantum Memories: A Tensor-Network Approach},
  url = {https://github.com/ccasey/time-crystal-tensor-network},
  year = {2025},
  note = {Tensor network simulation framework for discrete time crystal quantum memories}
}
```

### APA Format
```
Casey, C. (2025). Discrete time crystals for quantum memories: A tensor-network approach [Computer software]. GitHub. https://github.com/ccasey/time-crystal-tensor-network
```

### Plain Text
```
Connor Casey. "Discrete Time Crystals for Quantum Memories: A Tensor-Network Approach." 
GitHub repository, 2025. https://github.com/ccasey/time-crystal-tensor-network
```

## 📧 Contact

<sup>1</sup> College of Information and Computer Sciences, University of Massachusetts Amherst, USA  
<sup>2</sup> Department of Physics, University of Massachusetts Amherst, USA

- [**Connor Casey**](mailto:ccasey@umass.edu)<sup>1,2</sup>
---