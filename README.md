<div align="center">
    <h1>
        <img src="assets/header.jpg">
    </h1>
</div>


# Discrete Time Crystals for Quantum Memories: A Tensor-Network Approach
![Paper Status](https://img.shields.io/badge/paper-published-yellow)
![Code Size](https://img.shields.io/github/languages/code-size/ccasey/time-crystal-tensor-network)
![Repo Size](https://img.shields.io/github/repo-size/ccasey/time-crystal-tensor-network)
[![Parameters Documented](https://img.shields.io/badge/parameters-documented-blue)](#simulation-parameters)

Welcome to the **Discrete Time Crystals for Quantum Memories** repository! This project introduces a comprehensive tensor-network framework for modeling quantum memories as discrete time crystals (DTCs). Our implementation enables rigorous analysis of time-crystalline order, memory fidelity, and coherence protection in periodically driven many-body quantum systems.

## 🔬 Overview

Quantum memories are essential for quantum computing and communication networks, but their practical utility is limited by short coherence times. This work explores **discrete time crystals**—periodically driven many-body systems that exhibit stable subharmonic oscillations and break discrete time-translation symmetry—as a promising approach for passively protected quantum memories.

Our tensor-network framework provides:
- **Time-Evolving Block Decimation (TEBD)** for efficient many-body quantum simulation
- **Matrix Product State (MPS)** representation capturing large Hilbert spaces
- **Real and imaginary-time evolution** algorithms for ground state preparation and dynamics
- **Memory fidelity benchmarks** including Loschmidt echo calculations
- **Phase diagram mapping** across drive strength, interaction range, and disorder parameters
- **Spectral analysis tools** for identifying time-crystalline signatures

## 🚀 Key Features

### Tensor Network Methods
- **TEBD Algorithm**: Second-order Suzuki-Trotter decomposition for time evolution
- **TDVP Evolution**: Time-dependent variational principle for controlled bond dimension growth
- **MPS/MPO Framework**: Efficient representation of quantum many-body states and operators
- **Entanglement tracking**: Monitor entanglement entropy growth during evolution

### Physical Models
- **Floquet Kicked-Ising Model**: The paradigmatic DTC system with tunable parameters
- **Disorder modeling**: Random field and interaction disorder for realistic systems
- **Drive protocols**: Flexible periodic driving with arbitrary pulse shapes
- **Open system dynamics**: Lindblad master equation for decoherence effects

### Analysis Tools
- **Phase diagram generation**: Systematic parameter sweeps to map DTC stability regions
- **Spectral analysis**: Fourier transforms revealing subharmonic responses
- **Memory fidelity metrics**: Quantify information storage and retrieval performance


## 📁 Repository Structure

```
time-crystal-tensor-network/
├── src/
│   ├── core/                    # Core tensor network algorithms
│   │   ├── tensor_utils.py      # MPS/MPO operations and utilities
│   │   └── observables.py       # Measurement and correlation functions
│   ├── models/                  # Physical system implementations
│   │   └── kicked_ising.py      # Floquet kicked-Ising model
│   ├── dynamics/                # Time evolution algorithms
│   │   ├── tebd_evolution.py    # Time-evolving block decimation
│   │   ├── tdvp_evolution.py    # Time-dependent variational principle
│   │   └── open_system.py       # Open system Lindblad evolution
│   └── parameters.txt           # Default simulation parameters
├── examples/                    # Example scripts and demonstrations
│   └── loschmidt_echo_demo.py   # Memory fidelity demonstration
├── notebooks/                   # Jupyter notebooks with analysis
│   └── 02_paper_figures.ipynb   # Reproduce all paper figures
├── figures/                     # Generated plots and phase diagrams
├── tests/                       # Unit tests and validation
└── requirements.txt             # Python dependencies
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

# Run example DTC simulation
python examples/loschmidt_echo_demo.py
```

### Development Installation
```bash
# For development with additional tools
pip install -r requirements.txt
pip install -e .  # Editable installation
```

## 📊 Usage Examples

### Basic DTC Simulation
```python
from src.models.kicked_ising import KickedIsing
from src.dynamics.tebd_evolution import TEBDEvolution

# Initialize DTC model
model = KickedIsing(
    n_sites=12, 
    h_disorder=0.5,     # Disorder strength
    J=1.0,              # Interaction strength
    drive_amplitude=0.9  # Drive strength
)

# Set up TEBD evolution
evolution = TEBDEvolution(model, dt=0.1, max_bond_dim=64)

# Compute time-crystalline response
magnetization = evolution.evolve_and_measure(
    time_steps=100, 
    observable='magnetization'
)
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

### Memory Fidelity Analysis
```python
# Encode quantum information
initial_state = model.prepare_memory_state(information='quantum_data')

# Evolve with DTC protection
final_state = evolution.evolve_state(initial_state, time_steps=100)

# Compute memory fidelity
fidelity = evolution.compute_loschmidt_echo(initial_state, final_state)
print(f"Memory fidelity after 100 time steps: {fidelity:.4f}")
```

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

See [`src/parameters.txt`](src/parameters.txt) for complete parameter documentation and default values.

## 📈 Research Capabilities

Our framework enables comprehensive analysis of DTC quantum memories:

### Time-Crystalline Order
- **Subharmonic response detection** via spectral analysis
- **Order parameter calculations** for different DTC phases
- **Stability analysis** against perturbations and disorder

### Memory Performance
- **Loschmidt echo calculations** for information retrieval fidelity

- **Error rate analysis** for realistic noise models

### Phase Diagram Mapping
- **Systematic parameter sweeps** across experimentally relevant ranges
- **Critical point identification** using finite-size scaling
- **Thermal phase boundary** determination

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this framework in your research, please cite our work:

```bibtex
@inproceedings{casey2024dtc,
  title={Discrete Time Crystals for Quantum Memories: A Tensor-Network Approach},
  author={Casey, Connor},
  booktitle={IEEE International Conference on Quantum Computing and Engineering (QCE)},
  year={2025},
  status={in preparation}
}
```

## 📧 Contact

<sup>1</sup> College of Information and Computer Sciences, University of Massachusetts Amherst, USA  
<sup>2</sup> Department of Physics, University of Massachusetts Amherst, USA

- [**Connor Casey**](mailto:ccasey@umass.edu)<sup>1,2</sup>
- **Issues**: Please use the [GitHub Issues](https://github.com/ccasey/time-crystal-tensor-network/issues) page

---

**Ready to explore time crystals as quantum memories?** Start with our [example notebooks](notebooks/) or dive into the [tensor network algorithms](src/core/)!