# Time Crystals for Quantum Memories: A Tensor-Network Approach

This repository contains the implementation of a tensor-network framework for modeling quantum memories as discrete time crystals, as described in the paper "Time Crystals for Quantum Memories: A Tensor-Network Approach".

## Overview

The project implements:
- **Discrete Time Crystal (DTC) simulation** using the Floquet kicked-Ising model
- **Tensor network methods** including Matrix Product States (MPS) and Matrix Product Operators (MPO)
- **Time-Evolving Block Decimation (TEBD)** with 2nd-order Suzuki-Trotter decomposition
- **Open system dynamics** via Lindblad master equation
- **Memory fidelity benchmarks** including Loschmidt echo calculations
- **Phase diagram analysis** and coherence enhancement studies

## Repository Structure

```
time-crystal-tensor-network/
├── src/
│   ├── core/                 # Core tensor network implementations
│   ├── models/               # Physical models (kicked-Ising, etc.)
│   ├── dynamics/             # Time evolution algorithms
│   ├── analysis/             # Analysis tools and metrics
│   └── benchmarks/           # Performance benchmarking
├── notebooks/                # Jupyter notebooks for examples
├── tests/                    # Unit tests
├── data/                     # Generated data and results
├── figures/                  # Generated plots and figures
└── requirements.txt          # Python dependencies
```

## Key Features

- **Floquet Kicked-Ising Model**: Implementation of the driven many-body system that hosts discrete time crystals
- **TEBD Evolution**: Second-order Suzuki-Trotter time evolution with bond dimension control
- **Open System Dynamics**: Lindblad master equation evolution for realistic decoherence
- **Memory Fidelity**: Loschmidt echo and other quantum memory performance metrics
- **Phase Diagrams**: Tools for mapping DTC stability regions

## Installation

```bash
pip install -r requirements.txt
```

## Usage

See the notebooks in `notebooks/` for detailed examples and tutorials.

## Citation

If you use this code in your research, please cite:

```
@article{casey2024timecrystals,
  title={Time Crystals for Quantum Memories: A Tensor-Network Approach},
  author={Casey, Connor},
  journal={IEEE Conference},
  year={2024}
}
``` 