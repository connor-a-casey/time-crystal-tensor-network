# Time Crystal Paper Figure Generation

This directory contains scripts to generate the three main figures for the paper "Time Crystals for Quantum Memories: A Tensor-Network Approach".

## Generated Figures

### 1. Phase Diagram and Coherence Enhancement (`phase_diagram.png`)
- **Type**: Heat-map (rectangular) of the normalized sub-harmonic amplitude A₂T
- **X-axis**: Disorder strength h/J
- **Y-axis**: Drive period T·J
- **Colorbar**: A₂T (dimensionless)
- **Features**: 
  - Viridis colormap (color-blind safe)
  - Dashed white contour marking DTC lobe boundary
  - MBL transition line annotation

### 2. Memory-Fidelity Benchmarks (`memory_fidelity_benchmarks.png`)
- **Type**: Line charts showing Loschmidt echo and magnetization oscillations
- **Left panel**: Loschmidt echo L(t) = |⟨ψ₀|ψ(t)⟩|² vs time
- **Right panel**: Sub-harmonic magnetization oscillations
- **Multiple curves**: Different dephasing rates γ/J = [0, 10⁻⁴, 10⁻³, 10⁻²]

### 3. TEBD vs TDVP Performance (`tebd_vs_tdvp_performance.png`)
- **Type**: Line charts comparing TEBD and TDVP algorithms
- **Left panel**: Wall-clock time per 200 periods vs system size N
- **Right panel**: Peak bond dimension χₘₐₓ vs system size N
- **Two lines**: Blue = TEBD, Red = TDVP
- **Individual markers**: Show sampling density

## Usage

### Quick Test
First, run the test script to verify everything works:
```bash
python test_figure_generation.py
```

This will:
- Test basic tensor network functionality
- Generate a test phase diagram point
- Create a small test visualization
- Verify that all components work correctly

### Full Figure Generation
Once tests pass, generate all figures:
```bash
python generate_paper_figures.py
```

This will create high-resolution figures in the `figures/` directory in both PNG and PDF formats.

## System Requirements

- Python 3.8+
- Required packages (install with `pip install -r requirements.txt`):
  - numpy >= 1.24.0
  - scipy >= 1.10.0
  - matplotlib >= 3.7.0
  - seaborn >= 0.12.0
  - physics-tenpy >= 0.10.0
  - tqdm >= 4.65.0

## Computational Requirements

### Memory Requirements
- **Phase Diagram**: ~2-4 GB RAM (depends on resolution)
- **Memory Fidelity**: ~1-2 GB RAM
- **Performance Benchmarks**: ~1-2 GB RAM

### Runtime Estimates
- **Phase Diagram**: ~30-60 minutes (500 points at 32 sites, 150 periods each)
- **Memory Fidelity**: ~5-10 minutes (32 sites, 300 periods, 4 dephasing rates)
- **Performance Benchmarks**: ~10-20 minutes (6 system sizes, 200 periods each)

**Total estimated runtime**: 45-90 minutes on a modern laptop

## Parameters and Customization

### Phase Diagram Parameters
```python
generate_phase_diagram(
    h_range=(0.1, 0.8),      # Disorder strength range
    T_range=(0.5, 3.0),      # Drive period range
    n_points=(25, 20),       # Grid resolution
    n_sites=32,              # System size
    n_periods=150            # Evolution time
)
```

### Memory Fidelity Parameters
```python
generate_memory_fidelity_benchmarks(
    n_sites=32,              # System size
    n_periods=300,           # Evolution time
    gamma_values=[0.0, 1e-4, 1e-3, 1e-2]  # Dephasing rates
)
```

### Performance Benchmark Parameters
```python
generate_tebd_vs_tdvp_performance(
    system_sizes=[12, 16, 20, 24, 28, 32],  # System sizes to test
    n_periods=200            # Evolution time per benchmark
)
```

## Physical Parameters

The scripts use the following physical parameters consistent with the paper:

- **Ising coupling**: J = 1.0 (sets energy scale)
- **Disorder strength**: h/J = 0.3 (middle of DTC phase)
- **Drive period**: T·J = 2.0 (middle of DTC phase)
- **Bond dimension**: χₘₐₓ = 64-256 (depending on system size)
- **Truncation**: SVD cutoff = 10⁻¹⁰, SVD min = 10⁻¹²

## Output Files

All figures are saved in the `figures/` directory:

```
figures/
├── phase_diagram.png
├── phase_diagram.pdf
├── memory_fidelity_benchmarks.png
├── memory_fidelity_benchmarks.pdf
├── tebd_vs_tdvp_performance.png
└── tebd_vs_tdvp_performance.pdf
```

## Performance Optimization

### For faster generation:
1. **Reduce resolution**: Use fewer points in phase diagram
2. **Smaller systems**: Use fewer sites for testing
3. **Shorter evolution**: Use fewer periods
4. **Parallel processing**: Can be added for phase diagram calculation

### For higher quality:
1. **Higher resolution**: More points in phase diagram
2. **Larger systems**: More sites for better finite-size scaling
3. **Longer evolution**: More periods for better time-crystal signatures
4. **Higher bond dimension**: Larger χₘₐₓ for better accuracy

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all required packages are installed
2. **Memory errors**: Reduce system size or bond dimension
3. **Slow performance**: Reduce resolution or use smaller systems
4. **TeNPy errors**: Check that physics-tenpy is properly installed

### Error Messages

- **"Model failed to converge"**: Reduce truncation parameters
- **"Bond dimension exceeded"**: Increase χₘₐₓ or reduce system size
- **"TDVP evolution failed"**: Switch to TEBD-only for problematic points

## Technical Details

### Tensor Network Methods
- **MPS representation**: Matrix Product States for pure quantum states
- **TEBD evolution**: Time-Evolving Block Decimation with 2nd-order Suzuki-Trotter
- **TDVP evolution**: Time-Dependent Variational Principle (two-site variant)
- **Bond truncation**: SVD-based truncation with controlled error

### Physical Model
- **Kicked-Ising Hamiltonian**: Three-step Floquet evolution
- **Disorder**: Quenched random longitudinal fields
- **Initial state**: Néel state |↑↓↑↓...⟩
- **Observables**: Loschmidt echo, magnetization, subharmonic response

### Numerical Accuracy
- **Time step**: Optimized for each method
- **Truncation error**: Controlled to ~10⁻¹⁰
- **Finite-size effects**: Tested up to 32 sites
- **Statistical averaging**: Fixed disorder seed for reproducibility

## Citation

If you use these scripts in your research, please cite:

```bibtex
@article{casey2024timecrystals,
  title={Time Crystals for Quantum Memories: A Tensor-Network Approach},
  author={Casey, Connor},
  journal={IEEE Conference},
  year={2024}
}
``` 