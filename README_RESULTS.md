# Time Crystal Physics Results: Individual Figure Analysis

## Figure Overview
**Discrete Time Crystal Behavior in Kicked-Ising Model via Tensor Network Simulations**

This collection presents four individual publication-ready figures demonstrating the characteristic signatures of discrete time crystals (DTCs). Each figure shows both time-domain dynamics (top panel) and frequency-domain analysis (bottom panel) obtained from tensor network evolution using the Time-Evolving Block Decimation (TEBD) algorithm.

## Individual Figure Descriptions

### Figure A: Perfect DTC - Clean Period-Doubling
**File:** `figure_a_perfect_dtc.png/.pdf`  
**Parameters:** *h/J* = 0.25, *T·J* = 2.0, *N* = 32, χ_max = 256

**Time Series (Top Panel):** Shows pristine period-doubling oscillations in the staggered magnetization *M_s(t)* (teal/cyan line) with period 2*T*, while total magnetization *M(t)* (purple/lavender dashed line) remains near zero due to spin cancellation in the Néel initial state.

**Fourier Spectrum (Bottom Panel):** Exhibits a sharp peak at ω/2 (red line), the hallmark signature of discrete time translation symmetry breaking. The narrow spectral width indicates long coherence times and stable DTC behavior.

### Figure B: Disordered DTC - Realistic Conditions  
**File:** `figure_b_disordered_dtc.png/.pdf`  
**Parameters:** *h/J* = 0.4, *T·J* = 2.0, *N* = 32, χ_max = 256

**Time Series (Top Panel):** Demonstrates robust DTC behavior under increased disorder. While oscillations show more noise compared to Figure A, the period-2*T* structure persists, evidencing the many-body protection characteristic of DTCs.

**Fourier Spectrum (Bottom Panel):** The sub-harmonic peak at ω/2 remains prominent but broadened, indicating shorter coherence times due to disorder-induced decoherence while maintaining the fundamental DTC signature.

### Figure C: DTC with Dephasing - Open-System Dynamics
**File:** `figure_c_dephasing_dtc.png/.pdf`  
**Parameters:** *h/J* = 0.3, *T·J* = 2.0, γ/*J* = 0.01, *N* = 32, χ_max = 256

**Time Series (Top Panel):** Shows exponential decay of DTC oscillations due to environmental dephasing at rate γ/*J* = 0.01. The decay envelope follows exp(-γ*t*) for magnetization amplitudes.

**Fourier Spectrum (Bottom Panel):** The sub-harmonic peak gradually diminishes in amplitude, demonstrating how open-system effects suppress the DTC response while preserving the frequency signature.

### Figure D: Multi-Site Dynamics - Individual Spin Trajectories
**File:** `figure_d_multisite_dtc.png/.pdf`  
**Parameters:** *h/J* = 0.3, *T·J* = 2.0, *N* = 16, χ_max = 256

**Time Series (Top Panel):** Tracks individual spin magnetizations ⟨σ_i^z⟩ for sites 1, 3, 5, 7, 9, 11, revealing how each spin participates in the collective DTC oscillation with site-dependent phases and amplitudes.

**Average Spectrum (Bottom Panel):** Computed from all tracked sites, showing the emergence of the sub-harmonic peak from the collective many-body dynamics, confirming that DTC behavior is a genuine many-body phenomenon rather than single-particle physics.

## Visual Design

### Color Scheme
- **Staggered Magnetization:** Teal/Cyan (#48D5DC)
- **Total Magnetization:** Purple/Lavender (#9D8FBF) 
- **Fourier Spectrum:** Pink (#FF6B9D)
- **DTC Peak (ω/2):** Red (#FF4757)
- **Drive Frequency (ω):** Light Salmon (#FFA07A)
- **Individual Sites:** Diverse palette for multi-site visualization

### Format Specifications
- **Layout:** Each figure is a 2-panel vertical arrangement (8×8 inches)
- **Resolution:** 300 DPI for publication quality
- **File Types:** Both PNG (raster) and PDF (vector) formats
- **Typography:** Professional serif fonts with LaTeX mathematical notation
- **Styling:** Clean grids, consistent axis labeling, frameless legends

## Key Physics Insights

### 1. Sub-harmonic Response
All figures show spectral peaks at ω/2 (marked by red vertical lines), confirming discrete time translation symmetry breaking. This is the defining signature of discrete time crystals.

### 2. Many-Body Protection
The DTC signature survives disorder (Figure B) and even persists under dephasing (Figure C), demonstrating the many-body protection that makes DTCs robust quantum phases of matter.

### 3. Collective Behavior
Figure D reveals that individual spins participate in the collective DTC oscillation, with the sub-harmonic peak emerging from the average spectrum of all sites.

### 4. Experimental Relevance
The progression from perfect → disordered → open-system conditions shows how DTCs behave under increasingly realistic experimental conditions.

## Computational Details

### Model Parameters
- **Hamiltonian:** Floquet kicked-Ising model with nearest-neighbor interactions
- **Drive Protocol:** Stroboscopic π-pulses with period *T*
- **Disorder:** Quenched random longitudinal fields *h_i* ∈ [-*h*, *h*]
- **Initial State:** Néel state |↑↓↑↓...⟩ 
- **Evolution:** 200 Floquet periods per scenario

### Tensor Network Parameters
- **Algorithm:** Time-Evolving Block Decimation (TEBD)
- **Bond Dimension:** χ_max = 256
- **SVD Cutoff:** ε = 10^-7
- **Truncation:** SVD minimum = 10^-12

### Observable Definitions
- **Staggered Magnetization:** *M_s* = (1/*N*) Σ_i (-1)^i ⟨σ_i^z⟩
- **Total Magnetization:** *M* = (1/*N*) Σ_i ⟨σ_i^z⟩


### Spectral Analysis
- **Method:** Fast Fourier Transform (FFT) with Hanning window
- **Frequency Normalization:** ω/ω_drive where ω_drive = 2π/*T*
- **Power Normalization:** Peak amplitude normalization

## Usage Instructions

### To Reproduce Individual Figures:
```bash
# Run the simulation to generate all individual figures
python main2.py

# Generated files in figures/ directory:
# - figure_a_perfect_dtc.png/.pdf
# - figure_b_disordered_dtc.png/.pdf  
# - figure_c_dephasing_dtc.png/.pdf
# - figure_d_multisite_dtc.png/.pdf
```

### Parameter Modification:
Edit `src/parameters.txt` to adjust:
- System sizes (*N_SITES_*)
- Disorder strengths (*H_MAX*)
- Drive periods (*T_DRIVE*)
- Evolution times (*N_PERIODS_*)
- Tensor network parameters (*CHI_MAX*, *SVD_CUTOFF*)

### Custom Combinations:
Each figure is saved separately, allowing flexible combination:
- Journal 2×2 grid layout
- Single column arrangement 
- Presentation slide format
- Poster design integration

## Experimental Connections

### Platform Agnostic
This analysis applies to any experimental platform capable of implementing the kicked-Ising model:
- **Quantum Simulators:** Rydberg atom arrays, superconducting qubits
- **Solid-State Systems:** NV centers, quantum dots  
- **Trapped Systems:** Atomic and molecular systems in optical lattices
- **Condensed Matter:** Magnetic insulators, anyonic systems

### Measurable Signatures
- **Staggered Magnetization:** Accessible via site-resolved measurements
- **Spectral Peaks:** Observable in driven spectroscopy experiments

## Scientific Significance

### 1. Quantum Memory Applications
DTCs offer passive error protection through many-body localization, making them promising candidates for robust quantum memories in quantum networks.

### 2. Many-Body Localization
The results demonstrate the interplay between driving, disorder, and many-body interactions in creating stable non-equilibrium phases.

### 3. Spectroscopic Signatures
The clear sub-harmonic peaks provide unambiguous experimental signatures for detecting DTC behavior in laboratory systems.

### 4. Open-System Dynamics
Figure C shows how environmental coupling affects DTC coherence, crucial for understanding decoherence in realistic quantum devices.

## Figure Caption Template

**For Journal Use:**  
*Discrete time crystal signatures in kicked-Ising model simulations. (Top panels) Time evolution of staggered magnetization M_s(t) (teal) and total magnetization M(t) (purple dashed) over 100 Floquet periods. (Bottom panels) Fourier power spectra showing sub-harmonic peaks at ω/2 (red vertical line) characteristic of period-doubling. System parameters: N = 32 spins, χ_max = 256, T·J = 2. Figure A: Perfect DTC (h/J = 0.25). Figure B: Disordered DTC (h/J = 0.4). Figure C: DTC with dephasing (γ/J = 0.01). Figure D: Multi-site dynamics (N = 16, individual sites shown).*

---

**Note:** All results are generated from first-principles tensor network calculations with no phenomenological parameters or synthetic data. The individual figure format allows flexible arrangement for different publication layouts while maintaining consistent professional styling and scientific rigor. 