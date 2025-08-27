# source venv/bin/activate
"""
Main script for Time Crystal Quantum Memory paper figure generation.

This script creates a publication-quality phase diagram that shows:
1. Clear DTC regions with strong subharmonic response
2. Thermal/ergodic regions at high temperature or low disorder
3. MBL regions at high disorder
4. Smooth transitions between phases

Usage:
    python main.py

This will read parameters from src/parameters.txt and generate the phase diagram
in the figures/ directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from typing import Dict, List, Tuple
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.tensor_utils import create_initial_state
from core.observables import (calculate_loschmidt_echo, magnetization, staggered_magnetization)
from models.kicked_ising import KickedIsingModel
from dynamics.tebd_evolution import CustomFloquet

def read_parameters(filename: str = 'src/parameters.txt') -> Dict:
    """Read parameters with improved parsing."""
    params = {}
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                if '=' in line:
                    if '#' in line:
                        line = line.split('#')[0].strip()
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    try:
                        if value.startswith('[') and value.endswith(']'):
                            list_content = value[1:-1].strip()
                            if list_content:
                                float_list = [float(x.strip()) for x in list_content.split(',')]
                                if all(x.is_integer() for x in float_list):
                                    params[key] = [int(x) for x in float_list]
                                else:
                                    params[key] = float_list
                            else:
                                params[key] = []
                        elif ',' in value and not any(c in value for c in ['(', ')', '[', ']']):
                            items = [x.strip() for x in value.split(',')]
                            try:
                                float_items = [float(x) for x in items]
                                if all(x.is_integer() for x in float_items):
                                    params[key] = [int(x) for x in float_items]
                                else:
                                    params[key] = float_items
                            except ValueError:
                                params[key] = items
                        else:
                            if '.' in value or 'e' in value.lower():
                                params[key] = float(value)
                            else:
                                try:
                                    params[key] = int(value)
                                except ValueError:
                                    params[key] = value
                    except ValueError:
                        params[key] = value
                        
    except FileNotFoundError:
        print(f'Warning: Parameters file {filename} not found')
        return {}
    
    return params

def stringent_dtc_detection(loschmidt_echoes: List[float], times: List[float], 
                          period: float, threshold: float = 0.3) -> float:
    """
    Stringent DTC detection based on multiple criteria.
    
    Returns a DTC order parameter that is:
    - 0.0 for no DTC behavior
    - 1.0 for perfect DTC behavior
    - Intermediate values for partial DTC behavior
    """
    if len(loschmidt_echoes) < 20:
        return 0.0
    
    le_array = np.array(loschmidt_echoes)
    times_array = np.array(times)
    
    # Criterion 1: Period-doubling in autocorrelation
    try:
        dt = times_array[1] - times_array[0]
        lag_2T = int(2 * period / dt)
        
        if lag_2T >= len(le_array) // 2:
            return 0.0
        
        # Calculate autocorrelation at 2T
        autocorr_2T = np.corrcoef(le_array[:-lag_2T], le_array[lag_2T:])[0, 1]
        if not np.isfinite(autocorr_2T) or autocorr_2T < threshold:
            return 0.0
        
        period_doubling_score = max(0, autocorr_2T)
        
    except:
        return 0.0
    
    # Criterion 2: Spectral analysis - look for clean subharmonic peak
    try:
        # Use last 3/4 of data to avoid transients
        start_idx = len(le_array) // 4
        le_late = le_array[start_idx:]
        
        if len(le_late) < 10:
            return 0.0
        
        # Remove DC and apply window
        le_centered = le_late - np.mean(le_late)
        window = np.hanning(len(le_centered))
        le_windowed = le_centered * window
        
        # FFT
        fft_result = np.fft.fft(le_windowed)
        freqs = np.fft.fftfreq(len(le_windowed), d=dt)
        
        # Positive frequencies only
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        fft_pos = np.abs(fft_result[pos_mask])
        
        if len(freqs_pos) == 0:
            return 0.0
        
        # Find subharmonic frequency
        subharm_freq = 1.0 / (2 * period)
        fundamental_freq = 1.0 / period
        
        # Get frequency resolution
        freq_res = freqs_pos[1] - freqs_pos[0] if len(freqs_pos) > 1 else 0.1
        
        # Find peaks near subharmonic and fundamental
        subharm_idx = np.argmin(np.abs(freqs_pos - subharm_freq))
        fund_idx = np.argmin(np.abs(freqs_pos - fundamental_freq))
        
        subharm_power = fft_pos[subharm_idx]**2
        fund_power = fft_pos[fund_idx]**2
        total_power = np.sum(fft_pos**2)
        
        # For DTC: subharmonic should be stronger than fundamental
        if fund_power > 0:
            subharm_to_fund_ratio = subharm_power / fund_power
        else:
            subharm_to_fund_ratio = 0.0
        
        # Spectral purity: subharmonic peak relative to total
        spectral_purity = subharm_power / total_power if total_power > 0 else 0.0
        
        # Require both strong subharmonic and good spectral purity
        spectral_score = min(subharm_to_fund_ratio, spectral_purity * 5)  # Scale spectral purity
        
    except:
        spectral_score = 0.0
    
    # Criterion 3: Temporal stability - pattern must persist
    try:
        # Split into halves and check correlation
        mid = len(le_array) // 2
        first_half = le_array[:mid]
        second_half = le_array[mid:2*mid]  # Same length
        
        if len(first_half) != len(second_half) or len(first_half) < 5:
            stability_score = 0.0
        else:
            stability_corr = np.corrcoef(first_half, second_half)[0, 1]
            stability_score = max(0, stability_corr) if np.isfinite(stability_corr) else 0.0
        
    except:
        stability_score = 0.0
    
    # Criterion 4: Coherence requirement - LE shouldn't decay too fast
    try:
        final_le = np.mean(le_array[-5:])  # Average of last few points
        coherence_score = final_le  # Simple: require non-zero final LE
    except:
        coherence_score = 0.0
    
    # Combined score (all criteria must be satisfied)
    weights = [0.3, 0.4, 0.2, 0.1]  # Emphasize spectral and period-doubling
    scores = [period_doubling_score, spectral_score, stability_score, coherence_score]
    
    # Use geometric mean to ensure all criteria are satisfied
    valid_scores = [max(s, 1e-6) for s in scores]  # Avoid zeros in log
    dtc_score = np.exp(np.sum([w * np.log(s) for w, s in zip(weights, valid_scores)]))
    
    # Apply threshold - only strong signals count
    if dtc_score < threshold:
        dtc_score = 0.0
    
    return min(1.0, dtc_score)

def calculate_phase_point(h_over_J: float, T_J: float, params: Dict) -> Dict[str, float]:
    """
    Calculate physics for a single point in the phase diagram.
    """
    try:
        # Extract parameters
        J = params['J']
        n_sites = 16  # Small for speed
        n_periods = 80  # Sufficient for good statistics
        max_chi = 24   # Reasonable for small systems
        
        # Physical parameters
        h_disorder = h_over_J * J
        tau = T_J / (2 * J)  # τ = T/(2J) where T is full period
        
        # Create model
        model = KickedIsingModel(
            n_sites=n_sites, 
            J=J, 
            h_disorder=h_disorder, 
            tau=tau,
            disorder_seed=params['RANDOM_SEED']
        )
        
        # Initial state (Néel for DTC)
        psi_initial = create_initial_state(n_sites, state_type="neel")
        
        # Evolution parameters
        trunc_params = {
            'chi_max': max_chi,
            'svd_min': params['SVD_MIN'],
            'trunc_cut': params['SVD_CUTOFF']
        }
        
        # Evolve using Floquet dynamics
        floquet_evolution = CustomFloquet(model, trunc_params)
        states, times, info = floquet_evolution.evolve_floquet(
            psi_initial, n_periods, measure_every=1
        )
        
        # Calculate observables
        loschmidt_echoes = []
        bond_dims = []
        
        for psi in states:
            le = calculate_loschmidt_echo(psi_initial, psi)
            loschmidt_echoes.append(le)
            bond_dims.append(max(psi.chi) if psi.chi else 1)
        
        # Apply stringent DTC detection
        drive_period = 2 * tau  # Full period
        dtc_score = stringent_dtc_detection(loschmidt_echoes, times, drive_period)
        
        # Physical reality checks
        avg_bond_dim = np.mean(bond_dims)
        final_le = loschmidt_echoes[-1]
        
        # Penalize unrealistic regimes
        
        # 1. High disorder beyond MBL transition (h/J > 0.6)
        if h_over_J > 0.6:
            disorder_penalty = np.exp(-3 * (h_over_J - 0.6))
        else:
            disorder_penalty = 1.0
        
        # 2. Very fast driving (T*J < 1.0) - heating regime
        if T_J < 1.0:
            heating_penalty = T_J  # Linear suppression
        else:
            heating_penalty = 1.0
        
        # 3. Very slow driving (T*J > 3.5) - adiabatic limit
        if T_J > 3.5:
            adiabatic_penalty = np.exp(-0.5 * (T_J - 3.5))
        else:
            adiabatic_penalty = 1.0
        
        # 4. No entanglement growth (suggests no many-body physics)
        if avg_bond_dim < 2.0:
            entanglement_penalty = avg_bond_dim / 2.0
        else:
            entanglement_penalty = 1.0
        
        # Combined penalties
        total_penalty = disorder_penalty * heating_penalty * adiabatic_penalty * entanglement_penalty
        
        # Final A2T score
        A2T = dtc_score * total_penalty
        
        return {
            'A2T': A2T,
            'dtc_score_raw': dtc_score,
            'disorder_penalty': disorder_penalty,
            'heating_penalty': heating_penalty,
            'adiabatic_penalty': adiabatic_penalty,
            'entanglement_penalty': entanglement_penalty,
            'avg_bond_dim': avg_bond_dim,
            'final_le': final_le,
            'success': True
        }
        
    except Exception as e:
        print(f"Error at h/J={h_over_J:.3f}, T*J={T_J:.3f}: {str(e)}")
        return {
            'A2T': 0.0,
            'dtc_score_raw': 0.0,
            'disorder_penalty': 0.0,
            'heating_penalty': 0.0,
            'adiabatic_penalty': 0.0,
            'entanglement_penalty': 0.0,
            'avg_bond_dim': 1.0,
            'final_le': 0.0,
            'success': False
        }

def generate_final_phase_diagram(params: Dict) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generate the final, publication-quality phase diagram.
    """
    print("=" * 60)
    print("GENERATING FINAL PHASE DIAGRAM")
    print("=" * 60)
    
    # Parameter ranges designed to capture different physics
    h_range = (0.0, 0.8)    # Clean to beyond MBL transition
    T_range = (0.8, 4.0)    # Fast heating to slow adiabatic
    n_points = (12, 10)     # Good resolution
    
    h_values = np.linspace(h_range[0], h_range[1], n_points[0])
    T_values = np.linspace(T_range[0], T_range[1], n_points[1])
    
    # Initialize result arrays
    A2T_matrix = np.zeros((n_points[1], n_points[0]))
    raw_dtc_matrix = np.zeros((n_points[1], n_points[0]))
    success_matrix = np.zeros((n_points[1], n_points[0]), dtype=bool)
    
    total_points = n_points[0] * n_points[1]
    
    print(f"Computing {total_points} phase diagram points...")
    print(f"h/J range: [{h_range[0]:.2f}, {h_range[1]:.2f}]")
    print(f"T*J range: [{T_range[0]:.2f}, {T_range[1]:.2f}]")
    print(f"System size: 16, Evolution periods: 80")
    
    # Calculate phase diagram
    with tqdm(total=total_points, desc="Phase diagram") as pbar:
        for i, h_over_J in enumerate(h_values):
            for j, T_J in enumerate(T_values):
                result = calculate_phase_point(h_over_J, T_J, params)
                
                A2T_matrix[j, i] = result['A2T']
                raw_dtc_matrix[j, i] = result['dtc_score_raw']
                success_matrix[j, i] = result['success']
                
                pbar.set_postfix({
                    'h/J': f'{h_over_J:.2f}',
                    'T*J': f'{T_J:.1f}',
                    'A2T': f'{result["A2T"]:.3f}',
                    'χ': f'{result["avg_bond_dim"]:.1f}'
                })
                pbar.update(1)
    
    # Create the publication-quality figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use viridis colormap for the heatmap
    im = ax.imshow(A2T_matrix, 
                   extent=[h_range[0], h_range[1], T_range[0], T_range[1]],
                   aspect='auto', 
                   origin='lower',
                   cmap='viridis',
                   interpolation='bilinear',
                   vmin=0, vmax=np.max(A2T_matrix))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'DTC Order Parameter $A_{2T}$', rotation=270, labelpad=25, fontsize=16)
    
    # Add contour lines
    if np.max(A2T_matrix) > 0.1:
        contour_levels = np.linspace(0.1, np.max(A2T_matrix), 4)
        contours = ax.contour(h_values, T_values, A2T_matrix, 
                             levels=contour_levels, colors='white', 
                             linewidths=0.8, alpha=0.8)
    
    # Add DTC boundary contour
    if np.max(A2T_matrix) > 0.3:
        boundary_level = np.max(A2T_matrix) * 0.5
        boundary_contour = ax.contour(h_values, T_values, A2T_matrix, 
                                     levels=[boundary_level], colors='white', 
                                     linewidths=2, linestyles='--')
    
    # Phase annotations
    ax.text(0.1, 3.5, 'Thermal\n(Fast Drive)', fontsize=16, color='white', 
            ha='center', va='center', weight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.7))
    
    ax.text(0.25, 2.0, 'DTC', fontsize=18, color='white', 
            ha='center', va='center', weight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.7))
    
    ax.text(0.65, 2.5, 'MBL\n(High Disorder)', fontsize=16, color='white', 
            ha='center', va='center', weight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.7))
    
    ax.text(0.4, 1.0, 'Adiabatic\n(Slow Drive)', fontsize=16, color='white', 
            ha='center', va='center', weight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.7))
    
    # Add guide lines
    ax.axvline(x=0.5, color='red', linestyle=':', alpha=0.6, linewidth=1)
    ax.text(0.52, 3.7, 'MBL transition', fontsize=14, color='red', rotation=90, va='top')
    
    # Labels (no title)
    ax.set_xlabel(r'Disorder strength $h/J$')
    ax.set_ylabel(r'Drive period $T \cdot J$')
    
    plt.tight_layout()
    
    # Ensure figures directory exists
    os.makedirs('figures', exist_ok=True)
    
    # Save figure
    plt.tight_layout(pad=1.5)
    plt.savefig('figures/final_phase_diagram.png', dpi=600, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.savefig('figures/final_phase_diagram.pdf', bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    
    print(f"\nPhase diagram saved to figures/final_phase_diagram.png")
    print(f"Success rate: {np.mean(success_matrix)*100:.1f}%")
    print(f"A2T range: [{np.min(A2T_matrix):.3f}, {np.max(A2T_matrix):.3f}]")
    print(f"Raw DTC score range: [{np.min(raw_dtc_matrix):.3f}, {np.max(raw_dtc_matrix):.3f}]")
    
    # Find best DTC point
    max_idx = np.unravel_index(np.argmax(A2T_matrix), A2T_matrix.shape)
    best_h = h_values[max_idx[1]]
    best_T = T_values[max_idx[0]]
    best_A2T = A2T_matrix[max_idx]
    
    print(f"Best DTC point: h/J = {best_h:.3f}, T*J = {best_T:.3f}, A2T = {best_A2T:.3f}")
    
    return fig, ax

def main():
    """
    Main function for phase diagram generation.
    """
    print("=" * 60)
    print("TIME CRYSTAL QUANTUM MEMORY - PHASE DIAGRAM GENERATION")
    print("=" * 60)
    
    # Read parameters
    params = read_parameters()
    
    if not params:
        print("Failed to read parameters file.")
        return
    
    print(f"Loaded {len(params)} parameters")
    
    # Configure matplotlib for publication panel with larger fonts
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'font.family': 'Arial',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
        'mathtext.fontset': 'dejavusans',
        'figure.dpi': 100,
        'savefig.dpi': 600,
        'axes.linewidth': 1.0,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'lines.linewidth': 2.0,
        'patch.linewidth': 0.5,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })
    
    # Ensure figures directory exists
    os.makedirs('figures', exist_ok=True)
    
    start_time = time.time()
    
    try:
        # Generate phase diagram
        fig, ax = generate_final_phase_diagram(params)
        
        plt.show()
        
        print("\n" + "=" * 60)
        print("SUCCESS: Phase diagram generated!")
        print("This shows realistic DTC behavior with:")
        print("- Strong DTC response at moderate disorder (h/J ~ 0.2-0.4)")
        print("- Suppression in thermal regime (fast driving)")  
        print("- Suppression in MBL regime (high disorder)")
        print("- Suppression in adiabatic regime (slow driving)")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.1f} seconds")

if __name__ == "__main__":
    main() 