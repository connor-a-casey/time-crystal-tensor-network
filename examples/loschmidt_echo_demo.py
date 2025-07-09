#!/usr/bin/env python3
"""
Demonstration of Loschmidt Echo calculation for Time Crystal quantum memories.

This script implements the key calculation from the paper:
L(t) = |⟨ψ₀|ψ(t)⟩|²

The Loschmidt echo quantifies the return probability and is used to 
measure memory fidelity in the time crystal system.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.tensor_utils import create_initial_state
from core.observables import calculate_loschmidt_echo, magnetization, fidelity_decay
from models.kicked_ising import KickedIsingModel
from dynamics.tebd_evolution import CustomFloquet


def main():
    print("=" * 60)
    print("Time Crystal Quantum Memory - Loschmidt Echo Demonstration")
    print("=" * 60)
    
    # System parameters from the paper
    n_sites = 16
    J = 1.0
    h_disorder = 0.3 * J  # h/J = 0.3 from paper
    tau = 1.0
    gamma = 1e-3 * J  # Dephasing rate
    
    print(f"System Parameters:")
    print(f"  Sites: {n_sites}")
    print(f"  J: {J}")
    print(f"  h/J: {h_disorder/J}")
    print(f"  τ: {tau}")
    print(f"  γ/J: {gamma/J}")
    
    # Create the kicked-Ising model
    model = KickedIsingModel(
        n_sites=n_sites,
        J=J,
        h_disorder=h_disorder,
        tau=tau,
        disorder_seed=42
    )
    
    # Create initial state (Néel state for DTC)
    psi_initial = create_initial_state(n_sites, state_type="neel")
    print(f"\nInitial state: {psi_initial}")
    print(f"Initial magnetization: {magnetization(psi_initial, 'z'):.4f}")
    
    # Evolution parameters
    n_periods = 200
    trunc_params = {
        'chi_max': 64,
        'svd_min': 1e-12,
        'trunc_cut': 1e-10
    }
    
    # Create evolution engine
    floquet_evolution = CustomFloquet(model, trunc_params)
    
    print(f"\nStarting time evolution for {n_periods} periods...")
    
    # Evolve the system
    states, times, info = floquet_evolution.evolve_floquet(
        psi_initial, n_periods, measure_every=1
    )
    
    print(f"Evolution completed in {info['wall_time']:.2f} seconds")
    print(f"Rate: {info['periods_per_second']:.1f} periods/second")
    print(f"Final bond dimension: {info['final_bond_dim']}")
    
    # Calculate Loschmidt echo for each time point
    print("\nCalculating Loschmidt echo L(t) = |⟨ψ₀|ψ(t)⟩|²...")
    
    loschmidt_echoes = []
    magnetizations = []
    
    for i, psi in enumerate(states):
        # Calculate Loschmidt echo
        le = calculate_loschmidt_echo(psi_initial, psi)
        loschmidt_echoes.append(le)
        
        # Calculate magnetization
        mag = magnetization(psi, 'z')
        magnetizations.append(mag)
        
        if i % 50 == 0:
            print(f"  t = {times[i]:.2f}, L(t) = {le:.6f}, M = {mag:.4f}")
    
    # Convert to numpy arrays
    loschmidt_echoes = np.array(loschmidt_echoes)
    magnetizations = np.array(magnetizations)
    times = np.array(times)
    
    # Extract coherence time
    coherence_time = fidelity_decay(loschmidt_echoes.tolist(), times.tolist())
    print(f"\nCoherence time T₂ = {coherence_time:.2f} J⁻¹")
    
    # Compare with paper's result
    paper_coherence_time = 3e4  # From paper: T₂^(DTC) ≈ 3×10⁴ J⁻¹
    print(f"Paper reports: T₂^(DTC) ≈ {paper_coherence_time:.0e} J⁻¹")
    
    # Plot results
    print("\nGenerating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Loschmidt Echo vs Time
    axes[0, 0].semilogy(times, loschmidt_echoes, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time (J⁻¹)')
    axes[0, 0].set_ylabel('Loschmidt Echo L(t)')
    axes[0, 0].set_title('Memory Fidelity: L(t) = |⟨ψ₀|ψ(t)⟩|²')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(1e-6, 1)
    
    # Add exponential decay fit
    if coherence_time < np.inf:
        t_fit = np.linspace(0, times[-1], 100)
        decay_fit = np.exp(-t_fit / coherence_time)
        axes[0, 0].plot(t_fit, decay_fit, 'r--', alpha=0.7, 
                       label=f'T₂ = {coherence_time:.1f} J⁻¹')
        axes[0, 0].legend()
    
    # Plot 2: Magnetization Oscillations
    axes[0, 1].plot(times, magnetizations, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Time (J⁻¹)')
    axes[0, 1].set_ylabel('Magnetization ⟨Σᵢ σᵢᶻ⟩')
    axes[0, 1].set_title('Subharmonic Oscillations')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Frequency Spectrum
    fft_mag = np.fft.fft(magnetizations)
    freqs = np.fft.fftfreq(len(magnetizations), d=2*tau)
    positive_freqs = freqs[:len(freqs)//2]
    positive_fft = np.abs(fft_mag[:len(freqs)//2])
    
    axes[1, 0].plot(positive_freqs, positive_fft, 'g-', linewidth=2)
    axes[1, 0].axvline(x=1.0/(2*tau), color='red', linestyle='--', 
                      label='Fundamental')
    axes[1, 0].axvline(x=0.5/(2*tau), color='orange', linestyle='--', 
                      label='Subharmonic')
    axes[1, 0].set_xlabel('Frequency (J)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].set_title('Frequency Spectrum')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, 1.0/(2*tau))
    
    # Plot 4: Bond dimension growth
    bond_dims = info['bond_dimensions']
    axes[1, 1].plot(times, bond_dims, 'purple', linewidth=2)
    axes[1, 1].set_xlabel('Time (J⁻¹)')
    axes[1, 1].set_ylabel('Max Bond Dimension')
    axes[1, 1].set_title('Entanglement Growth')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('loschmidt_echo_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY RESULTS")
    print("=" * 60)
    print(f"Final Loschmidt echo: {loschmidt_echoes[-1]:.6f}")
    print(f"Memory coherence time: {coherence_time:.2f} J⁻¹")
    print(f"Average magnetization: {np.mean(magnetizations):.4f}")
    print(f"Magnetization oscillation amplitude: {np.std(magnetizations):.4f}")
    
    # Check for DTC behavior
    subharmonic_power = positive_fft[np.argmin(np.abs(positive_freqs - 0.5/(2*tau)))]
    fundamental_power = positive_fft[np.argmin(np.abs(positive_freqs - 1.0/(2*tau)))]
    
    print(f"Subharmonic/Fundamental ratio: {subharmonic_power/fundamental_power:.3f}")
    
    if subharmonic_power/fundamental_power > 0.1:
        print("✓ Strong discrete time crystal behavior detected!")
    else:
        print("✗ Weak time crystal signature")
    
    print("=" * 60)
    
    return {
        'loschmidt_echoes': loschmidt_echoes,
        'times': times,
        'coherence_time': coherence_time,
        'magnetizations': magnetizations
    }


if __name__ == "__main__":
    results = main() 