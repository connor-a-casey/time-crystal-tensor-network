"""
Observable calculations for quantum many-body systems using TeNPy.
"""

import numpy as np
from tenpy.networks.mps import MPS
from tenpy.linalg import np_conserved as npc
from typing import List, Tuple


def calculate_loschmidt_echo(psi_initial: MPS, psi_evolved: MPS) -> float:
    """
    Calculate the Loschmidt echo (return probability).
    
    The Loschmidt echo is defined as:
    L(t) = |⟨ψ₀|ψ(t)⟩|²
    
    Args:
        psi_initial: Initial MPS state |ψ₀⟩
        psi_evolved: Time-evolved MPS state |ψ(t)⟩
    
    Returns:
        Loschmidt echo L(t)
    """
    overlap = psi_initial.overlap(psi_evolved)
    return abs(overlap)**2


def magnetization(psi: MPS, direction: str = 'z', site: int = None) -> float:
    """
    Calculate magnetization of an MPS state.
    
    Args:
        psi: MPS state
        direction: Direction ('x', 'y', or 'z')
        site: Specific site (if None, calculates total)
    
    Returns:
        Magnetization value
    """
    pauli_dict = {
        'x': np.array([[0, 1], [1, 0]], dtype=complex),
        'y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'z': np.array([[1, 0], [0, -1]], dtype=complex)
    }
    
    op = pauli_dict[direction.lower()]
    
    # Get proper leg charges from the first site
    site_leg = psi.sites[0].leg
    leg_in = site_leg
    leg_out = site_leg.conj()
    
    op_tensor = npc.Array.from_ndarray(
        op, 
        labels=['p', 'p*'], 
        legcharges=[leg_in, leg_out]
    )
    
    if site is not None:
        # Single site magnetization
        result = psi.expectation_value(op_tensor, sites=[site])
        return float(result[0].real) if hasattr(result, '__len__') else float(result.real)
    else:
        # Total magnetization
        total_mag = 0.0
        for i in range(psi.L):
            mag_i = psi.expectation_value(op_tensor, sites=[i])
            mag_val = float(mag_i[0].real) if hasattr(mag_i, '__len__') else float(mag_i.real)
            total_mag += mag_val
        return total_mag


def correlation_function(psi: MPS, op1: str, op2: str, i: int, j: int) -> complex:
    """
    Calculate two-point correlation function ⟨σᵢ^op1 σⱼ^op2⟩.
    
    Args:
        psi: MPS state
        op1, op2: Pauli operators ('x', 'y', 'z')
        i, j: Site indices
    
    Returns:
        Correlation function value
    """
    pauli_dict = {
        'x': np.array([[0, 1], [1, 0]], dtype=complex),
        'y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'z': np.array([[1, 0], [0, -1]], dtype=complex),
        'i': np.eye(2, dtype=complex)
    }
    
    # Get proper leg charges from the first site
    site_leg = psi.sites[0].leg
    leg_in = site_leg
    leg_out = site_leg.conj()
    
    op1_tensor = npc.Array.from_ndarray(
        pauli_dict[op1.lower()], 
        labels=['p', 'p*'], 
        legcharges=[leg_in, leg_out]
    )
    op2_tensor = npc.Array.from_ndarray(
        pauli_dict[op2.lower()], 
        labels=['p', 'p*'], 
        legcharges=[leg_in, leg_out]
    )
    
    if i == j:
        # Same site correlation
        op_combined = pauli_dict[op1.lower()] @ pauli_dict[op2.lower()]
        op_combined_tensor = npc.Array.from_ndarray(
            op_combined, 
            labels=['p', 'p*'], 
            legcharges=[leg_in, leg_out]
        )
        result = psi.expectation_value(op_combined_tensor, sites=[i])
        return result[0] if hasattr(result, '__len__') else result
    else:
        # Different site correlation
        return psi.correlation_function(op1_tensor, op2_tensor, sites1=[i], sites2=[j])[0, 0]


def subharmonic_response(magnetization_data: List[float], drive_period: float) -> Tuple[float, float]:
    """
    Calculate subharmonic response from magnetization time series.
    
    Args:
        magnetization_data: Time series of magnetization
        drive_period: Period of the drive
    
    Returns:
        Tuple of (fundamental_amplitude, subharmonic_amplitude)
    """
    # Perform FFT
    fft_data = np.fft.fft(magnetization_data)
    freqs = np.fft.fftfreq(len(magnetization_data))
    
    # Find peaks at fundamental and subharmonic frequencies
    fundamental_freq = 1.0 / drive_period
    subharmonic_freq = fundamental_freq / 2.0
    
    # Find closest frequency bins
    fund_idx = np.argmin(np.abs(freqs - fundamental_freq))
    subharm_idx = np.argmin(np.abs(freqs - subharmonic_freq))
    
    fundamental_amplitude = abs(fft_data[fund_idx])
    subharmonic_amplitude = abs(fft_data[subharm_idx])
    
    return fundamental_amplitude, subharmonic_amplitude


def extract_subharmonic_amplitude(times: np.ndarray, magnetizations: np.ndarray, period: float) -> float:
    """
    Extract the sub-harmonic amplitude A2T from magnetization time series.
    
    This function analyzes the Fourier spectrum of the magnetization to identify
    period-doubling oscillations characteristic of discrete time crystals.
    
    Args:
        times: Array of time points
        magnetizations: Array of magnetization values
        period: Drive period T
    
    Returns:
        Sub-harmonic amplitude A2T (normalized)
    """
    if len(times) < 10 or len(magnetizations) < 10:
        return 0.0
    
    # Remove any NaN or infinite values
    valid_mask = np.isfinite(magnetizations) & np.isfinite(times)
    if np.sum(valid_mask) < 10:
        return 0.0
    
    times_clean = times[valid_mask]
    mags_clean = magnetizations[valid_mask]
    
    # Ensure uniform time spacing for FFT
    dt = np.mean(np.diff(times_clean))
    if dt <= 0:
        return 0.0
    
    # Detrend the data (remove DC component)
    mags_detrended = mags_clean - np.mean(mags_clean)
    
    # Apply window function to reduce spectral leakage
    window = np.hanning(len(mags_detrended))
    mags_windowed = mags_detrended * window
    
    # Compute FFT
    fft_result = np.fft.fft(mags_windowed)
    freqs = np.fft.fftfreq(len(mags_windowed), d=dt)
    
    # Only consider positive frequencies
    positive_freq_mask = freqs > 0
    freqs_pos = freqs[positive_freq_mask]
    fft_pos = fft_result[positive_freq_mask]
    
    # Target frequencies
    fundamental_freq = 1.0 / period  # ω = 2π/T
    subharmonic_freq = fundamental_freq / 2.0  # ω/2 = π/T
    
    # Find closest frequency bins
    if len(freqs_pos) == 0:
        return 0.0
    
    # Find subharmonic peak
    subharm_idx = np.argmin(np.abs(freqs_pos - subharmonic_freq))
    
    # Get amplitude at subharmonic frequency
    subharmonic_amplitude = np.abs(fft_pos[subharm_idx])
    
    # Normalize by the maximum amplitude in the spectrum for stability
    max_amplitude = np.max(np.abs(fft_pos))
    if max_amplitude > 1e-12:
        normalized_amplitude = subharmonic_amplitude / max_amplitude
    else:
        normalized_amplitude = 0.0
    
    return float(normalized_amplitude)


def calculate_magnetization(psi: MPS, direction: str = 'z') -> float:
    """
    Calculate total magnetization of an MPS state.
    
    Args:
        psi: MPS state
        direction: Direction ('x', 'y', or 'z')
    
    Returns:
        Total magnetization value
    """
    return magnetization(psi, direction)


def entanglement_spectrum(psi: MPS, cut: int) -> np.ndarray:
    """
    Calculate entanglement spectrum (Schmidt values) across a cut.
    
    Args:
        psi: MPS state
        cut: Cut position
    
    Returns:
        Array of Schmidt values
    """
    # Get the Schmidt values at the cut
    schmidt_values = psi.get_SL(cut)
    return schmidt_values


def fidelity_decay(loschmidt_echoes: List[float], times: List[float]) -> float:
    """
    Extract coherence time from Loschmidt echo decay.
    
    Fits to exponential decay: F(t) = F₀ * exp(-t/T₂)
    
    Args:
        loschmidt_echoes: List of Loschmidt echo values
        times: Corresponding time values
    
    Returns:
        Coherence time T₂
    """
    # Fit to exponential decay
    log_fidelity = np.log(np.maximum(loschmidt_echoes, 1e-10))
    
    # Linear fit to log(F) vs t
    coeffs = np.polyfit(times, log_fidelity, 1)
    decay_rate = -coeffs[0]
    
    # T₂ = 1/decay_rate
    coherence_time = 1.0 / decay_rate if decay_rate > 0 else np.inf
    
    return coherence_time


def order_parameter(psi: MPS, sublattice_a: List[int], sublattice_b: List[int]) -> float:
    """
    Calculate order parameter for discrete time crystal.
    
    Args:
        psi: MPS state
        sublattice_a: Sites in sublattice A
        sublattice_b: Sites in sublattice B
    
    Returns:
        Order parameter |⟨S_A⟩ - ⟨S_B⟩|
    """
    # Calculate average magnetization on each sublattice
    mag_a = np.mean([magnetization(psi, 'z', site) for site in sublattice_a])
    mag_b = np.mean([magnetization(psi, 'z', site) for site in sublattice_b])
    
    return abs(mag_a - mag_b)


def participation_ratio(psi: MPS) -> float:
    """
    Calculate participation ratio as a measure of localization.
    
    Args:
        psi: MPS state
    
    Returns:
        Participation ratio
    """
    # Calculate local density |⟨i|ψ⟩|² for each site
    local_amplitudes = []
    
    for i in range(psi.L):
        # Create projector onto site i
        proj_up = np.array([[1, 0], [0, 0]], dtype=complex)
        proj_down = np.array([[0, 0], [0, 1]], dtype=complex)
        
        # Get proper leg charges from the first site
        site_leg = psi.sites[0].leg
        leg_in = site_leg
        leg_out = site_leg.conj()
        
        proj_up_tensor = npc.Array.from_ndarray(
            proj_up, 
            labels=['p', 'p*'], 
            legcharges=[leg_in, leg_out]
        )
        proj_down_tensor = npc.Array.from_ndarray(
            proj_down, 
            labels=['p', 'p*'], 
            legcharges=[leg_in, leg_out]
        )
        
        prob_up_result = psi.expectation_value(proj_up_tensor, sites=[i])
        prob_down_result = psi.expectation_value(proj_down_tensor, sites=[i])
        
        prob_up = float(prob_up_result[0].real) if hasattr(prob_up_result, '__len__') else float(prob_up_result.real)
        prob_down = float(prob_down_result[0].real) if hasattr(prob_down_result, '__len__') else float(prob_down_result.real)
        
        local_amplitudes.append(prob_up + prob_down)
    
    local_amplitudes = np.array(local_amplitudes)
    
    # Participation ratio
    numerator = (np.sum(local_amplitudes))**2
    denominator = np.sum(local_amplitudes**2)
    
    return numerator / denominator if denominator > 0 else 0.0 