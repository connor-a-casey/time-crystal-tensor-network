#!/usr/bin/env python3
"""
Physics validation tests for time crystal tensor network simulations.

These tests focus on validating the physics implementation and ensuring
that the simulations produce physically meaningful results that match
known theoretical expectations for discrete time crystals.
"""

import unittest
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.tensor_utils import create_initial_state
from core.observables import (calculate_loschmidt_echo, magnetization, 
                             staggered_magnetization, extract_subharmonic_amplitude)
from models.kicked_ising import KickedIsingModel
from dynamics.tebd_evolution import CustomFloquet


class TestDTCPhysics(unittest.TestCase):
    """Test discrete time crystal physics."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Standard DTC parameters
        self.dtc_model = KickedIsingModel(
            n_sites=8,
            J=1.0,
            h_disorder=0.25,  # Optimal DTC regime
            tau=1.0,          # T = 2*tau = 2
            disorder_seed=42
        )
        
        # Non-DTC model (too much disorder)
        self.mbl_model = KickedIsingModel(
            n_sites=8,
            J=1.0,
            h_disorder=1.0,   # Beyond MBL transition
            tau=1.0,
            disorder_seed=42
        )
        
        # Clean model (no disorder)
        self.clean_model = KickedIsingModel(
            n_sites=8,
            J=1.0,
            h_disorder=0.0,   # No disorder
            tau=1.0,
            disorder_seed=42
        )
        
    def test_period_doubling_signature(self):
        """Test for period-doubling signature in DTC regime."""
        psi_initial = create_initial_state(8, "neel")
        n_periods = 30
        
        # Evolve DTC system
        states, times = self.dtc_model.evolve(psi_initial, n_periods)
        
        # Calculate staggered magnetization
        stag_mags = [staggered_magnetization(psi) for psi in states]
        
        # Extract subharmonic amplitude
        times_array = np.array(times)
        stag_array = np.array(stag_mags)
        drive_period = 2 * self.dtc_model.tau
        
        subharm_amp = extract_subharmonic_amplitude(times_array, stag_array, drive_period)
        
        # DTC should show significant subharmonic response
        self.assertGreater(subharm_amp, 0.1, 
                          "DTC regime should show period-doubling signature")
        
    def test_mbl_regime_behavior(self):
        """Test behavior in MBL regime (high disorder)."""
        psi_initial = create_initial_state(8, "neel")
        n_periods = 20
        
        # Evolve MBL system
        states, times = self.mbl_model.evolve(psi_initial, n_periods)
        
        # Calculate observables
        loschmidt_echoes = [calculate_loschmidt_echo(psi_initial, psi) for psi in states]
        stag_mags = [staggered_magnetization(psi) for psi in states]
        
        # In strong MBL regime, should have good memory but reduced oscillations
        final_le = loschmidt_echoes[-1]
        self.assertGreater(final_le, 0.1, "MBL should preserve some memory")
        
        # Oscillations should be less coherent than optimal DTC
        stag_variation = np.std(stag_mags)
        self.assertGreater(stag_variation, 0.0, "Should have some dynamics")

        
    def test_initial_state_dependence(self):
        """Test DTC behavior with different initial states."""
        n_periods = 20
        initial_states = {
            'neel': create_initial_state(8, "neel"),
            'all_up': create_initial_state(8, "all_up"),
            'all_down': create_initial_state(8, "all_down")
        }
        
        subharm_amplitudes = {}
        
        for state_name, psi_initial in initial_states.items():
            states, times = self.dtc_model.evolve(psi_initial, n_periods)
            stag_mags = [staggered_magnetization(psi) for psi in states]
            
            times_array = np.array(times)
            stag_array = np.array(stag_mags)
            drive_period = 2 * self.dtc_model.tau
            
            subharm_amp = extract_subharmonic_amplitude(times_array, stag_array, drive_period)
            subharm_amplitudes[state_name] = subharm_amp
        
        # Neel state should be optimal for DTC
        self.assertGreater(subharm_amplitudes['neel'], 0.05,
                          "Neel state should show DTC signatures")
        
        # All other states might show weaker but non-zero signatures
        for state_name, amp in subharm_amplitudes.items():
            self.assertGreaterEqual(amp, 0.0, f"{state_name} should have non-negative amplitude")


class TestTensorNetworkProperties(unittest.TestCase):
    """Test tensor network specific properties."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = KickedIsingModel(
            n_sites=12,
            J=1.0,
            h_disorder=0.3,
            tau=1.0,
            disorder_seed=42
        )
        

class TestPhysicalConsistency(unittest.TestCase):
    """Test physical consistency and conservation laws."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = KickedIsingModel(
            n_sites=6,
            J=1.0,
            h_disorder=0.2,
            tau=0.8,
            disorder_seed=42
        )
        
    def test_norm_conservation(self):
        """Test that unitarity is preserved (norm conservation)."""
        psi_initial = create_initial_state(6, "neel")
        initial_norm = psi_initial.norm
        
        psi_current = psi_initial.copy()
        
        # Evolve for many steps
        for step in range(20):
            psi_current = self.model.floquet_step(psi_current)
            current_norm = psi_current.norm
            
            self.assertAlmostEqual(current_norm, initial_norm, places=8,
                                  msg=f"Norm not conserved at step {step}")
        
    def test_hermiticity_of_observables(self):
        """Test that observables give real expectation values."""
        psi = create_initial_state(6, "neel")
        
        # Evolve to get a non-trivial state
        for _ in range(5):
            psi = self.model.floquet_step(psi)
        
        # Test various observables
        mag_z = magnetization(psi, 'z')
        mag_x = magnetization(psi, 'x')
        mag_y = magnetization(psi, 'y')
        stag_mag = staggered_magnetization(psi)
        
        # All should be real (imaginary parts should be negligible)
        self.assertAlmostEqual(mag_z.imag if hasattr(mag_z, 'imag') else 0, 0.0, places=10)
        self.assertAlmostEqual(mag_x.imag if hasattr(mag_x, 'imag') else 0, 0.0, places=10)
        self.assertAlmostEqual(mag_y.imag if hasattr(mag_y, 'imag') else 0, 0.0, places=10)
        self.assertAlmostEqual(stag_mag.imag if hasattr(stag_mag, 'imag') else 0, 0.0, places=10)
        
    def test_physical_bounds(self):
        """Test that observables respect physical bounds."""
        psi = create_initial_state(6, "neel")
        
        # Evolve to get non-trivial state
        for _ in range(10):
            psi = self.model.floquet_step(psi)
        
        # Test magnetization bounds
        for direction in ['x', 'y', 'z']:
            total_mag = magnetization(psi, direction)
            # Total magnetization should be bounded by number of sites
            self.assertLessEqual(abs(total_mag), 6.1,  # Small tolerance for numerics
                               f"Magnetization {direction} exceeds physical bounds")
            
            # Single site magnetization should be bounded by 1
            for site in range(3):  # Test a few sites
                site_mag = magnetization(psi, direction, site=site)
                self.assertLessEqual(abs(site_mag), 1.1,
                                   f"Site magnetization exceeds bounds")
        
        # Loschmidt echo should be between 0 and 1
        psi_initial = create_initial_state(6, "neel")
        le = calculate_loschmidt_echo(psi_initial, psi)
        self.assertGreaterEqual(le, 0.0, "Loschmidt echo should be non-negative")
        self.assertLessEqual(le, 1.0, "Loschmidt echo should not exceed 1")


class TestParameterDependence(unittest.TestCase):
    """Test how physics changes with different parameters."""
    
    def test_disorder_strength_scaling(self):
        """Test how DTC signatures change with disorder strength."""
        n_periods = 20
        psi_initial = create_initial_state(8, "neel")
        
        disorder_strengths = [0.1, 0.3, 0.6]  # Weak, optimal, strong
        subharm_amps = []
        
        for h_disorder in disorder_strengths:
            model = KickedIsingModel(
                n_sites=8,
                J=1.0,
                h_disorder=h_disorder,
                tau=1.0,
                disorder_seed=42
            )
            
            states, times = model.evolve(psi_initial, n_periods)
            stag_mags = [staggered_magnetization(psi) for psi in states]
            
            times_array = np.array(times)
            stag_array = np.array(stag_mags)
            
            subharm_amp = extract_subharmonic_amplitude(times_array, stag_array, 2.0)
            subharm_amps.append(subharm_amp)
        
        # Expect optimal disorder around h/J ~ 0.3
        optimal_idx = 1  # 0.3 disorder
        self.assertGreater(subharm_amps[optimal_idx], 0.05,
                          "Optimal disorder should show DTC signatures")
        
        # All should be non-negative
        for i, amp in enumerate(subharm_amps):
            self.assertGreaterEqual(amp, 0.0,
                                   f"Amplitude should be non-negative for h={disorder_strengths[i]}")
            
    def test_drive_frequency_dependence(self):
        """Test dependence on drive frequency (tau parameter)."""
        n_periods = 15
        psi_initial = create_initial_state(8, "neel")
        
        tau_values = [0.5, 1.0, 2.0]  # Different drive frequencies
        final_overlaps = []
        
        for tau in tau_values:
            model = KickedIsingModel(
                n_sites=8,
                J=1.0,
                h_disorder=0.3,
                tau=tau,
                disorder_seed=42
            )
            
            states, times = model.evolve(psi_initial, n_periods)
            final_le = calculate_loschmidt_echo(psi_initial, states[-1])
            final_overlaps.append(final_le)
        
        # All should be valid
        for i, overlap in enumerate(final_overlaps):
            self.assertGreaterEqual(overlap, 0.0,
                                   f"Overlap should be non-negative for tau={tau_values[i]}")
            self.assertLessEqual(overlap, 1.0,
                                f"Overlap should not exceed 1 for tau={tau_values[i]}")
            
    def test_system_size_scaling(self):
        """Test how results scale with system size."""
        sizes = [6, 8, 10]
        n_periods = 15
        
        final_bond_dims = []
        final_overlaps = []
        
        for n_sites in sizes:
            model = KickedIsingModel(
                n_sites=n_sites,
                J=1.0,
                h_disorder=0.3,
                tau=1.0,
                disorder_seed=42
            )
            
            psi_initial = create_initial_state(n_sites, "neel")
            
            trunc_params = {'chi_max': 32, 'svd_min': 1e-12, 'trunc_cut': 1e-8}
            floquet_evo = CustomFloquet(model, trunc_params)
            
            states, times, info = floquet_evo.evolve_floquet(psi_initial, n_periods)
            
            final_bond_dim = info['final_bond_dim']
            final_le = calculate_loschmidt_echo(psi_initial, states[-1])
            
            final_bond_dims.append(final_bond_dim)
            final_overlaps.append(final_le)
        
        # Larger systems should generally have larger bond dimensions
        self.assertGreaterEqual(final_bond_dims[1], final_bond_dims[0],
                               "Bond dimension should not decrease with system size")
        
        # All overlaps should be physical
        for i, overlap in enumerate(final_overlaps):
            self.assertGreaterEqual(overlap, 0.0,
                                   f"Overlap should be non-negative for N={sizes[i]}")


if __name__ == '__main__':
    print("Running Physics Validation Tests")
    print("=" * 50)
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    unittest.main(verbosity=2) 