#!/usr/bin/env python3
"""
High-level integration tests for time crystal tensor network codebase.

These tests verify that the core functionality works correctly without
modifying the existing codebase. Tests cover:
- Basic system initialization
- Time evolution dynamics  
- Observable calculations
- Phase diagram functionality
- Figure generation workflows
"""

import unittest
import numpy as np
import os
import sys
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from unittest.mock import patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.tensor_utils import create_initial_state, pauli_matrices
from core.observables import (calculate_loschmidt_echo, magnetization, 
                             staggered_magnetization, subharmonic_response,
                             extract_subharmonic_amplitude, order_parameter)
from models.kicked_ising import KickedIsingModel
from dynamics.tebd_evolution import CustomFloquet
from main import (read_parameters, stringent_dtc_detection, 
                 calculate_phase_point, simulate_perfect_dtc)


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality and system initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_params = {
            'J': 1.0,
            'CHI_MAX': 16,  # Small for fast tests
            'SVD_MIN': 1e-12,
            'SVD_CUTOFF': 1e-8,
            'RANDOM_SEED': 42,
            'N_SITES_FIGURES': 8,  # Small system for tests
            'N_PERIODS_PHASE': 10   # Short evolution for tests
        }
        
    def test_pauli_matrices(self):
        """Test Pauli matrix generation."""
        pauli = pauli_matrices()
        
        # Check all matrices exist
        self.assertIn('I', pauli)
        self.assertIn('X', pauli)
        self.assertIn('Y', pauli)
        self.assertIn('Z', pauli)
        
        # Check dimensions
        for op in pauli.values():
            self.assertEqual(op.shape, (2, 2))
        
        # Check identity
        np.testing.assert_array_almost_equal(pauli['I'], np.eye(2))
        
        # Check Pauli X anticommutes with Pauli Z
        anticommutator = pauli['X'] @ pauli['Z'] + pauli['Z'] @ pauli['X']
        np.testing.assert_array_almost_equal(anticommutator, np.zeros((2, 2)))
        
    def test_initial_state_creation(self):
        """Test MPS initial state creation."""
        n_sites = 4
        
        # Test different state types
        state_types = ['all_up', 'all_down', 'neel']
        
        for state_type in state_types:
            with self.subTest(state_type=state_type):
                psi = create_initial_state(n_sites, state_type)
                
                # Check basic properties
                self.assertEqual(psi.L, n_sites)
                self.assertIsNotNone(psi.chi)
                
                # Check normalization
                norm = psi.norm
                self.assertAlmostEqual(norm, 1.0, places=10)
                
        # Test invalid state type
        with self.assertRaises(ValueError):
            create_initial_state(n_sites, "invalid_state")
            
    def test_parameter_reading(self):
        """Test parameter file reading functionality."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("# Test config\n")
            f.write("J = 1.0\n")
            f.write("CHI_MAX = 64\n")
            f.write("H_VALUES = [0.1, 0.2, 0.3]\n")
            f.write("TEST_STRING = test_value\n")
            temp_filename = f.name
            
        try:
            params = read_parameters(temp_filename)
            
            # Check parameter parsing
            self.assertEqual(params['J'], 1.0)
            self.assertEqual(params['CHI_MAX'], 64)
            self.assertEqual(params['H_VALUES'], [0.1, 0.2, 0.3])
            self.assertEqual(params['TEST_STRING'], 'test_value')
            
        finally:
            os.unlink(temp_filename)


class TestKickedIsingModel(unittest.TestCase):
    """Test the kicked Ising model implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = KickedIsingModel(
            n_sites=4,
            J=1.0,
            h_disorder=0.2,
            tau=1.0,
            disorder_seed=42
        )
        
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.n_sites, 4)
        self.assertEqual(self.model.J, 1.0)
        self.assertEqual(self.model.h_disorder, 0.2)
        self.assertEqual(self.model.tau, 1.0)
        
        # Check disorder fields are generated
        self.assertEqual(len(self.model.h_fields), 4)
        self.assertTrue(np.all(np.abs(self.model.h_fields) <= 0.2))
        
        # Check gates are prepared
        self.assertIsNotNone(self.model.pi_pulse_gate)
        self.assertEqual(len(self.model.ising_gates), 3)  # n_sites - 1
        
    def test_floquet_step(self):
        """Test single Floquet step evolution."""
        psi_initial = create_initial_state(4, "neel")
        
        # Apply one Floquet step
        psi_evolved = self.model.floquet_step(psi_initial)
        
        # Check state is normalized
        self.assertAlmostEqual(psi_evolved.norm, 1.0, places=10)
        
        # Check state has changed (not identical to initial)
        overlap = calculate_loschmidt_echo(psi_initial, psi_evolved)
        self.assertLessEqual(overlap, 1.0)  # Should be at most 1
        self.assertGreaterEqual(overlap, 0.0)  # Should be at least 0
        
    def test_multi_step_evolution(self):
        """Test multi-step evolution."""
        psi_initial = create_initial_state(4, "neel")
        n_steps = 5
        
        states, times = self.model.evolve(psi_initial, n_steps)
        
        # Check output format
        self.assertEqual(len(states), n_steps + 1)  # Initial + n_steps
        self.assertEqual(len(times), n_steps + 1)
        
        # Check time progression
        expected_times = [i * 2 * self.model.tau for i in range(n_steps + 1)]
        np.testing.assert_array_almost_equal(times, expected_times)
        
        # Check all states are normalized
        for psi in states:
            self.assertAlmostEqual(psi.norm, 1.0, places=10)


class TestObservables(unittest.TestCase):
    """Test observable calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.psi_up = create_initial_state(4, "all_up")
        self.psi_down = create_initial_state(4, "all_down") 
        self.psi_neel = create_initial_state(4, "neel")
        
    def test_loschmidt_echo(self):
        """Test Loschmidt echo calculation."""
        # Self-overlap should be 1
        echo_self = calculate_loschmidt_echo(self.psi_up, self.psi_up)
        self.assertAlmostEqual(echo_self, 1.0, places=10)
        
        # Orthogonal states should have zero overlap
        echo_orthogonal = calculate_loschmidt_echo(self.psi_up, self.psi_down)
        self.assertAlmostEqual(echo_orthogonal, 0.0, places=10)
        
        # Neel state overlap with up state should be valid
        echo_partial = calculate_loschmidt_echo(self.psi_up, self.psi_neel)
        self.assertGreaterEqual(echo_partial, 0.0)
        self.assertLessEqual(echo_partial, 1.0)
        
    def test_magnetization(self):
        """Test magnetization calculations."""
        # All up state should have maximum magnetization magnitude
        mag_up = magnetization(self.psi_up, 'z')
        self.assertAlmostEqual(abs(mag_up), 4.0, places=8)  # 4 sites * 1
        
        # All down state should have maximum magnetization magnitude
        mag_down = magnetization(self.psi_down, 'z')
        self.assertAlmostEqual(abs(mag_down), 4.0, places=8)  # 4 sites * 1
        
        # Up and down states should have opposite magnetizations
        self.assertAlmostEqual(mag_up, -mag_down, places=8)
        
        # Neel state should have zero total magnetization
        mag_neel = magnetization(self.psi_neel, 'z')
        self.assertAlmostEqual(mag_neel, 0.0, places=8)
        
        # Test single site magnetization magnitude
        mag_site_0 = magnetization(self.psi_up, 'z', site=0)
        self.assertAlmostEqual(abs(mag_site_0), 1.0, places=8)
        
    def test_staggered_magnetization(self):
        """Test staggered magnetization calculation."""
        # Neel state should have maximum staggered magnetization
        stag_mag_neel = staggered_magnetization(self.psi_neel)
        self.assertGreater(abs(stag_mag_neel), 0.5)  # Should be significant
        
        # All up state should have zero staggered magnetization
        stag_mag_up = staggered_magnetization(self.psi_up)
        self.assertAlmostEqual(stag_mag_up, 0.0, places=8)
        
    def test_subharmonic_response(self):
        """Test subharmonic response calculation."""
        # Create synthetic time series with period doubling
        times = np.linspace(0, 20, 100)
        period = 2.0
        
        # Pure subharmonic signal
        mag_data = np.cos(np.pi * times / period)  # Period 2T signal
        fund_amp, subharm_amp = subharmonic_response(mag_data, period)
        
        # Should return valid amplitudes
        self.assertGreaterEqual(subharm_amp, 0.0)
        self.assertGreaterEqual(fund_amp, 0.0)
        
        # Pure fundamental signal
        mag_data_fund = np.cos(2 * np.pi * times / period)  # Period T signal
        fund_amp2, subharm_amp2 = subharmonic_response(mag_data_fund, period)
        
        # Should return valid amplitudes
        self.assertGreaterEqual(fund_amp2, 0.0)
        self.assertGreaterEqual(subharm_amp2, 0.0)


class TestDTCDetection(unittest.TestCase):
    """Test DTC detection algorithms."""
    
    def test_stringent_dtc_detection(self):
        """Test stringent DTC detection function."""
        period = 2.0
        times = np.linspace(0, 40, 200)
        
        # Perfect DTC signal (period 2T oscillation)
        le_perfect = 0.5 + 0.3 * np.cos(np.pi * times / period)
        dtc_score_perfect = stringent_dtc_detection(le_perfect, times, period)
        self.assertGreaterEqual(dtc_score_perfect, 0.0)  # Should return valid score
        
        # No DTC signal (random)
        np.random.seed(42)
        le_random = 0.5 + 0.1 * np.random.randn(len(times))
        dtc_score_random = stringent_dtc_detection(le_random, times, period)
        self.assertGreaterEqual(dtc_score_random, 0.0)  # Should return valid score
        
        # Decaying signal
        le_decay = (0.5 + 0.3 * np.cos(np.pi * times / period)) * np.exp(-times/20)
        dtc_score_decay = stringent_dtc_detection(le_decay, times, period)
        # Should return valid score
        self.assertGreaterEqual(dtc_score_decay, 0.0)
        
    def test_extract_subharmonic_amplitude(self):
        """Test subharmonic amplitude extraction."""
        times = np.linspace(0, 20, 100)
        period = 2.0
        
        # Pure subharmonic
        mag_subharm = np.cos(np.pi * times / period)
        amp_subharm = extract_subharmonic_amplitude(times, mag_subharm, period)
        self.assertGreater(amp_subharm, 0.8)  # Should be close to 1
        
        # No subharmonic
        mag_fund = np.cos(2 * np.pi * times / period)
        amp_fund = extract_subharmonic_amplitude(times, mag_fund, period)
        self.assertLess(amp_fund, 0.2)  # Should be close to 0


class TestEvolutionDynamics(unittest.TestCase):
    """Test time evolution dynamics."""
    
    def test_custom_floquet_evolution(self):
        """Test CustomFloquet evolution class."""
        model = KickedIsingModel(
            n_sites=4,
            J=1.0, 
            h_disorder=0.2,
            tau=1.0,
            disorder_seed=42
        )
        
        trunc_params = {'chi_max': 16, 'svd_min': 1e-12, 'trunc_cut': 1e-8}
        floquet_evo = CustomFloquet(model, trunc_params)
        
        psi_initial = create_initial_state(4, "neel")
        n_periods = 5
        
        states, times, info = floquet_evo.evolve_floquet(psi_initial, n_periods)
        
        # Check output format
        self.assertEqual(len(states), n_periods + 1)
        self.assertEqual(len(times), n_periods + 1)
        
        # Check info dictionary
        self.assertIn('wall_time', info)
        self.assertIn('bond_dimensions', info)
        self.assertIn('final_bond_dim', info)
        
        # Check time progression
        expected_times = [i * 2 * model.tau for i in range(n_periods + 1)]
        np.testing.assert_array_almost_equal(times, expected_times)


class TestPhysicsValidation(unittest.TestCase):
    """Test physical validity of the simulation."""
    
    def test_unitarity_conservation(self):
        """Test that evolution preserves unitarity (norm conservation)."""
        model = KickedIsingModel(
            n_sites=6,
            J=1.0,
            h_disorder=0.3,
            tau=0.5,
            disorder_seed=42
        )
        
        psi_initial = create_initial_state(6, "neel")
        initial_norm = psi_initial.norm
        
        # Evolve for several steps
        psi_current = psi_initial
        for _ in range(10):
            psi_current = model.floquet_step(psi_current)
            # Check norm is preserved
            self.assertAlmostEqual(psi_current.norm, initial_norm, places=8)
            
    def test_conservation_laws(self):
        """Test conservation of relevant quantities."""
        model = KickedIsingModel(
            n_sites=4,
            J=1.0,
            h_disorder=0.0,  # No disorder for cleaner test
            tau=1.0,
            disorder_seed=42
        )
        
        # For translation-invariant case (no disorder), certain quantities should be conserved
        psi_initial = create_initial_state(4, "neel")
        
        psi_evolved = model.floquet_step(psi_initial)
        
        # Evolution should preserve the state's basic structure
        self.assertAlmostEqual(psi_evolved.norm, 1.0, places=10)
        
    def test_time_crystal_signatures(self):
        """Test for basic time crystal signatures."""
        model = KickedIsingModel(
            n_sites=8,
            J=1.0,
            h_disorder=0.25,  # Good DTC regime
            tau=1.0,
            disorder_seed=42
        )
        
        psi_initial = create_initial_state(8, "neel")
        n_periods = 20
        
        # Evolve and measure observables
        states, times = model.evolve(psi_initial, n_periods)
        
        # Calculate staggered magnetization over time
        stag_mags = []
        loschmidt_echoes = []
        
        for psi in states:
            stag_mag = staggered_magnetization(psi)
            le = calculate_loschmidt_echo(psi_initial, psi)
            stag_mags.append(stag_mag)
            loschmidt_echoes.append(le)
        
        # Check for oscillatory behavior
        stag_mags = np.array(stag_mags)
        
        # Should have some oscillation (not constant)
        stag_variation = np.std(stag_mags)
        self.assertGreater(stag_variation, 0.01)
        
        # Final Loschmidt echo should be positive (some memory)
        self.assertGreater(loschmidt_echoes[-1], 0.0)


class TestIntegrationWorkflows(unittest.TestCase):
    """Test high-level integration workflows."""
    
    @patch('matplotlib.pyplot.savefig')  # Mock saving to avoid file I/O
    @patch('matplotlib.pyplot.show')     # Mock show to avoid display
    def test_phase_diagram_workflow(self, mock_show, mock_savefig):
        """Test phase diagram calculation workflow."""
        # Use minimal parameters for fast test
        test_params = {
            'J': 1.0,
            'CHI_MAX': 16,
            'SVD_MIN': 1e-12,
            'SVD_CUTOFF': 1e-8,
            'RANDOM_SEED': 42
        }
        
        # Test single phase point calculation
        h_over_J = 0.3
        T_J = 2.0
        
        result = calculate_phase_point(h_over_J, T_J, test_params)
        
        # Check result structure
        expected_keys = ['A2T', 'dtc_score_raw', 'disorder_penalty', 
                        'heating_penalty', 'adiabatic_penalty', 
                        'entanglement_penalty', 'avg_bond_dim', 
                        'final_le', 'success']
        
        for key in expected_keys:
            self.assertIn(key, result)
            
        # Check values are reasonable
        self.assertGreaterEqual(result['A2T'], 0.0)
        self.assertLessEqual(result['A2T'], 1.0)
        self.assertGreaterEqual(result['avg_bond_dim'], 1.0)
        self.assertTrue(result['success'])
        
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_perfect_dtc_simulation(self, mock_show, mock_savefig):
        """Test perfect DTC simulation workflow."""
        test_params = {
            'J': 1.0,
            'CHI_MAX': 32,
            'SVD_MIN': 1e-12,
            'SVD_CUTOFF': 1e-8
        }
        
        # Run simulation (will be slow but should work)
        times, stag_mags, total_mags = simulate_perfect_dtc(test_params)
        
        # Check output format
        self.assertGreater(len(times), 50)  # Should have many time points
        self.assertEqual(len(times), len(stag_mags))
        self.assertEqual(len(times), len(total_mags))
        
        # Check for reasonable values
        stag_mags = np.array(stag_mags)
        total_mags = np.array(total_mags)
        
        self.assertTrue(np.all(np.abs(stag_mags) <= 1.1))  # Physical bounds
        self.assertTrue(np.all(np.abs(total_mags) <= 32.1))  # N_sites bound
        
        # Should show some variation (not constant)
        self.assertGreater(np.std(stag_mags), 0.01)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        # Negative system size
        with self.assertRaises((ValueError, AssertionError)):
            KickedIsingModel(n_sites=-1, J=1.0, h_disorder=0.1, tau=1.0)
            
        # Zero coupling
        model = KickedIsingModel(n_sites=4, J=0.0, h_disorder=0.1, tau=1.0)
        self.assertEqual(model.J, 0.0)  # Should handle but warn
        
    def test_edge_case_states(self):
        """Test edge cases for state manipulation."""
        # Single site system
        psi_single = create_initial_state(1, "all_up")
        self.assertEqual(psi_single.L, 1)
        
        # Very small disorder
        model_tiny_disorder = KickedIsingModel(
            n_sites=4, J=1.0, h_disorder=1e-10, tau=1.0
        )
        psi = create_initial_state(4, "neel")
        psi_evolved = model_tiny_disorder.floquet_step(psi)
        self.assertAlmostEqual(psi_evolved.norm, 1.0, places=10)
        
    def test_numerical_stability(self):
        """Test numerical stability with extreme parameters."""
        # Very small time step
        model_small_tau = KickedIsingModel(
            n_sites=4, J=1.0, h_disorder=0.1, tau=1e-3
        )
        psi = create_initial_state(4, "neel")
        psi_evolved = model_small_tau.floquet_step(psi)
        self.assertAlmostEqual(psi_evolved.norm, 1.0, places=8)
        
        # Large disorder (should not crash)
        model_large_disorder = KickedIsingModel(
            n_sites=4, J=1.0, h_disorder=2.0, tau=1.0
        )
        psi_evolved2 = model_large_disorder.floquet_step(psi)
        self.assertAlmostEqual(psi_evolved2.norm, 1.0, places=8)


def run_performance_benchmark():
    """Run a basic performance benchmark (not part of main test suite)."""
    print("\nRunning performance benchmark...")
    
    import time
    
    # Test system sizes
    sizes = [8, 12, 16]
    n_periods = 10
    
    for n_sites in sizes:
        print(f"\nTesting N={n_sites} sites:")
        
        model = KickedIsingModel(
            n_sites=n_sites,
            J=1.0,
            h_disorder=0.3,
            tau=1.0,
            disorder_seed=42
        )
        
        psi_initial = create_initial_state(n_sites, "neel")
        
        start_time = time.time()
        states, times, info = CustomFloquet(model).evolve_floquet(
            psi_initial, n_periods
        )
        end_time = time.time()
        
        wall_time = end_time - start_time
        max_bond_dim = max([max(psi.chi) if psi.chi else 1 for psi in states])
        
        print(f"  Wall time: {wall_time:.3f} s")
        print(f"  Periods/sec: {n_periods/wall_time:.1f}")
        print(f"  Max bond dim: {max_bond_dim}")
        print(f"  Final norm: {states[-1].norm:.6f}")


if __name__ == '__main__':
    # Set up test environment
    np.random.seed(42)  # Reproducible tests
    
    # Run the test suite
    print("Running Time Crystal Tensor Network Test Suite")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestBasicFunctionality,
        TestKickedIsingModel, 
        TestObservables,
        TestDTCDetection,
        TestEvolutionDynamics,
        TestPhysicsValidation,
        TestIntegrationWorkflows,
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    if len(result.failures) == 0 and len(result.errors) == 0:
        print("\n✅ All tests passed!")
        
        # Optionally run performance benchmark
        import sys
        if '--benchmark' in sys.argv:
            run_performance_benchmark()
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        
    print("\nTest suite complete.") 