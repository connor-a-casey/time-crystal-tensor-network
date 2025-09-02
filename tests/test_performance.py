#!/usr/bin/env python3
"""
Performance and benchmarking tests for time crystal tensor network code.

These tests verify that the code runs within reasonable time and memory
constraints, and help identify performance bottlenecks.
"""

import unittest
import numpy as np
import os
import sys
import time
import psutil
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.tensor_utils import create_initial_state
from core.observables import calculate_loschmidt_echo, staggered_magnetization
from models.kicked_ising import KickedIsingModel
from dynamics.tebd_evolution import CustomFloquet
from main import calculate_phase_point


class TestPerformance(unittest.TestCase):
    """Test performance characteristics of the codebase."""
    
    def setUp(self):
        """Set up performance testing environment."""
        self.process = psutil.Process()
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
        
    def test_single_floquet_step_performance(self):
        """Test performance of single Floquet step."""
        sizes = [8, 12, 16]
        max_times = [0.1, 0.5, 2.0]  # Maximum allowed times in seconds
        
        for n_sites, max_time in zip(sizes, max_times):
            with self.subTest(n_sites=n_sites):
                model = KickedIsingModel(
                    n_sites=n_sites,
                    J=1.0,
                    h_disorder=0.3,
                    tau=1.0,
                    disorder_seed=42
                )
                
                psi_initial = create_initial_state(n_sites, "neel")
                
                # Time single step
                start_time = time.time()
                psi_evolved = model.floquet_step(psi_initial)
                end_time = time.time()
                
                step_time = end_time - start_time
                
                # Check performance constraint
                self.assertLess(step_time, max_time,
                               f"Single step took {step_time:.3f}s for N={n_sites}, "
                               f"exceeding limit of {max_time}s")
                
                # Verify correctness
                self.assertAlmostEqual(psi_evolved.norm, 1.0, places=8)
                
    def test_evolution_scaling(self):
        """Test how evolution time scales with system size and evolution length."""
        base_size = 8
        base_periods = 10
        
        # Test system size scaling
        sizes = [8, 12, 16]
        size_times = []
        
        for n_sites in sizes:
            model = KickedIsingModel(
                n_sites=n_sites,
                J=1.0,
                h_disorder=0.3,
                tau=1.0,
                disorder_seed=42
            )
            
            psi_initial = create_initial_state(n_sites, "neel")
            
            start_time = time.time()
            states, times = model.evolve(psi_initial, base_periods)
            end_time = time.time()
            
            evolution_time = end_time - start_time
            size_times.append(evolution_time)
            
            # Should not take too long
            self.assertLess(evolution_time, 10.0,
                           f"Evolution took too long for N={n_sites}")
        
        # Test period scaling
        periods = [5, 10, 20]
        period_times = []
        
        model = KickedIsingModel(
            n_sites=base_size,
            J=1.0,
            h_disorder=0.3,
            tau=1.0,
            disorder_seed=42
        )
        
        for n_periods in periods:
            psi_initial = create_initial_state(base_size, "neel")
            
            start_time = time.time()
            states, times = model.evolve(psi_initial, n_periods)
            end_time = time.time()
            
            evolution_time = end_time - start_time
            period_times.append(evolution_time)
        
        # Time should scale roughly linearly with number of periods
        time_ratio = period_times[2] / period_times[0]  # 20 periods / 5 periods
        expected_ratio = periods[2] / periods[0]  # 4
        
        # Allow factor of 2 tolerance for overhead
        self.assertLess(time_ratio, expected_ratio * 2,
                       "Evolution time scaling worse than expected")
        
    def test_memory_usage(self):
        """Test memory usage characteristics."""
        initial_memory = self.get_memory_usage()
        
        # Create several models and states
        models = []
        states = []
        
        for i in range(5):
            model = KickedIsingModel(
                n_sites=12,
                J=1.0,
                h_disorder=0.3,
                tau=1.0,
                disorder_seed=42 + i
            )
            models.append(model)
            
            psi = create_initial_state(12, "neel")
            # Evolve to create non-trivial state
            for _ in range(5):
                psi = model.floquet_step(psi)
            states.append(psi)
        
        peak_memory = self.get_memory_usage()
        memory_increase = peak_memory - initial_memory
        
        # Clean up
        del models, states
        
        # Memory increase should be reasonable (< 100 MB for test)
        self.assertLess(memory_increase, 100,
                       f"Memory usage increased by {memory_increase:.1f} MB")
        
    def test_bond_dimension_performance(self):
        """Test performance with different bond dimension limits."""
        chi_values = [8, 16, 32, 64]
        times = []
        
        model = KickedIsingModel(
            n_sites=12,
            J=1.0,
            h_disorder=0.3,
            tau=1.0,
            disorder_seed=42
        )
        
        psi_initial = create_initial_state(12, "neel")
        n_periods = 10
        
        for chi_max in chi_values:
            trunc_params = {'chi_max': chi_max, 'svd_min': 1e-12, 'trunc_cut': 1e-8}
            floquet_evo = CustomFloquet(model, trunc_params)
            
            start_time = time.time()
            states, times_evo, info = floquet_evo.evolve_floquet(psi_initial, n_periods)
            end_time = time.time()
            
            evolution_time = end_time - start_time
            times.append(evolution_time)
            
            # Should complete in reasonable time
            self.assertLess(evolution_time, 30.0,
                           f"Evolution with chi_max={chi_max} took too long")
        
        # Higher bond dimensions should take longer but not exponentially so
        for i in range(1, len(times)):
            ratio = times[i] / times[i-1]
            self.assertLess(ratio, 10.0,
                           f"Time scaling too steep between chi_max={chi_values[i-1]} "
                           f"and chi_max={chi_values[i]}")


class TestBenchmarks(unittest.TestCase):
    """Benchmark tests for different components."""
    
    def setUp(self):
        """Set up benchmarking environment."""
        self.process = psutil.Process()
        
    def benchmark_observable_calculations(self):
        """Benchmark observable calculation performance."""
        # Create test state
        model = KickedIsingModel(
            n_sites=16,
            J=1.0,
            h_disorder=0.3,
            tau=1.0,
            disorder_seed=42
        )
        
        psi = create_initial_state(16, "neel")
        # Evolve to get interesting state
        for _ in range(10):
            psi = model.floquet_step(psi)
        
        # Benchmark different observables
        observables = {
            'staggered_magnetization': lambda: staggered_magnetization(psi),
            'total_magnetization_z': lambda: sum(psi.expectation_value('Sz', sites=[i]) for i in range(psi.L)),
            'loschmidt_echo': lambda: calculate_loschmidt_echo(create_initial_state(16, "neel"), psi)
        }
        
        times = {}
        n_runs = 10
        
        for obs_name, obs_func in observables.items():
            start_time = time.time()
            for _ in range(n_runs):
                result = obs_func()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / n_runs
            times[obs_name] = avg_time
            
            # All should be fast
            self.assertLess(avg_time, 0.1,
                           f"{obs_name} took {avg_time:.4f}s on average")
        
        return times
        
    def test_phase_point_calculation_performance(self):
        """Test performance of single phase diagram point calculation."""
        test_params = {
            'J': 1.0,
            'CHI_MAX': 32,
            'SVD_MIN': 1e-12,
            'SVD_CUTOFF': 1e-8,
            'RANDOM_SEED': 42
        }
        
        h_over_J = 0.3
        T_J = 2.0
        
        start_time = time.time()
        result = calculate_phase_point(h_over_J, T_J, test_params)
        end_time = time.time()
        
        calculation_time = end_time - start_time
        
        # Should complete in reasonable time (generous limit for CI)
        self.assertLess(calculation_time, 60.0,
                       f"Phase point calculation took {calculation_time:.1f}s")
        
        # Should return valid result
        self.assertTrue(result['success'])
        self.assertGreaterEqual(result['A2T'], 0.0)
        self.assertLessEqual(result['A2T'], 1.0)
        
    def test_concurrent_model_creation(self):
        """Test performance when creating multiple models."""
        n_models = 10
        models = []
        
        start_time = time.time()
        
        for i in range(n_models):
            model = KickedIsingModel(
                n_sites=8,
                J=1.0,
                h_disorder=0.3,
                tau=1.0,
                disorder_seed=42 + i
            )
            models.append(model)
        
        end_time = time.time()
        creation_time = end_time - start_time
        
        # Should be fast to create models
        self.assertLess(creation_time, 5.0,
                       f"Creating {n_models} models took {creation_time:.1f}s")
        
        # All models should have different disorder realizations
        h_fields = [model.h_fields for model in models]
        
        # Check that models with different seeds have different disorder
        for i in range(1, len(h_fields)):
            self.assertFalse(np.allclose(h_fields[0], h_fields[i]),
                           f"Models 0 and {i} have identical disorder fields")


class TestScalabilityLimits(unittest.TestCase):
    """Test scalability limits and identify bottlenecks."""
    
    def test_maximum_practical_system_size(self):
        """Test the largest system size that can be handled reasonably."""
        max_sizes_to_test = [16, 20, 24]  # Progressively larger
        max_time_per_size = 30.0  # seconds
        
        largest_working_size = 0
        
        for n_sites in max_sizes_to_test:
            try:
                model = KickedIsingModel(
                    n_sites=n_sites,
                    J=1.0,
                    h_disorder=0.3,
                    tau=1.0,
                    disorder_seed=42
                )
                
                psi_initial = create_initial_state(n_sites, "neel")
                
                # Test with conservative bond dimension
                trunc_params = {'chi_max': 64, 'svd_min': 1e-12, 'trunc_cut': 1e-8}
                floquet_evo = CustomFloquet(model, trunc_params)
                
                start_time = time.time()
                states, times, info = floquet_evo.evolve_floquet(psi_initial, 5)  # Short evolution
                end_time = time.time()
                
                evolution_time = end_time - start_time
                
                if evolution_time < max_time_per_size:
                    largest_working_size = n_sites
                else:
                    break
                    
            except (MemoryError, Exception):
                break
        
        # Should be able to handle at least 16 sites
        self.assertGreaterEqual(largest_working_size, 16,
                               f"Cannot handle systems with ≥16 sites efficiently")
        
    def test_maximum_evolution_length(self):
        """Test maximum evolution length for fixed system size."""
        model = KickedIsingModel(
            n_sites=12,
            J=1.0,
            h_disorder=0.3,
            tau=1.0,
            disorder_seed=42
        )
        
        psi_initial = create_initial_state(12, "neel")
        
        # Test progressively longer evolutions
        period_counts = [10, 50, 100, 200]
        max_time_per_test = 60.0  # seconds
        
        longest_working_evolution = 0
        
        for n_periods in period_counts:
            try:
                start_time = time.time()
                states, times = model.evolve(psi_initial, n_periods)
                end_time = time.time()
                
                evolution_time = end_time - start_time
                
                if evolution_time < max_time_per_test:
                    longest_working_evolution = n_periods
                else:
                    break
                    
            except (MemoryError, Exception):
                break
        
        # Should handle at least 50 periods
        self.assertGreaterEqual(longest_working_evolution, 50,
                               "Cannot handle moderate evolution lengths")
        
    def test_memory_scaling_with_bond_dimension(self):
        """Test how memory scales with bond dimension."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        chi_values = [8, 16, 32]
        memory_usage = []
        
        model = KickedIsingModel(
            n_sites=12,
            J=1.0,
            h_disorder=0.3,
            tau=1.0,
            disorder_seed=42
        )
        
        for chi_max in chi_values:
            # Force garbage collection before each test
            import gc
            gc.collect()
            
            before_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            psi_initial = create_initial_state(12, "neel")
            trunc_params = {'chi_max': chi_max, 'svd_min': 1e-12, 'trunc_cut': 1e-8}
            floquet_evo = CustomFloquet(model, trunc_params)
            
            # Evolve to build up entanglement
            states, times, info = floquet_evo.evolve_floquet(psi_initial, 15)
            
            after_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = after_memory - before_memory
            memory_usage.append(memory_increase)
            
            # Clean up
            del states, floquet_evo, psi_initial
            
        # Memory should not grow too quickly with bond dimension
        for i, mem_use in enumerate(memory_usage):
            self.assertLess(mem_use, 200,
                           f"Memory usage {mem_use:.1f} MB too high for chi_max={chi_values[i]}")


def run_comprehensive_benchmark():
    """Run a comprehensive benchmark (not part of main test suite)."""
    print("\nRunning Comprehensive Performance Benchmark")
    print("=" * 60)
    
    # System size benchmark
    print("\n1. System Size Scaling:")
    sizes = [8, 12, 16, 20]
    n_periods = 10
    
    for n_sites in sizes:
        try:
            model = KickedIsingModel(
                n_sites=n_sites,
                J=1.0,
                h_disorder=0.3,
                tau=1.0,
                disorder_seed=42
            )
            
            psi_initial = create_initial_state(n_sites, "neel")
            
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            states, times = model.evolve(psi_initial, n_periods)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            wall_time = end_time - start_time
            memory_used = end_memory - start_memory
            max_bond_dim = max([max(psi.chi) if psi.chi else 1 for psi in states])
            
            print(f"  N={n_sites:2d}: {wall_time:6.2f}s, {memory_used:6.1f}MB, χ_max={max_bond_dim:2d}")
            
        except Exception as e:
            print(f"  N={n_sites:2d}: Failed - {str(e)[:50]}")
    
    # Bond dimension benchmark
    print("\n2. Bond Dimension Scaling:")
    chi_values = [8, 16, 32, 64, 128]
    
    model = KickedIsingModel(
        n_sites=12,
        J=1.0,
        h_disorder=0.3,
        tau=1.0,
        disorder_seed=42
    )
    
    for chi_max in chi_values:
        try:
            psi_initial = create_initial_state(12, "neel")
            trunc_params = {'chi_max': chi_max, 'svd_min': 1e-12, 'trunc_cut': 1e-8}
            floquet_evo = CustomFloquet(model, trunc_params)
            
            start_time = time.time()
            states, times, info = floquet_evo.evolve_floquet(psi_initial, 15)
            end_time = time.time()
            
            wall_time = end_time - start_time
            final_chi = info['final_bond_dim']
            
            print(f"  χ_max={chi_max:3d}: {wall_time:6.2f}s, final_χ={final_chi:2d}")
            
        except Exception as e:
            print(f"  χ_max={chi_max:3d}: Failed - {str(e)[:50]}")
    
    print("\nBenchmark complete.")


if __name__ == '__main__':
    print("Running Performance Tests")
    print("=" * 40)
    
    # Run standard performance tests
    unittest.main(verbosity=2, exit=False)
    
    # Run comprehensive benchmark if requested
    if '--benchmark' in sys.argv:
        run_comprehensive_benchmark() 