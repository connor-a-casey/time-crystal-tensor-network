"""
Time-Evolving Block Decimation (TEBD) implementation using TeNPy.

This module provides a wrapper around TeNPy's TEBD algorithm for time evolution
specifically optimized for the kicked-Ising model.
"""

import numpy as np
from tenpy.networks.mps import MPS
from tenpy.algorithms import tebd
from tenpy.models.model import NearestNeighborModel
from tenpy.networks.site import SpinHalfSite
from tenpy.models.lattice import Chain
from typing import List, Dict, Tuple, Optional
import time


class TEBDEvolution:
    """
    TEBD evolution wrapper for time evolution of MPS states.
    
    This class provides a convenient interface to TeNPy's TEBD algorithm
    with support for custom Hamiltonians and truncation parameters.
    """
    
    def __init__(self, model, dt: float = 0.1, max_chi: int = 100, trunc_params: Dict = None):
        """
        Initialize TEBD evolution.
        
        Args:
            model: TeNPy model or custom model with gate structure
            dt: Time step
            max_chi: Maximum bond dimension
            trunc_params: Truncation parameters for TEBD
        """
        self.model = model
        self.dt = dt
        self.max_chi = max_chi
        
        if trunc_params is None:
            self.trunc_params = {
                'chi_max': max_chi,
                'svd_min': 1e-12,
                'trunc_cut': 1e-10
            }
        else:
            self.trunc_params = trunc_params
            if 'chi_max' not in self.trunc_params:
                self.trunc_params['chi_max'] = max_chi
    
    def evolve(self, psi_initial: MPS, total_time: float, 
               observe_every: int = 1) -> Tuple[List[MPS], List[float], Dict]:
        """
        Evolve MPS state using TEBD algorithm.
        
        Args:
            psi_initial: Initial MPS state
            total_time: Total evolution time
            observe_every: Store state every N steps
        
        Returns:
            Tuple of (states, times, info)
        """
        n_steps = int(total_time / self.dt)
        
        # initialize TEBD engine
        tebd_params = {
            'dt': self.dt,
            'order': 2,  # Second-order Suzuki-Trotter
            'N_steps': n_steps,
            'trunc_params': self.trunc_params,
            'verbose': False
        }
        
        # create TEBD engine
        eng = tebd.TEBDEngine(psi_initial, self.model, tebd_params)
        
        # storage for results
        states = [psi_initial.copy()]
        times = [0.0]
        bond_dims = [psi_initial.chi]
        entanglement_entropies = [psi_initial.entanglement_entropy()]
        
        # evolution loop
        start_time = time.time()
        for step in range(n_steps):
            # Single TEBD step
            eng.run_one_step()
            
            # store results
            if step % observe_every == 0:
                states.append(eng.psi.copy())
                times.append((step + 1) * self.dt)
                bond_dims.append(eng.psi.chi)
                entanglement_entropies.append(eng.psi.entanglement_entropy())
        
        wall_time = time.time() - start_time
        
        info = {
            'wall_time': wall_time,
            'bond_dimensions': bond_dims,
            'entanglement_entropies': entanglement_entropies,
            'truncation_errors': eng.trunc_err.eps,
            'final_bond_dim': eng.psi.chi,
            'n_steps': n_steps
        }
        
        return states, times, info
    
    def real_time_evolution(self, psi_initial: MPS, hamiltonian, 
                           total_time: float, observe_every: int = 1) -> Tuple[List[MPS], List[float], Dict]:
        """
        Real-time evolution with a given Hamiltonian.
        
        Args:
            psi_initial: Initial MPS state
            hamiltonian: Hamiltonian for evolution
            total_time: Total evolution time
            observe_every: Store state every N steps
        
        Returns:
            Tuple of (states, times, info)
        """
        # This method would implement real-time evolution
        # For now, we'll use the model-based evolution
        return self.evolve(psi_initial, total_time, observe_every)
    
    def suzuki_trotter_gates(self, hamiltonian_terms: Dict, dt: float) -> List[np.ndarray]:
        """
        Construct Suzuki-Trotter gates from Hamiltonian terms.
        
        Args:
            hamiltonian_terms: Dictionary of Hamiltonian terms
            dt: Time step
        
        Returns:
            List of time evolution gates
        """
        import scipy.linalg
        
        gates = []
        
        for term_name, term_operator in hamiltonian_terms.items():
            if term_name != 'single_site_terms':
                # Two-site terms
                gate = scipy.linalg.expm(-1j * dt * term_operator)
                gates.append(gate)
        
        return gates
    
    def benchmark_performance(self, psi_initial: MPS, n_steps: int = 100) -> Dict:
        """
        Benchmark TEBD performance.
        
        Args:
            psi_initial: Initial state
            n_steps: Number of steps for benchmark
        
        Returns:
            Performance metrics
        """
        total_time = n_steps * self.dt
        
        start_time = time.time()
        states, times, info = self.evolve(psi_initial, total_time, observe_every=n_steps)
        end_time = time.time()
        
        wall_time = end_time - start_time
        
        return {
            'wall_time': wall_time,
            'steps_per_second': n_steps / wall_time,
            'final_bond_dim': info['final_bond_dim'],
            'memory_usage': sum(info['bond_dimensions']) * 8 / 1024**2,  # Rough estimate in MB
            'truncation_error': info['truncation_errors'][-1] if info['truncation_errors'] else 0
        }
    
    def evolve_floquet_period(self, psi: MPS) -> MPS:
        """
        Evolve one Floquet period using the model's floquet_step method.
        
        Args:
            psi: Input MPS state
            
        Returns:
            MPS after one Floquet period
        """
        return self.model.floquet_step(psi, self.trunc_params)


class CustomFloquet:
    """
    Custom Floquet evolution for kicked-Ising model.
    
    This class implements the specific three-step Floquet evolution
    described in the paper without relying on TeNPy's built-in models.
    """
    
    def __init__(self, kicked_ising_model, trunc_params: Dict = None):
        """
        Initialize custom Floquet evolution.
        
        Args:
            kicked_ising_model: Instance of KickedIsingModel
            trunc_params: Truncation parameters
        """
        self.model = kicked_ising_model
        
        if trunc_params is None:
            self.trunc_params = {
                'chi_max': 100,
                'svd_min': 1e-12,
                'trunc_cut': 1e-10
            }
        else:
            self.trunc_params = trunc_params
    
    def evolve_floquet(self, psi_initial: MPS, n_periods: int, 
                      measure_every: int = 1) -> Tuple[List[MPS], List[float], Dict]:
        """
        Evolve using Floquet dynamics.
        
        Args:
            psi_initial: Initial MPS state
            n_periods: Number of Floquet periods
            measure_every: Measure every N periods
        
        Returns:
            Tuple of (states, times, info)
        """
        states = [psi_initial.copy()]
        times = [0.0]
        bond_dims = [max(psi_initial.chi) if psi_initial.chi else 1]
        
        psi_current = psi_initial.copy()
        
        start_time = time.time()
        
        for period in range(n_periods):
            # Apply one Floquet step
            psi_current = self.model.floquet_step(psi_current, self.trunc_params)
            
            # Store results
            if period % measure_every == 0:
                states.append(psi_current.copy())
                times.append((period + 1) * 2 * self.model.tau)
                bond_dims.append(max(psi_current.chi) if psi_current.chi else 1)
        
        wall_time = time.time() - start_time
        
        info = {
            'wall_time': wall_time,
            'bond_dimensions': bond_dims,
            'periods_per_second': n_periods / wall_time,
            'final_bond_dim': max(psi_current.chi) if psi_current.chi else 1,
            'n_periods': n_periods
        }
        
        return states, times, info 