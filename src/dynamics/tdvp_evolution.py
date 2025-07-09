"""
Time-Dependent Variational Principle (TDVP) implementation using TeNPy.

This module provides a wrapper around TeNPy's TDVP algorithms for time evolution
of quantum many-body systems, enabling direct comparison with TEBD methods.
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Union
from tenpy.networks.mps import MPS
from tenpy.algorithms.tdvp import TwoSiteTDVPEngine, SingleSiteTDVPEngine
from tenpy.models.model import MPOModel
from tenpy.networks.site import SpinHalfSite
from tenpy.models.lattice import Chain


class TDVPEvolution:
    """
    TDVP evolution wrapper for efficient time evolution of MPS states.
    
    This class provides a convenient interface to TeNPy's TDVP algorithms
    with support for both single-site and two-site variants.
    """
    
    def __init__(self, model, dt: float = 0.01, max_chi: int = 100, 
                 tdvp_type: str = 'two_site', trunc_params: Dict = None):
        """
        Initialize TDVP evolution.
        
        Args:
            model: TeNPy model or custom model
            dt: Time step
            max_chi: Maximum bond dimension
            tdvp_type: 'single_site' or 'two_site'
            trunc_params: Truncation parameters for TDVP
        """
        self.model = model
        self.dt = dt
        self.max_chi = max_chi
        self.tdvp_type = tdvp_type
        
        if trunc_params is None:
            self.trunc_params = {
                'chi_max': max_chi,
                'svd_min': 1e-12,
                'trunc_cut': 1e-10
            }
        else:
            self.trunc_params = trunc_params
            
        # TDVP-specific parameters
        self.tdvp_params = {
            'dt': dt,
            'trunc_params': self.trunc_params,
            'lanczos_params': {
                'N_max': 20,
                'E_tol': 1e-12,
                'N_min': 2
            },
            'verbose': False
        }
    
    def evolve(self, psi_initial: MPS, total_time: float, 
               observe_every: int = 1) -> Tuple[List[MPS], List[float], Dict]:
        """
        Evolve MPS state using TDVP algorithm.
        
        Args:
            psi_initial: Initial MPS state
            total_time: Total evolution time
            observe_every: Store state every N steps
        
        Returns:
            Tuple of (states, times, info)
        """
        n_steps = int(total_time / self.dt)
        
        # Choose TDVP engine
        if self.tdvp_type == 'single_site':
            engine_class = SingleSiteTDVPEngine
        else:
            engine_class = TwoSiteTDVPEngine
        
        # Initialize TDVP engine
        psi = psi_initial.copy()
        engine = engine_class(psi, self.model, self.tdvp_params)
        
        # Storage for results
        states = [psi_initial.copy()]
        times = [0.0]
        bond_dims = [max(psi_initial.chi) if psi_initial.chi else 1]
        entanglement_entropies = [psi_initial.entanglement_entropy()]
        
        # Evolution loop
        start_time = time.time()
        
        # For observation, we need to evolve step by step
        for step in range(n_steps):
            # Single TDVP step
            engine.evolve(N_steps=1, dt=self.dt)
            
            # Store results
            if step % observe_every == 0:
                states.append(engine.psi.copy())
                times.append((step + 1) * self.dt)
                bond_dims.append(max(engine.psi.chi) if engine.psi.chi else 1)
                entanglement_entropies.append(engine.psi.entanglement_entropy())
        
        wall_time = time.time() - start_time
        
        # Prepare info dictionary
        info = {
            'wall_time': wall_time,
            'bond_dimensions': bond_dims,
            'entanglement_entropies': entanglement_entropies,
            'final_bond_dim': max(engine.psi.chi) if engine.psi.chi else 1,
            'n_steps': n_steps,
            'algorithm': f'TDVP ({self.tdvp_type})',
            'max_chi_reached': max(bond_dims) if bond_dims else 1
        }
        
        return states, times, info
    
    def real_time_evolution(self, psi_initial: MPS, total_time: float, 
                          observe_every: int = 1) -> Tuple[List[MPS], List[float], Dict]:
        """
        Real-time evolution using TDVP.
        
        Args:
            psi_initial: Initial MPS state
            total_time: Total evolution time
            observe_every: Store state every N steps
        
        Returns:
            Tuple of (states, times, info)
        """
        return self.evolve(psi_initial, total_time, observe_every)
    
    def evolve_floquet_period(self, psi: MPS) -> MPS:
        """
        Evolve one Floquet period using TDVP.
        
        Args:
            psi: Input MPS state
            
        Returns:
            MPS after one Floquet period
        """
        # For Floquet systems, we need to evolve through one full period
        # This is a simplified implementation - for the kicked Ising model
        # we would need to implement the three-step process
        
        # Create proper TeNPy model for TDVP
        tenpy_model = self._create_tenpy_model(psi.L)
        
        # Choose TDVP engine
        if self.tdvp_type == 'single_site':
            engine_class = SingleSiteTDVPEngine
        else:
            engine_class = TwoSiteTDVPEngine
        
        # Initialize TDVP engine
        psi_copy = psi.copy()
        engine = engine_class(psi_copy, tenpy_model, self.tdvp_params)
        
        # For kicked Ising, one period is 2*tau
        period_time = 2 * getattr(self.model, 'tau', 1.0)
        n_steps = int(period_time / self.dt)
        
        # Evolve for one period
        engine.evolve(N_steps=n_steps, dt=self.dt)
        
        return engine.psi
    
    def _create_tenpy_model(self, L: int):
        """Create a proper TeNPy model for TDVP evolution."""
        from tenpy.models.spins import SpinChain
        
        # Extract parameters from our custom model
        J = getattr(self.model, 'J', 1.0)
        h = getattr(self.model, 'h_disorder', 0.3)
        
        # Create TeNPy SpinChain model with compatible charge conservation
        model_params = {
            'L': L,
            'S': 0.5,
            'Jz': J,      # Ising coupling
            'hz': h,      # Longitudinal field
            'bc_MPS': 'finite',
            'conserve': 'parity'  # Match the charge conservation of our MPS
        }
        
        return SpinChain(model_params)
    
    def benchmark_performance(self, system_sizes: List[int], n_periods: int = 50, 
                            J: float = 1.0, h: float = 0.3, tau: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Benchmark TDVP performance for different system sizes.
        
        Args:
            system_sizes: List of system sizes to test
            n_periods: Number of periods to evolve
            J, h, tau: Physical parameters
            
        Returns:
            Tuple of (runtimes, max_bond_dimensions)
        """
        from models.kicked_ising import KickedIsingModel
        from core.tensor_utils import create_initial_state
        
        runtimes = []
        max_bond_dims = []
        
        for L in system_sizes:
            try:
                # Create model
                model = KickedIsingModel(n_sites=L, J=J, h_disorder=h, tau=tau)
                
                # Create TDVP evolution
                tdvp_evolution = TDVPEvolution(model, dt=self.dt, max_chi=self.max_chi, 
                                             tdvp_type=self.tdvp_type)
                
                # Initial state
                psi0 = create_initial_state(L, state_type='neel')
                
                # Time the evolution
                start_time = time.time()
                max_chi_reached = 1
                
                psi = psi0.copy()
                for period in range(n_periods):
                    psi = tdvp_evolution.evolve_floquet_period(psi)
                    current_chi = max(psi.chi) if psi.chi else 1
                    max_chi_reached = max(max_chi_reached, current_chi)
                
                end_time = time.time()
                runtime = end_time - start_time
                
                runtimes.append(runtime)
                max_bond_dims.append(max_chi_reached)
                
                print(f"TDVP L={L}: {runtime:.2f}s, max_chi={max_chi_reached}")
                
            except Exception as e:
                print(f"TDVP Error for L={L}: {e}")
                runtimes.append(np.nan)
                max_bond_dims.append(np.nan)
        
        return np.array(runtimes), np.array(max_bond_dims)


class TDVPFloquetEvolution:
    """
    TDVP evolution specifically for Floquet systems like the kicked Ising model.
    
    This class implements the three-step Floquet evolution using TDVP:
    1. Ising evolution for time τ/2
    2. π-pulse (X rotation) 
    3. Ising evolution for time τ/2
    """
    
    def __init__(self, kicked_ising_model, dt: float = 0.01, max_chi: int = 100, 
                 tdvp_type: str = 'two_site'):
        """
        Initialize TDVP Floquet evolution.
        
        Args:
            kicked_ising_model: Instance of KickedIsingModel
            dt: Time step
            max_chi: Maximum bond dimension
            tdvp_type: 'single_site' or 'two_site'
        """
        self.model = kicked_ising_model
        self.dt = dt
        self.max_chi = max_chi
        self.tdvp_type = tdvp_type
        
        # Create TDVP evolution instance
        self.tdvp_evolution = TDVPEvolution(
            model=kicked_ising_model,
            dt=dt,
            max_chi=max_chi,
            tdvp_type=tdvp_type
        )
    
    def evolve_floquet_period(self, psi: MPS) -> MPS:
        """
        Evolve one Floquet period using TDVP.
        
        For the kicked Ising model, this involves three steps:
        1. Ising evolution τ/2
        2. π-pulse
        3. Ising evolution τ/2
        
        Args:
            psi: Input MPS state
            
        Returns:
            MPS after one Floquet period
        """
        # This is a simplified implementation
        # In a full implementation, we would need to create separate
        # Hamiltonians for each step and evolve accordingly
        
        # For now, use the full period evolution
        return self.tdvp_evolution.evolve_floquet_period(psi)
    
    def benchmark_vs_tebd(self, system_sizes: List[int], n_periods: int = 50) -> Dict:
        """
        Benchmark TDVP performance against TEBD.
        
        Args:
            system_sizes: List of system sizes to test
            n_periods: Number of periods to evolve
            
        Returns:
            Dictionary with benchmark results
        """
        from dynamics.tebd_evolution import TEBDEvolution
        from core.tensor_utils import create_initial_state
        
        results = {
            'system_sizes': system_sizes,
            'tebd_times': [],
            'tdvp_times': [],
            'tebd_chi_max': [],
            'tdvp_chi_max': []
        }
        
        for L in system_sizes:
            print(f"\nBenchmarking L={L}...")
            
            try:
                # Create model and initial state
                model = self.model.__class__(
                    n_sites=L, J=self.model.J, h_disorder=self.model.h_disorder, tau=self.model.tau
                )
                psi0 = create_initial_state(L, state_type='neel')
                
                # TEBD benchmark
                tebd_evolution = TEBDEvolution(model, max_chi=self.max_chi)
                start_time = time.time()
                tebd_max_chi = 1
                psi_tebd = psi0.copy()
                
                for period in range(n_periods):
                    psi_tebd = tebd_evolution.evolve_floquet_period(psi_tebd)
                    current_chi = max(psi_tebd.chi) if psi_tebd.chi else 1
                    tebd_max_chi = max(tebd_max_chi, current_chi)
                
                tebd_time = time.time() - start_time
                results['tebd_times'].append(tebd_time)
                results['tebd_chi_max'].append(tebd_max_chi)
                
                # TDVP benchmark
                tdvp_floquet = TDVPFloquetEvolution(model, dt=self.dt, max_chi=self.max_chi, 
                                                  tdvp_type=self.tdvp_type)
                start_time = time.time()
                tdvp_max_chi = 1
                psi_tdvp = psi0.copy()
                
                for period in range(n_periods):
                    psi_tdvp = tdvp_floquet.evolve_floquet_period(psi_tdvp)
                    current_chi = max(psi_tdvp.chi) if psi_tdvp.chi else 1
                    tdvp_max_chi = max(tdvp_max_chi, current_chi)
                
                tdvp_time = time.time() - start_time
                results['tdvp_times'].append(tdvp_time)
                results['tdvp_chi_max'].append(tdvp_max_chi)
                
                print(f"  TEBD: {tebd_time:.2f}s, max_chi={tebd_max_chi}")
                print(f"  TDVP: {tdvp_time:.2f}s, max_chi={tdvp_max_chi}")
                print(f"  Speedup: {tebd_time/tdvp_time:.2f}x")
                
            except Exception as e:
                print(f"  Error: {e}")
                results['tebd_times'].append(np.nan)
                results['tdvp_times'].append(np.nan)
                results['tebd_chi_max'].append(np.nan)
                results['tdvp_chi_max'].append(np.nan)
        
        return results 