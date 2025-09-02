"""
Floquet Kicked-Ising Model implementation using TeNPy.

This module implements the kicked-Ising model from the paper:

U_F = exp[-iτ/2 Σ_j(J σ_j^z σ_{j+1}^z + h σ_j^z)] 
    × exp[-iπ/2 Σ_j σ_j^x]
    × exp[-iτ/2 Σ_j(J σ_j^z σ_{j+1}^z + h σ_j^z)]

where J is the Ising coupling, h is the longitudinal field, and τ is the half-period. 
The original formula was outlined in [Phys. Rev. Lett. 117, 090402 (2016) 
https://arxiv.org/abs/1603.08001]
"""

import numpy as np
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite
from tenpy.models.lattice import Chain
from tenpy.models.model import CouplingModel
from tenpy.linalg import np_conserved as npc
from typing import List, Dict, Tuple
import scipy.linalg


class KickedIsingModel:
    """
    Floquet kicked-Ising model for discrete time crystal simulations.
    
    This implements the three-step Floquet evolution:
    1. Ising evolution for time τ/2
    2. π-pulse (X rotation) 
    3. Ising evolution for time τ/2
    """
    
    def __init__(self, n_sites: int, J: float, h_disorder: float, tau: float, 
                 bc: str = 'open', disorder_seed: int = None):
        """
        Initialize the kicked-Ising model.
        
        Args:
            n_sites: Number of sites
            J: Ising coupling strength
            h_disorder: Disorder strength for longitudinal field
            tau: Half-period between kicks
            bc: Boundary conditions ('open' or 'periodic')
            disorder_seed: Random seed for disorder
        """
        self.n_sites = n_sites
        self.J = J
        self.h_disorder = h_disorder
        self.tau = tau
        self.bc = bc
        
        # Generate random disorder
        if disorder_seed is not None:
            np.random.seed(disorder_seed)
        
        # Disorder field h_i drawn from [-h_disorder, h_disorder]
        self.h_fields = np.random.uniform(-h_disorder, h_disorder, n_sites)
        
        # Create sites
        self.sites = [SpinHalfSite(conserve=None) for _ in range(n_sites)]
        
        # Pauli matrices
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.sigma_I = np.eye(2, dtype=complex)
        
        # Precompute gates
        self._prepare_gates()
    
    def _prepare_gates(self):
        """Prepare the Floquet evolution gates."""
        # π-pulse gate: exp[-iπ/2 σ^x]
        self.pi_pulse_gate = scipy.linalg.expm(-1j * np.pi/2 * self.sigma_x)
        
        # Ising evolution gates for each bond
        self.ising_gates = []
        
        for i in range(self.n_sites - 1):
            # Two-site Hamiltonian: J σ_i^z σ_{i+1}^z + h_i σ_i^z + h_{i+1} σ_{i+1}^z
            h_two_site = (self.J * np.kron(self.sigma_z, self.sigma_z) + 
                         self.h_fields[i] * np.kron(self.sigma_z, self.sigma_I) +
                         self.h_fields[i+1] * np.kron(self.sigma_I, self.sigma_z))
            
            # Time evolution operator for τ/2
            u_gate = scipy.linalg.expm(-1j * self.tau/2 * h_two_site)
            self.ising_gates.append(u_gate)
        
        # Handle boundary conditions
        if self.bc == 'periodic' and self.n_sites > 2:
            # Add boundary coupling
            h_boundary = (self.J * np.kron(self.sigma_z, self.sigma_z) + 
                         self.h_fields[-1] * np.kron(self.sigma_z, self.sigma_I) +
                         self.h_fields[0] * np.kron(self.sigma_I, self.sigma_z))
            u_boundary = scipy.linalg.expm(-1j * self.tau/2 * h_boundary)
            self.ising_gates.append(u_boundary)
    
    def floquet_step(self, psi: MPS, trunc_params: Dict = None) -> MPS:
        """
        Apply one Floquet step to the MPS.
        
        Args:
            psi: Input MPS state
            trunc_params: Truncation parameters for TEBD
        
        Returns:
            MPS after one Floquet step
        """
        if trunc_params is None:
            trunc_params = {'chi_max': 100, 'svd_min': 1e-12}
        
        # Start with a copy of the input state
        psi_evolved = psi.copy()
        
        # Step 1: First Ising evolution (τ/2)
        psi_evolved = self._apply_ising_evolution(psi_evolved, trunc_params)
        
        # Step 2: π-pulse on all sites
        psi_evolved = self._apply_pi_pulse(psi_evolved, trunc_params)
        
        # Step 3: Second Ising evolution (τ/2)
        psi_evolved = self._apply_ising_evolution(psi_evolved, trunc_params)
        
        return psi_evolved
    
    def _apply_ising_evolution(self, psi: MPS, trunc_params: Dict) -> MPS:
        """Apply Ising evolution gates using TEBD-like approach."""
        psi_evolved = psi.copy()
        
        # Apply even bonds
        for i in range(0, len(self.ising_gates), 2):
            if i < len(self.ising_gates):
                bond_idx = i if i < self.n_sites - 1 else (i, 0)  # Handle periodic boundary
                psi_evolved = self._apply_two_site_gate(
                    psi_evolved, self.ising_gates[i], bond_idx, trunc_params
                )
        
        # Apply odd bonds
        for i in range(1, len(self.ising_gates), 2):
            if i < len(self.ising_gates):
                bond_idx = i if i < self.n_sites - 1 else (i, 0)  # Handle periodic boundary
                psi_evolved = self._apply_two_site_gate(
                    psi_evolved, self.ising_gates[i], bond_idx, trunc_params
                )
        
        return psi_evolved
    
    def _apply_pi_pulse(self, psi: MPS, trunc_params: Dict) -> MPS:
        """Apply π-pulse to all sites."""
        psi_evolved = psi.copy()
        
        # Apply single-site π-pulse to each site
        for i in range(self.n_sites):
            psi_evolved = self._apply_single_site_gate(
                psi_evolved, self.pi_pulse_gate, i
            )
        
        return psi_evolved
    
    def _apply_two_site_gate(self, psi: MPS, gate: np.ndarray, bond_idx: int, 
                            trunc_params: Dict) -> MPS:
        """Apply a two-site gate using TeNPy's built-in methods."""
        # Get proper leg charges from the sites
        site_leg = psi.sites[0].leg
        leg_in = site_leg
        leg_out = site_leg.conj()
        
        # Convert gate to TeNPy format
        gate_tensor = npc.Array.from_ndarray(
            gate.reshape(2, 2, 2, 2), 
            labels=['p0', 'p1', 'p0*', 'p1*'],
            legcharges=[leg_in, leg_in, leg_out, leg_out]
        )
        
        # Apply gate
        psi_new = psi.copy()
        
        if isinstance(bond_idx, int):
            i, j = bond_idx, bond_idx + 1
        else:
            i, j = bond_idx
        
        # Use TeNPy's apply_local_op for two-site gates
        psi_new.apply_local_op(i, gate_tensor, unitary=True)
        
        return psi_new
    
    def _apply_single_site_gate(self, psi: MPS, gate: np.ndarray, site: int) -> MPS:
        """Apply a single-site gate."""
        # get proper leg charges from the site
        site_leg = psi.sites[0].leg
        leg_in = site_leg
        leg_out = site_leg.conj()
        
        # convert gate to TeNPy format
        gate_tensor = npc.Array.from_ndarray(
            gate, 
            labels=['p', 'p*'], 
            legcharges=[leg_in, leg_out]
        )
        
        # Apply gate
        psi_new = psi.copy()
        psi_new.apply_local_op(site, gate_tensor, unitary=True)
        
        return psi_new
    
    def evolve(self, psi_initial: MPS, n_steps: int, 
              trunc_params: Dict = None) -> Tuple[List[MPS], List[float]]:
        """
        Evolve the system for multiple Floquet steps.
        
        Args:
            psi_initial: Initial MPS state
            n_steps: Number of Floquet steps
            trunc_params: Truncation parameters
        
        Returns:
            Tuple of (states_list, times_list)
        """
        if trunc_params is None:
            trunc_params = {'chi_max': 100, 'svd_min': 1e-12}
        
        states = [psi_initial.copy()]
        times = [0.0]
        
        psi_current = psi_initial.copy()
        
        for step in range(n_steps):
            # apply Floquet step
            psi_current = self.floquet_step(psi_current, trunc_params)
            
            # Store results
            states.append(psi_current.copy())
            times.append((step + 1) * 2 * self.tau)  # Full period is 2τ
        
        return states, times
    
    def get_hamiltonian_terms(self) -> Dict[str, np.ndarray]:
        """
        Get the Hamiltonian terms for analysis.
        
        Returns:
            Dictionary containing Hamiltonian components
        """
        return {
            'J': self.J,
            'h_fields': self.h_fields,
            'tau': self.tau,
            'pi_pulse': self.pi_pulse_gate,
            'ising_gates': self.ising_gates
        }
    
    def calculate_phase_diagram_point(self, psi_initial: MPS, n_steps: int = 200,
                                    trunc_params: Dict = None) -> Dict[str, float]:
        """
        Calculate observables for a single point in the phase diagram.
        
        Args:
            psi_initial: Initial state
            n_steps: Number of evolution steps
            trunc_params: Truncation parameters
        
        Returns:
            Dictionary with calculated observables
        """
        from ..core.observables import (calculate_loschmidt_echo, magnetization, 
                                       subharmonic_response, order_parameter)
        
        states, times = self.evolve(psi_initial, n_steps, trunc_params)
        
        # Cclculate observables
        loschmidt_echoes = []
        magnetizations = []
        
        for psi in states:
            # Loschmidt echo
            le = calculate_loschmidt_echo(psi_initial, psi)
            loschmidt_echoes.append(le)
            
            # Magnetization
            mag = magnetization(psi, 'z')
            magnetizations.append(mag)
        
        # calculate subharmonic response
        drive_period = 2 * self.tau
        fund_amp, subharm_amp = subharmonic_response(magnetizations, drive_period)
        
        # order parameter
        sublattice_a = list(range(0, self.n_sites, 2))
        sublattice_b = list(range(1, self.n_sites, 2))
        order_param = order_parameter(states[-1], sublattice_a, sublattice_b)
        
        return {
            'loschmidt_echo_final': loschmidt_echoes[-1],
            'subharmonic_amplitude': subharm_amp,
            'fundamental_amplitude': fund_amp,
                         'order_parameter': order_param,
             'max_bond_dimension': max(states[-1].chi) if states[-1].chi else 1,
             'final_magnetization': magnetizations[-1]
        } 