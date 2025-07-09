"""
Tensor network utilities using TeNPy for MPS and MPO operations.
"""

import numpy as np
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
from tenpy.models.spins import SpinChain
from tenpy.algorithms import tebd
from tenpy.linalg import np_conserved as npc


def pauli_matrices():
    """Return the Pauli matrices as numpy arrays."""
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    sigma_I = np.eye(2, dtype=complex)
    
    return {
        'I': sigma_I,
        'X': sigma_x,
        'Y': sigma_y,
        'Z': sigma_z
    }


def create_initial_state(n_sites: int, state_type: str = "all_up") -> MPS:
    """
    Create initial MPS state using TeNPy.
    
    Args:
        n_sites: Number of sites
        state_type: Type of state ("all_up", "all_down", "neel", "random")
    
    Returns:
        TeNPy MPS object
    """
    from tenpy.networks.site import SpinHalfSite
    
    # Create spin-1/2 sites
    sites = [SpinHalfSite(conserve='parity') for _ in range(n_sites)]
    
    if state_type == "all_up":
        # All spins up (|0⟩ state)
        product_state = ["up"] * n_sites
    elif state_type == "all_down":
        # All spins down (|1⟩ state)
        product_state = ["down"] * n_sites
    elif state_type == "neel":
        # Néel state |↑↓↑↓...⟩
        product_state = ["up" if i % 2 == 0 else "down" for i in range(n_sites)]
    elif state_type == "random":
        # Random product state
        product_state = [np.random.choice(["up", "down"]) for _ in range(n_sites)]
    else:
        raise ValueError(f"Unknown state type: {state_type}")
    
    # Create MPS from product state
    psi = MPS.from_product_state(sites, product_state, bc='finite')
    
    return psi


def apply_two_site_gate(psi: MPS, gate: np.ndarray, i: int, j: int, 
                       trunc_params: dict = None) -> MPS:
    """
    Apply a two-site gate to an MPS.
    
    Args:
        psi: Input MPS
        gate: Two-site gate as 4x4 matrix
        i, j: Site indices (must be adjacent)
        trunc_params: Truncation parameters for SVD
    
    Returns:
        Updated MPS
    """
    if trunc_params is None:
        trunc_params = {'chi_max': 100, 'svd_min': 1e-12}
    
    if abs(i - j) != 1:
        raise ValueError("Sites must be adjacent for two-site gate")
    
    # Ensure correct ordering
    if i > j:
        i, j = j, i
    
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
    
    # Apply gate using TeNPy's built-in function
    psi_new = psi.copy()
    psi_new.apply_local_op(i, gate_tensor, unitary=True)
    
    return psi_new


def create_time_evolution_gates(J: float, h: float, tau: float, n_sites: int):
    """
    Create time evolution gates for the kicked Ising model.
    
    Args:
        J: Ising coupling strength
        h: Longitudinal field strength
        tau: Time step
        n_sites: Number of sites
    
    Returns:
        Dictionary of gates for TEBD evolution
    """
    pauli = pauli_matrices()
    
    # Ising interaction term: J σ_i^z σ_{i+1}^z
    zz_interaction = J * np.kron(pauli['Z'], pauli['Z'])
    
    # Longitudinal field terms: h σ_i^z
    z_field_left = h * np.kron(pauli['Z'], pauli['I'])
    z_field_right = h * np.kron(pauli['I'], pauli['Z'])
    
    # Total two-site Hamiltonian
    h_two_site = zz_interaction + z_field_left + z_field_right
    
    # Time evolution operator
    u_two_site = np.exp(-1j * tau * h_two_site)
    
    # π-pulse (X rotation)
    pi_pulse = np.exp(-1j * np.pi/2 * pauli['X'])
    
    return {
        'ising_evolution': u_two_site,
        'pi_pulse': pi_pulse
    }


def measure_magnetization(psi: MPS, direction: str = 'z') -> float:
    """
    Measure total magnetization of an MPS.
    
    Args:
        psi: MPS state
        direction: Direction ('x', 'y', or 'z')
    
    Returns:
        Total magnetization
    """
    pauli = pauli_matrices()
    op = pauli[direction.upper()]
    
    total_mag = 0.0
    for i in range(psi.L):
        # Convert to TeNPy operator
        op_tensor = npc.Array.from_ndarray(op, labels=['p', 'p*'])
        mag_i = psi.expectation_value(op_tensor, sites=[i])
        total_mag += mag_i.real
    
    return total_mag


def calculate_entanglement_entropy(psi: MPS, cut: int) -> float:
    """
    Calculate entanglement entropy across a cut.
    
    Args:
        psi: MPS state
        cut: Cut position
    
    Returns:
        Entanglement entropy
    """
    return psi.entanglement_entropy()[cut]


def mps_overlap(psi1: MPS, psi2: MPS) -> complex:
    """
    Calculate overlap between two MPS states.
    
    Args:
        psi1, psi2: MPS states
    
    Returns:
        Overlap ⟨psi1|psi2⟩
    """
    return psi1.overlap(psi2) 