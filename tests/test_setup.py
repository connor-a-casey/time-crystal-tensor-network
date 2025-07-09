#!/usr/bin/env python3
"""
Test script to verify the time crystal framework setup.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from core.tensor_utils import create_initial_state, pauli_matrices
        print("✓ Core tensor utilities imported successfully")
        
        from core.observables import calculate_loschmidt_echo, magnetization
        print("✓ Observable calculations imported successfully")
        
        from models.kicked_ising import KickedIsingModel
        print("✓ Kicked-Ising model imported successfully")
        
        from dynamics.tebd_evolution import CustomFloquet
        print("✓ Time evolution algorithms imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of the framework."""
    print("\nTesting basic functionality...")
    
    try:
        # Test Pauli matrices
        from core.tensor_utils import pauli_matrices
        pauli = pauli_matrices()
        assert pauli['X'].shape == (2, 2)
        print("✓ Pauli matrices work correctly")
        
        # Test state creation
        from core.tensor_utils import create_initial_state
        psi = create_initial_state(4, state_type="neel")
        assert psi.L == 4
        print("✓ Initial state creation works")
        
        # Test magnetization calculation
        from core.observables import magnetization
        try:
            mag = magnetization(psi, 'z')
            assert isinstance(mag, (float, complex))
            print("✓ Magnetization calculation works")
        except Exception as e:
            print(f"✗ Magnetization calculation failed: {e}")
            return False
        
        # Test kicked-Ising model
        from models.kicked_ising import KickedIsingModel
        model = KickedIsingModel(n_sites=4, J=1.0, h_disorder=0.3, tau=1.0)
        assert model.n_sites == 4
        print("✓ Kicked-Ising model initialization works")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False

def test_small_simulation():
    """Test a small simulation to ensure everything works together."""
    print("\nTesting small simulation...")
    
    try:
        from core.tensor_utils import create_initial_state
        from core.observables import calculate_loschmidt_echo
        from models.kicked_ising import KickedIsingModel
        from dynamics.tebd_evolution import CustomFloquet
        
        # Small system
        n_sites = 6
        model = KickedIsingModel(n_sites=n_sites, J=1.0, h_disorder=0.3, tau=1.0)
        psi_initial = create_initial_state(n_sites, state_type="neel")
        
        # Short evolution
        trunc_params = {'chi_max': 16, 'svd_min': 1e-12, 'trunc_cut': 1e-10}
        floquet = CustomFloquet(model, trunc_params)
        
        states, times, info = floquet.evolve_floquet(psi_initial, n_periods=5, measure_every=1)
        
        # Calculate Loschmidt echo
        le = calculate_loschmidt_echo(psi_initial, states[-1])
        
        assert len(states) == 6  # Initial + 5 periods
        assert 0 <= le <= 1  # Loschmidt echo should be between 0 and 1
        
        print(f"✓ Small simulation completed successfully")
        print(f"  - Final Loschmidt echo: {le:.6f}")
        print(f"  - Evolution time: {info['wall_time']:.3f} seconds")
        print(f"  - Final bond dimension: {info['final_bond_dim']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Small simulation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Time Crystal Framework - Setup Test")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test basic functionality
    functionality_ok = test_basic_functionality()
    
    # Test simulation
    simulation_ok = test_small_simulation()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if imports_ok and functionality_ok and simulation_ok:
        print("🎉 All tests passed! The framework is ready to use.")
        print("\nNext steps:")
        print("1. Run: python examples/loschmidt_echo_demo.py")
        print("2. Open: notebooks/01_basic_example.ipynb")
        print("3. Install dependencies: pip install -r requirements.txt")
        return True
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 