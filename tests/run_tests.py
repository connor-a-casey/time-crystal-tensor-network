#!/usr/bin/env python3
"""
Comprehensive test runner for time crystal tensor network project.

This script runs the complete test suite (basic functionality, physics validation,
and performance benchmarks) and provides detailed reporting. Designed for 
continuous integration and development validation.
"""

import os
import sys
import time
import unittest
import subprocess
from typing import Dict, List, Tuple
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def run_test_module(module_name: str, verbosity: int = 1) -> Tuple[bool, Dict]:
    """
    Run a specific test module and return results.
    
    Args:
        module_name: Name of the test module to run
        verbosity: Verbosity level for test output
        
    Returns:
        Tuple of (success, results_dict)
    """
    try:
        # Import the test module
        test_module = __import__(f'test_{module_name}', fromlist=[''])
        
        # Discover tests
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_module)
        
        # Run tests
        runner = unittest.TextTestRunner(
            verbosity=verbosity,
            stream=open(os.devnull, 'w') if verbosity == 0 else sys.stdout
        )
        
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        results = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
            'success': len(result.failures) == 0 and len(result.errors) == 0,
            'wall_time': end_time - start_time,
            'failure_details': result.failures,
            'error_details': result.errors
        }
        
        return results['success'], results
        
    except Exception as e:
        return False, {
            'tests_run': 0,
            'failures': 0,
            'errors': 1,
            'skipped': 0,
            'success': False,
            'wall_time': 0.0,
            'failure_details': [],
            'error_details': [('Module Import', str(e))]
        }


def check_dependencies() -> Dict[str, bool]:
    """Check if all required dependencies are available."""
    dependencies = {
        'numpy': True,
        'matplotlib': True,
        'tenpy': True,
        'scipy': True,
        'tqdm': True,
        'psutil': True
    }
    
    for dep in dependencies:
        try:
            __import__(dep)
        except ImportError:
            dependencies[dep] = False
    
    return dependencies


def run_code_quality_checks() -> Dict[str, any]:
    """Run basic code quality checks."""
    results = {}
    
    # Check if main modules can be imported
    core_modules = [
        'core.tensor_utils',
        'core.observables', 
        'models.kicked_ising',
        'dynamics.tebd_evolution'
    ]
    
    import_results = {}
    for module in core_modules:
        try:
            __import__(module, fromlist=[''])
            import_results[module] = True
        except Exception as e:
            import_results[module] = str(e)
    
    results['imports'] = import_results
    
    # Check if main script runs without errors
    try:
        # Try importing main without running it
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        import main
        results['main_import'] = True
    except Exception as e:
        results['main_import'] = str(e)
    
    return results


def generate_test_report(all_results: Dict[str, Dict], output_file: str = None):
    """Generate a comprehensive test report."""
    report_lines = []
    
    # Header
    report_lines.append("=" * 80)
    report_lines.append("TIME CRYSTAL TENSOR NETWORK - TEST REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Summary
    total_tests = sum(results.get('tests_run', 0) for results in all_results.values())
    total_failures = sum(results.get('failures', 0) for results in all_results.values())
    total_errors = sum(results.get('errors', 0) for results in all_results.values())
    total_time = sum(results.get('wall_time', 0) for results in all_results.values())
    
    overall_success = total_failures == 0 and total_errors == 0
    
    report_lines.append("SUMMARY")
    report_lines.append("-" * 40)
    report_lines.append(f"Total Tests:    {total_tests}")
    report_lines.append(f"Failures:       {total_failures}")
    report_lines.append(f"Errors:         {total_errors}")
    report_lines.append(f"Total Time:     {total_time:.2f}s")
    report_lines.append(f"Overall Result: {'PASS' if overall_success else 'FAIL'}")
    report_lines.append("")
    
    # Module-by-module results
    report_lines.append("DETAILED RESULTS")
    report_lines.append("-" * 40)
    
    for module_name, results in all_results.items():
        if not results:
            continue
            
        status = "PASS" if results.get('success', False) else "FAIL"
        tests_run = results.get('tests_run', 0)
        failures = results.get('failures', 0)
        errors = results.get('errors', 0)
        wall_time = results.get('wall_time', 0)
        
        report_lines.append(f"{module_name:20s} | {status:4s} | "
                           f"{tests_run:3d} tests | {failures:2d} failures | "
                           f"{errors:2d} errors | {wall_time:6.2f}s")
    
    report_lines.append("")
    
    # Failure details
    has_failures = any(results.get('failures', 0) > 0 or results.get('errors', 0) > 0 
                      for results in all_results.values())
    
    if has_failures:
        report_lines.append("FAILURE DETAILS")
        report_lines.append("-" * 40)
        
        for module_name, results in all_results.items():
            if not results:
                continue
                
            failures = results.get('failure_details', [])
            errors = results.get('error_details', [])
            
            if failures or errors:
                report_lines.append(f"\n{module_name}:")
                
                for test, traceback in failures:
                    report_lines.append(f"  FAILURE: {test}")
                    # Extract just the assertion message
                    lines = traceback.split('\n')
                    for line in lines:
                        if 'AssertionError:' in line:
                            report_lines.append(f"    {line.strip()}")
                            break
                
                for test, traceback in errors:
                    report_lines.append(f"  ERROR: {test}")
                    # Extract the main error message
                    lines = traceback.split('\n')
                    for line in lines:
                        if any(exc in line for exc in ['Error:', 'Exception:']):
                            report_lines.append(f"    {line.strip()}")
                            break
    
    # Performance summary
    report_lines.append("")
    report_lines.append("PERFORMANCE SUMMARY")
    report_lines.append("-" * 40)
    
    if 'performance' in all_results:
        perf_results = all_results['performance']
        if perf_results.get('success', False):
            report_lines.append("✓ Performance tests passed")
            report_lines.append(f"  Execution time: {perf_results.get('wall_time', 0):.2f}s")
        else:
            report_lines.append("✗ Performance tests failed")
    
    # Recommendations
    report_lines.append("")
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("-" * 40)
    
    if overall_success:
        report_lines.append("✓ All tests passing - codebase appears healthy")
        if total_time > 60:
            report_lines.append("⚠ Consider optimizing slow tests for CI")
    else:
        report_lines.append("✗ Fix failing tests before deployment")
        if total_errors > 0:
            report_lines.append("! Priority: Fix error conditions first")
        if total_failures > 0:
            report_lines.append("! Check assertion failures for logic errors")
    
    # Join report
    report = '\n'.join(report_lines)
    
    # Output report
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Test report saved to: {output_file}")
    
    print(report)


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='Run time crystal tensor network tests')
    parser.add_argument('--verbose', '-v', action='count', default=1,
                       help='Increase verbosity (use -vv for very verbose)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file for test report')
    
    args = parser.parse_args()
    
    print("Time Crystal Tensor Network - Test Suite")
    print("=" * 50)
    
    # Check dependencies first
    print("Checking dependencies...")
    deps = check_dependencies()
    missing_deps = [dep for dep, available in deps.items() if not available]
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Please install missing dependencies before running tests.")
        return 1
    else:
        print("✅ All dependencies available")
    
    # Check code quality
    print("\nRunning code quality checks...")
    quality_results = run_code_quality_checks()
    
    import_issues = [mod for mod, result in quality_results['imports'].items() 
                    if result is not True]
    if import_issues:
        print(f"❌ Import issues: {', '.join(import_issues)}")
        return 1
    else:
        print("✅ All core modules import successfully")
    
    # Always run all test modules
    test_modules = ['basic_functionality', 'physics_validation', 'performance']
    
    print(f"\nRunning all test modules: {', '.join(test_modules)}")
    print("")
    
    # Run tests
    all_results = {}
    overall_success = True
    
    for module_name in test_modules:
        print(f"Running {module_name} tests...")
        success, results = run_test_module(module_name, args.verbose)
        all_results[module_name] = results
        
        if not success:
            overall_success = False
        
        # Print module summary
        if results:
            status = "PASS" if success else "FAIL"
            print(f"  {status}: {results['tests_run']} tests, "
                  f"{results['failures']} failures, {results['errors']} errors "
                  f"({results['wall_time']:.1f}s)")
        print("")
    
    # Generate final report
    print("")
    generate_test_report(all_results, args.output)
    
    # Return appropriate exit code
    return 0 if overall_success else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code) 