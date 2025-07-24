#!/usr/bin/env python3
"""
Test runner script for AI Anchor system.

Runs comprehensive test suites with different configurations and reporting.
"""

import sys
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional


def run_command(cmd: List[str], description: str) -> Dict[str, any]:
    """
    Run a command and return results.
    
    Args:
        cmd: Command to run as list of strings
        description: Description of what the command does
        
    Returns:
        Dictionary with results
    """
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} completed in {duration:.2f}s")
            return {
                "success": True,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        else:
            print(f"❌ {description} failed in {duration:.2f}s")
            print(f"Exit code: {result.returncode}")
            if result.stdout:
                print(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                print(f"STDERR:\n{result.stderr}")
            
            return {
                "success": False,
                "duration": duration,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
    
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ {description} failed with exception: {e}")
        return {
            "success": False,
            "duration": duration,
            "error": str(e)
        }


def run_basic_structure_tests() -> bool:
    """Run basic structure tests without external dependencies."""
    print("\n" + "="*60)
    print("🧪 BASIC STRUCTURE TESTS")
    print("="*60)
    
    tests = [
        {
            "script": "test_anchor_basic.py",
            "description": "Anchor agent structure test"
        },
        {
            "script": "test_web_basic.py", 
            "description": "Web interface structure test"
        }
    ]
    
    all_passed = True
    
    for test in tests:
        result = run_command(
            ["python", test["script"]],
            test["description"]
        )
        
        if not result["success"]:
            all_passed = False
    
    return all_passed


def run_unit_tests(
    test_path: Optional[str] = None,
    verbose: bool = False,
    coverage: bool = False
) -> bool:
    """
    Run unit tests using pytest.
    
    Args:
        test_path: Specific test path to run
        verbose: Enable verbose output
        coverage: Enable coverage reporting
        
    Returns:
        True if all tests passed
    """
    print("\n" + "="*60)
    print("🧪 UNIT TESTS")
    print("="*60)
    
    # Check if pytest is available
    try:
        import pytest
        print("✅ pytest is available")
    except ImportError:
        print("❌ pytest not available - skipping unit tests")
        print("Install with: pip install pytest pytest-asyncio")
        return False
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    if test_path:
        cmd.append(test_path)
    else:
        cmd.append("tests/")
    
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    if coverage:
        try:
            import pytest_cov
            cmd.extend(["--cov=src", "--cov-report=term-missing"])
            print("✅ Coverage reporting enabled")
        except ImportError:
            print("⚠️ pytest-cov not available - skipping coverage")
    
    # Add other useful options
    cmd.extend([
        "--tb=short",  # Shorter traceback format
        "--strict-markers",  # Strict marker checking
        "--disable-warnings",  # Reduce noise
    ])
    
    result = run_command(cmd, "Unit tests with pytest")
    
    return result["success"]


def run_integration_tests(verbose: bool = False) -> bool:
    """
    Run integration tests.
    
    Args:
        verbose: Enable verbose output
        
    Returns:
        True if all tests passed
    """
    print("\n" + "="*60)
    print("🧪 INTEGRATION TESTS")
    print("="*60)
    
    try:
        import pytest
    except ImportError:
        print("❌ pytest not available - skipping integration tests")
        return False
    
    cmd = [
        "python", "-m", "pytest",
        "tests/test_integration/",
        "-v" if verbose else "-q",
        "--tb=short",
        "--disable-warnings"
    ]
    
    result = run_command(cmd, "Integration tests")
    
    return result["success"]


def run_performance_tests() -> bool:
    """Run performance benchmarks."""
    print("\n" + "="*60)
    print("🚀 PERFORMANCE TESTS")
    print("="*60)
    
    try:
        import pytest
    except ImportError:
        print("❌ pytest not available - skipping performance tests")
        return False
    
    # Look for performance test markers
    cmd = [
        "python", "-m", "pytest",
        "-m", "performance",
        "-v",
        "--tb=short",
        "--disable-warnings"
    ]
    
    result = run_command(cmd, "Performance benchmarks")
    
    if not result["success"] and "no tests ran" in result["stdout"].lower():
        print("ℹ️ No performance tests found (mark tests with @pytest.mark.performance)")
        return True  # Not a failure if no performance tests exist
    
    return result["success"]


def run_lint_checks() -> bool:
    """Run code quality checks."""
    print("\n" + "="*60)
    print("🔍 CODE QUALITY CHECKS")
    print("="*60)
    
    checks = []
    
    # Check for flake8
    try:
        import flake8
        checks.append({
            "cmd": ["python", "-m", "flake8", "src/", "--max-line-length=120", "--ignore=E203,W503"],
            "description": "flake8 style check"
        })
    except ImportError:
        print("⚠️ flake8 not available - skipping style checks")
    
    # Check for mypy
    try:
        import mypy
        checks.append({
            "cmd": ["python", "-m", "mypy", "src/", "--ignore-missing-imports"],
            "description": "mypy type checking"
        })
    except ImportError:
        print("⚠️ mypy not available - skipping type checks")
    
    # Check for bandit (security)
    try:
        import bandit
        checks.append({
            "cmd": ["python", "-m", "bandit", "-r", "src/", "-f", "json"],
            "description": "bandit security check"
        })
    except ImportError:
        print("⚠️ bandit not available - skipping security checks")
    
    if not checks:
        print("ℹ️ No linting tools available")
        return True
    
    all_passed = True
    
    for check in checks:
        result = run_command(check["cmd"], check["description"])
        if not result["success"]:
            all_passed = False
    
    return all_passed


def generate_test_report(results: Dict[str, bool]) -> None:
    """Generate a summary test report."""
    print("\n" + "="*60)
    print("📊 TEST SUMMARY REPORT")
    print("="*60)
    
    total_suites = len(results)
    passed_suites = sum(1 for result in results.values() if result)
    failed_suites = total_suites - passed_suites
    
    print(f"\nTest Suites Run: {total_suites}")
    print(f"✅ Passed: {passed_suites}")
    print(f"❌ Failed: {failed_suites}")
    
    if failed_suites == 0:
        print(f"\n🎉 All test suites passed!")
        print("System is ready for deployment.")
    else:
        print(f"\n⚠️ {failed_suites} test suite(s) failed.")
        print("Review failures before deployment.")
    
    print("\nDetailed Results:")
    for suite_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {suite_name}: {status}")
    
    # Success rate
    success_rate = (passed_suites / total_suites) * 100
    print(f"\nOverall Success Rate: {success_rate:.1f}%")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="AI Anchor Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --unit-only        # Run only unit tests  
  python run_tests.py --basic-only       # Run only basic structure tests
  python run_tests.py --verbose          # Verbose output
  python run_tests.py --coverage         # Include coverage report
  python run_tests.py --no-lint          # Skip linting
        """
    )
    
    parser.add_argument(
        "--basic-only",
        action="store_true",
        help="Run only basic structure tests"
    )
    
    parser.add_argument(
        "--unit-only", 
        action="store_true",
        help="Run only unit tests"
    )
    
    parser.add_argument(
        "--integration-only",
        action="store_true",
        help="Run only integration tests"
    )
    
    parser.add_argument(
        "--test-path",
        type=str,
        help="Specific test path to run"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true", 
        help="Include coverage reporting"
    )
    
    parser.add_argument(
        "--no-lint",
        action="store_true",
        help="Skip code quality checks"
    )
    
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Run performance tests"
    )
    
    args = parser.parse_args()
    
    print("🎙️ AI Anchor Test Runner")
    print("=" * 60)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Run basic structure tests
    if args.basic_only or not any([args.unit_only, args.integration_only]):
        results["Basic Structure"] = run_basic_structure_tests()
    
    # Run unit tests
    if args.unit_only or not any([args.basic_only, args.integration_only]):
        results["Unit Tests"] = run_unit_tests(
            test_path=args.test_path,
            verbose=args.verbose,
            coverage=args.coverage
        )
    
    # Run integration tests
    if args.integration_only or not any([args.basic_only, args.unit_only]):
        results["Integration Tests"] = run_integration_tests(verbose=args.verbose)
    
    # Run performance tests
    if args.performance:
        results["Performance Tests"] = run_performance_tests()
    
    # Run code quality checks
    if not args.no_lint and not any([args.basic_only, args.unit_only, args.integration_only]):
        results["Code Quality"] = run_lint_checks()
    
    # Generate report
    generate_test_report(results)
    
    # Exit with appropriate code
    if all(results.values()):
        print(f"\n✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Some tests failed. See details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()