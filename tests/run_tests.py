import argparse
import os
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List

try:
    import coverage
except ImportError:
    coverage = None

try:
    import cv2
except ImportError:
    cv2 = None

"""Test Runner

Comprehensive test runner for YOLO Vision Studio with coverage reporting.
"""

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def discover_tests() -> unittest.TestSuite:
    """Discover all tests in the tests directory"""
    test_dir = Path(__file__).parent
    loader = unittest.TestLoader()

    # Discover tests
    suite = loader.discover(str(test_dir), pattern="test_*.py")
    return suite


def run_tests_with_coverage() -> Dict[str, Any]:
    """Run tests with coverage reporting if available"""
    try:
        # Initialize coverage
        cov = coverage.Coverage()
        cov.start()

        # Run tests
        suite = discover_tests()
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        # Stop coverage and generate report
        cov.stop()
        cov.save()

        # Generate coverage report
        print("\n" + "=" * 50)
        print("COVERAGE REPORT")
        print("=" * 50)
        cov.report()

        # Generate HTML report if possible
        try:
            html_dir = Path(__file__).parent / "coverage_html"
            cov.html_report(directory=str(html_dir))
            print("\nHTML coverage report generated: {html_dir}")
        except Exception as e:
            print("Could not generate HTML report: {e}")

        return {
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped) if hasattr(result, "skipped") else 0,
            "success": result.wasSuccessful(),
            "coverage_available": True,
        }

    except ImportError:
        print("Coverage.py not available. Running tests without coverage.")
        return run_tests_basic()


def run_tests_basic() -> Dict[str, Any]:
    """Run tests without coverage"""
    suite = discover_tests()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped) if hasattr(result, "skipped") else 0,
        "success": result.wasSuccessful(),
        "coverage_available": False,
    }


def print_test_summary(results: Dict[str, Any]):
    """Print test summary"""
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    print("Tests run: {results['tests_run']}")
    print("Failures: {results['failures']}")
    print("Errors: {results['errors']}")
    print("Skipped: {results['skipped']}")

    if results["success"]:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED!")

    if not results["coverage_available"]:
        print("\n📊 To get coverage reports, install coverage.py:")
        print("   pip install coverage")


def check_dependencies():
    """Check if required dependencies are available"""
    print("Checking dependencies...")

    dependencies = {
        "torch": "PyTorch",
        "numpy": "NumPy",
        "opencv-python": "OpenCV",
        "pyyaml": "PyYAML",
        "pathlib": "Pathlib (built-in)",
    }

    missing = []
    available = []

    for module, name in dependencies.items():
        try:
            if module == "opencv-python":
                import cv2
            elif module == "pathlib":
                # pathlib is built-in, no need to import
                pass
            else:
                __import__(module)
            available.append(name)
        except ImportError:
            missing.append(name)

    print("✅ Available: {', '.join(available)}")
    if missing:
        print("❌ Missing: {', '.join(missing)}")
        print("\nInstall missing dependencies with:")
        print("pip install torch numpy opencv-python pyyaml")

    return len(missing) == 0


def run_specific_test(test_name: str):
    """Run a specific test module"""
    try:
        # Import the specific test module
        test_module = __import__("test_{test_name}", fromlist=[""])

        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_module)

        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        return result.wasSuccessful()

    except ImportError as e:
        print("Could not import test module 'test_{test_name}': {e}")
        return False


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Run YOLO Vision Studio tests")
    parser.add_argument(
        "--test", type=str, help="Run specific test module (e.g., hardware, config)"
    )
    parser.add_argument(
        "--no-coverage", action="store_true", help="Skip coverage reporting"
    )
    parser.add_argument(
        "--check-deps", action="store_true", help="Check dependencies only"
    )

    args = parser.parse_args()

    # Check dependencies
    if args.check_deps:
        deps_ok = check_dependencies()
        sys.exit(0 if deps_ok else 1)

    print("YOLO Vision Studio Test Suite")
    print("=" * 50)

    # Check basic dependencies
    if not check_dependencies():
        print("\n❌ Missing required dependencies. Please install them first.")
        sys.exit(1)

    start_time = time.time()

    # Run specific test if requested
    if args.test:
        print("\nRunning specific test: {args.test}")
        success = run_specific_test(args.test)
        print("\nTest {'PASSED' if success else 'FAILED'}")
        sys.exit(0 if success else 1)

    # Run all tests
    print("\nRunning all tests...")

    if args.no_coverage:
        results = run_tests_basic()
    else:
        results = run_tests_with_coverage()

    end_time = time.time()

    # Print summary
    print_test_summary(results)
    print("\nTotal time: {end_time - start_time:.2f} seconds")

    # Exit with appropriate code
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
