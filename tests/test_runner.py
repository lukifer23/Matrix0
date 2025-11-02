"""Unified test runner and reporting for Matrix0 test suite."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import pytest


logger = logging.getLogger(__name__)


class Matrix0TestRunner:
    """Unified test runner with reporting and profiling."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("test_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Any] = {}
    
    def run_tests(
        self,
        test_paths: List[str],
        markers: Optional[List[str]] = None,
        verbose: bool = False,
        performance: bool = False,
        integration: bool = False,
        stress: bool = False,
    ) -> int:
        """Run tests and collect results."""
        pytest_args = []
        
        # Add test paths
        if test_paths:
            pytest_args.extend(test_paths)
        else:
            pytest_args.append("tests/")
        
        # Add markers
        if markers:
            pytest_args.extend(["-m", " or ".join(markers)])
        
        # Performance tests
        if performance:
            pytest_args.extend(["-m", "performance"])
        
        # Integration tests
        if integration:
            pytest_args.extend(["-m", "integration"])
        
        # Stress tests
        if stress:
            pytest_args.extend(["-m", "stress"])
        
        # Output options
        pytest_args.extend([
            "-v" if verbose else "-q",
            "--tb=short",
            f"--junitxml={self.output_dir}/junit.xml",
            f"--html={self.output_dir}/report.html",
            "--self-contained-html",
        ])
        
        # Run pytest
        start_time = time.time()
        exit_code = pytest.main(pytest_args)
        elapsed = time.time() - start_time
        
        # Collect results
        self.results = {
            "exit_code": exit_code,
            "elapsed_time": elapsed,
            "test_paths": test_paths,
            "markers": markers,
        }
        
        return exit_code
    
    def save_results(self):
        """Save test results to JSON."""
        results_file = self.output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {results_file}")


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="Matrix0 Test Suite Runner")
    parser.add_argument(
        "test_paths",
        nargs="*",
        help="Test paths to run (default: all tests)",
    )
    parser.add_argument(
        "-m", "--marker",
        action="append",
        help="Run tests with specific markers",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Run performance tests",
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests",
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Run stress tests",
    )
    parser.add_argument(
        "--error-handling",
        action="store_true",
        help="Run error handling tests",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output directory for test results",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create test runner
    runner = Matrix0TestRunner(output_dir=args.output)
    
    # Determine markers
    markers = args.marker or []
    if args.performance:
        markers.append("performance")
    if args.integration:
        markers.append("integration")
    if args.stress:
        markers.append("stress")
    if args.error_handling:
        markers.append("error_handling")
    
    # Run tests
    exit_code = runner.run_tests(
        test_paths=args.test_paths,
        markers=markers if markers else None,
        verbose=args.verbose,
        performance=args.performance,
        integration=args.integration,
        stress=args.stress,
    )
    
    # Save results
    runner.save_results()
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

