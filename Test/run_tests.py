"""
Test Runner Script
Runs all tests in the Test directory with detailed reporting.
"""

import subprocess
import sys
import os

def main():
    """Run all tests with pytest."""
    # Get the test directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(test_dir)
    
    # Add project root to path
    sys.path.insert(0, project_root)
    
    # Build pytest command
    pytest_args = [
        sys.executable, "-m", "pytest",
        test_dir,
        "-v",                    # Verbose output
        "--tb=short",            # Short traceback
        "-x",                    # Stop on first failure (remove for full run)
        "--color=yes",           # Colored output
        "-W", "ignore::DeprecationWarning",  # Ignore deprecation warnings
    ]
    
    # Check if specific test file is requested
    if len(sys.argv) > 1:
        pytest_args = [
            sys.executable, "-m", "pytest",
            *sys.argv[1:],
            "-v",
            "--tb=short",
            "--color=yes",
        ]
    
    print("=" * 60)
    print("Running Payment Default Prediction Model Tests")
    print("=" * 60)
    print(f"Project Root: {project_root}")
    print(f"Test Directory: {test_dir}")
    print("=" * 60)
    print()
    
    # Run pytest
    result = subprocess.run(pytest_args, cwd=project_root)
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
