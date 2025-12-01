#!/usr/bin/env python
"""Test runner script for CI/CD pipeline"""
import subprocess
import sys
import os

def run_command(command, env=None):
    """Run a shell command"""
    env_vars = os.environ.copy()
    if env:
        env_vars.update(env)
    
    result = subprocess.run(
        command,
        shell=True,
        env=env_vars,
        capture_output=True,
        text=True
    )
    
    print(f"Command: {command}")
    print(f"Return code: {result.returncode}")
    if result.stdout:
        print(f"Output:\n{result.stdout}")
    if result.stderr:
        print(f"Errors:\n{result.stderr}")
    
    return result.returncode

def main():
    """Run all tests"""
    env = {
        'CAPSTONE_TEST': os.getenv('CAPSTONE_TEST', ''),
        'PYTHONPATH': os.getcwd()
    }
    
    test_files = [
        'test/test_preprocessing.py',
        'test/test_feature_engineering.py', 
        'test/test_model_evaluation.py',
        'test/test_model_registry.py'
    ]
    
    all_passed = True
    
    # Run DVC pipeline first
    print("🚀 Running DVC pipeline...")
    dvc_result = run_command("dvc repro", env)
    if dvc_result != 0:
        print("❌ DVC pipeline failed")
        all_passed = False
    
    # Run each test file
    for test_file in test_files:
        print(f"\n🧪 Running {test_file}...")
        cmd = f"python -m pytest {test_file} -v --tb=short"
        result = run_command(cmd, env)
        if result != 0:
            all_passed = False
    
    # Run comprehensive test with coverage
    print("\n📊 Running comprehensive test with coverage...")
    coverage_cmd = "python -m pytest tests/ -v --cov=src --cov-report=xml"
    run_command(coverage_cmd, env)
    
    # Promote model if all tests passed
    if all_passed:
        print("\n✅ All tests passed! Promoting model to production...")
        promote_result = run_command("python scripts/promote_model.py", env)
        if promote_result != 0:
            print("❌ Model promotion failed")
            sys.exit(1)
        print("🎉 Model successfully promoted to production!")
    else:
        print("\n❌ Some tests failed. Model will not be promoted.")
        sys.exit(1)

if __name__ == "__main__":
    main()