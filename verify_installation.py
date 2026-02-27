"""
Verification script to check if all dependencies are installed correctly
"""

import sys

def check_imports():
    """Check if all required packages can be imported"""
    print("Checking dependencies...")
    
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'plotly': 'plotly',
        'streamlit': 'streamlit',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm'
    }
    
    failed_imports = []
    
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - NOT INSTALLED")
            failed_imports.append(package_name)
    
    return failed_imports

def check_modules():
    """Check if project modules can be imported"""
    print("\nChecking project modules...")
    
    modules = [
        'src.drift_detector',
        'src.drift_detector.detector',
        'src.drift_detector.statistical_tests',
        'src.visualization',
        'src.visualization.plotter',
        'src.mitigation',
        'src.mitigation.strategies',
        'src.mitigation.adaptive_training'
    ]
    
    failed_modules = []
    
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module} - {str(e)}")
            failed_modules.append(module)
    
    return failed_modules

def main():
    """Main verification function"""
    print("=" * 60)
    print("Adaptive Drift Intelligence - Installation Verification")
    print("=" * 60)
    print(f"Python version: {sys.version}\n")
    
    # Check dependencies
    failed_packages = check_imports()
    
    # Check modules
    failed_modules = check_modules()
    
    # Summary
    print("\n" + "=" * 60)
    if not failed_packages and not failed_modules:
        print("✓ All checks passed! Installation is complete.")
        print("\nYou can now:")
        print("  - Run: streamlit run app.py")
        print("  - Run: python example_usage.py")
        return 0
    else:
        print("✗ Some checks failed.")
        if failed_packages:
            print(f"\nMissing packages: {', '.join(failed_packages)}")
            print("Install with: pip install -r requirements.txt")
        if failed_modules:
            print(f"\nModule import errors: {', '.join(failed_modules)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
