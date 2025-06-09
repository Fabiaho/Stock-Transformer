#!/usr/bin/env python3
"""
Setup script for stock transformer predictor.
Handles environment setup, dependency installation, and initial configuration.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, check=True, capture_output=False):
    """Run a shell command."""
    print(f"Running: {cmd}")
    if capture_output:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if check and result.returncode != 0:
            print(f"Command failed: {result.stderr}")
            sys.exit(1)
        return result.stdout.strip()
    else:
        result = subprocess.run(cmd, shell=True)
        if check and result.returncode != 0:
            print(f"Command failed with exit code: {result.returncode}")
            sys.exit(1)


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major != 3 or version.minor < 10:
        print(f"Error: Python 3.10+ required, found {version.major}.{version.minor}")
        sys.exit(1)
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")


def check_poetry():
    """Check if Poetry is installed."""
    try:
        version = run_command("poetry --version", capture_output=True)
        print(f"✓ {version}")
        return True
    except:
        print("✗ Poetry not found")
        return False


def install_poetry():
    """Install Poetry."""
    print("Installing Poetry...")
    run_command("curl -sSL https://install.python-poetry.org | python3 -")
    
    # Add to PATH for current session
    home = Path.home()
    poetry_bin = home / ".local" / "bin"
    if poetry_bin.exists():
        os.environ["PATH"] = f"{poetry_bin}:{os.environ['PATH']}"
    
    print("✓ Poetry installed")


def setup_poetry_environment():
    """Setup Poetry virtual environment and install dependencies."""
    print("Configuring Poetry...")
    run_command("poetry config virtualenvs.in-project true")
    
    print("Installing dependencies...")
    run_command("poetry install")
    
    # Add colorlog dependency that's missing from pyproject.toml
    print("Adding additional dependencies...")
    run_command("poetry add colorlog psutil")
    
    print("✓ Dependencies installed")


def create_directories():
    """Create necessary project directories."""
    directories = [
        "data/raw/price_cache",
        "data/processed", 
        "data/external",
        "models/checkpoints",
        "models/experiments", 
        "models/final",
        "results/figures",
        "results/reports",
        "results/logs",
        "results/experiments"
    ]
    
    print("Creating project directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✓ Project directories created")


def create_env_file():
    """Create example .env file."""
    env_content = """# Environment variables for stock transformer predictor

# API Keys (add your own)
# ALPHA_VANTAGE_API_KEY=your_key_here
# FINNHUB_API_KEY=your_key_here
# NEWS_API_KEY=your_key_here

# Weights & Biases (optional)
# WANDB_API_KEY=your_key_here
# WANDB_PROJECT=stock-transformer

# Data settings
DATA_CACHE_DIR=data/raw/price_cache
RESULTS_DIR=results

# Model settings
DEFAULT_MODEL_DIR=models
DEFAULT_LOG_DIR=results/logs

# Training settings
DEFAULT_BATCH_SIZE=32
DEFAULT_LEARNING_RATE=1e-4
DEFAULT_MAX_EPOCHS=100
"""
    
    env_path = Path(".env.example")
    if not env_path.exists():
        with open(env_path, "w") as f:
            f.write(env_content)
        print("✓ Created .env.example file")
    else:
        print("✓ .env.example already exists")


def setup_git_hooks():
    """Setup git hooks for development."""
    if Path(".git").exists():
        print("Setting up git hooks...")
        run_command("poetry run pre-commit install", check=False)
        print("✓ Git hooks configured")
    else:
        print("✓ Not a git repository, skipping git hooks")


def run_validation():
    """Run validation to ensure everything is working."""
    print("Running validation tests...")
    
    try:
        # Test imports
        print("Testing imports...")
        test_script = """
import sys
sys.path.append('.')
try:
    from src.data.collectors.price_collector import PriceCollector
    from src.data.processors.technical_indicators import TechnicalIndicators
    from src.data.dataset import StockSequenceDataset
    from src.data.datamodule import StockDataModule
    from src.models.transformer import StockTransformer
    from src.models.lightning_module import StockTransformerLightning
    print('✓ All imports successful')
except ImportError as e:
    print(f'✗ Import failed: {e}')
    sys.exit(1)
"""
        
        result = run_command(f"poetry run python -c \"{test_script}\"", capture_output=True)
        print(result)
        
        # Test basic functionality
        print("Testing basic data collection...")
        test_data_script = """
import sys
sys.path.append('.')
from src.data.collectors.price_collector import PriceCollector
import tempfile
try:
    collector = PriceCollector()
    data = collector.fetch_stock_data('AAPL', '2023-01-01', '2023-01-10')
    print(f'✓ Data collection test passed: {len(data)} days')
except Exception as e:
    print(f'✗ Data collection test failed: {e}')
"""
        
        result = run_command(f"poetry run python -c \"{test_data_script}\"", capture_output=True)
        print(result)
        
        print("✓ Validation completed successfully")
        
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False
        
    return True


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*50)
    print("🎉 Setup completed successfully!")
    print("="*50)
    print("\nNext steps:")
    print("1. Activate the virtual environment:")
    print("   poetry shell")
    print("\n2. Run a quick training test:")
    print("   python scripts/train_model.py --fast-dev-run")
    print("\n3. Start a full training run:")
    print("   python scripts/train_model.py --experiment-name my_first_experiment")
    print("\n4. Monitor training progress:")
    print("   tensorboard --logdir results/logs")
    print("\n5. Check the example script:")
    print("   bash scripts/run_training_example.sh")
    print("\nConfiguration files are in the 'config/' directory.")
    print("Check README.md for detailed usage instructions.")


def main():
    parser = argparse.ArgumentParser(description="Setup Stock Transformer Predictor")
    parser.add_argument('--skip-poetry', action='store_true', 
                       help='Skip Poetry installation')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip validation tests')
    parser.add_argument('--dev-setup', action='store_true',
                       help='Setup for development (includes git hooks)')
    
    args = parser.parse_args()
    
    print("Stock Transformer Predictor Setup")
    print("=" * 40)
    
    # Check Python version
    check_python_version()
    
    # Handle Poetry
    if not args.skip_poetry:
        if not check_poetry():
            install_poetry()
        
        setup_poetry_environment()
    
    # Create project structure
    create_directories()
    create_env_file()
    
    # Development setup
    if args.dev_setup:
        setup_git_hooks()
    
    # Validation
    if not args.skip_validation:
        validation_success = run_validation()
        if not validation_success:
            print("Setup completed with warnings. Some validation tests failed.")
            return
    
    print_next_steps()


if __name__ == "__main__":
    main()