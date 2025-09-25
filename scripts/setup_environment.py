#!/usr/bin/env python3
"""Environment setup script for morphogenesis simulation.

This script sets up the complete development and runtime environment
for the morphogenesis project, including dependencies, configuration,
and validation of the installation.
"""

import os
import sys
import subprocess
import shutil
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import venv
import argparse


class EnvironmentSetup:
    """Handles environment setup and validation."""

    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)

        self.venv_path = self.project_root / ".venv"
        self.config_path = self.project_root / "config"
        self.data_path = self.project_root / "data"
        self.logs_path = self.project_root / "logs"
        self.results_path = self.project_root / "results"

        # Required Python version
        self.min_python_version = (3, 8)
        self.recommended_python_version = (3, 10)

        # Required system tools
        self.required_tools = ["git", "pip"]
        self.optional_tools = ["ffmpeg", "graphviz"]

        # Development vs production requirements
        self.base_requirements = [
            "numpy>=1.21.0",
            "scipy>=1.7.0",
            "matplotlib>=3.5.0",
            "pandas>=1.3.0",
            "asyncio",
            "aiofiles>=0.8.0",
            "pyyaml>=6.0",
            "click>=8.0.0",
            "tqdm>=4.62.0",
            "psutil>=5.8.0"
        ]

        self.optional_requirements = [
            "h5py>=3.0.0",  # For HDF5 serialization
            "numba>=0.56.0",  # For performance acceleration
            "jupyter>=1.0.0",  # For interactive analysis
            "plotly>=5.0.0",  # For interactive plots
            "dash>=2.0.0",  # For web dashboard
            "pytest>=6.0.0",  # For testing
            "pytest-asyncio>=0.18.0",  # For async testing
            "black>=21.0.0",  # For code formatting
            "flake8>=4.0.0",  # For linting
            "mypy>=0.910",  # For type checking
            "sphinx>=4.0.0",  # For documentation
        ]

    def setup_complete_environment(
        self,
        include_optional: bool = True,
        create_venv: bool = True,
        install_dev_tools: bool = True,
        force_reinstall: bool = False
    ) -> bool:
        """Set up complete environment.

        Args:
            include_optional: Install optional dependencies
            create_venv: Create virtual environment
            install_dev_tools: Install development tools
            force_reinstall: Force reinstallation of existing components

        Returns:
            True if setup successful, False otherwise
        """
        print("🔧 Setting up Morphogenesis simulation environment...")

        try:
            # Step 1: Validate system requirements
            print("\n📋 Step 1: Validating system requirements")
            if not self._validate_system_requirements():
                print("❌ System requirements validation failed")
                return False
            print("✅ System requirements validated")

            # Step 2: Create project directories
            print("\n📁 Step 2: Creating project directories")
            self._create_project_directories()
            print("✅ Project directories created")

            # Step 3: Set up virtual environment
            if create_venv:
                print("\n🐍 Step 3: Setting up virtual environment")
                if not self._setup_virtual_environment(force_reinstall):
                    print("❌ Virtual environment setup failed")
                    return False
                print("✅ Virtual environment setup complete")

            # Step 4: Install dependencies
            print("\n📦 Step 4: Installing dependencies")
            if not self._install_dependencies(include_optional, force_reinstall):
                print("❌ Dependency installation failed")
                return False
            print("✅ Dependencies installed")

            # Step 5: Set up configuration
            print("\n⚙️ Step 5: Setting up configuration")
            self._setup_configuration()
            print("✅ Configuration setup complete")

            # Step 6: Install development tools
            if install_dev_tools:
                print("\n🛠️ Step 6: Installing development tools")
                if not self._install_development_tools():
                    print("⚠️ Some development tools failed to install")
                else:
                    print("✅ Development tools installed")

            # Step 7: Validate installation
            print("\n✓ Step 7: Validating installation")
            if not self._validate_installation():
                print("❌ Installation validation failed")
                return False
            print("✅ Installation validated")

            print("\n🎉 Environment setup complete!")
            print(f"📍 Project root: {self.project_root}")
            print(f"🐍 Virtual environment: {self.venv_path}")
            print("\n💡 Next steps:")
            print("1. Activate virtual environment: source .venv/bin/activate")
            print("2. Run validation: python scripts/validate_installation.py")
            print("3. Start experimenting: python scripts/run_experiments.py --help")

            return True

        except Exception as e:
            print(f"❌ Environment setup failed: {e}")
            return False

    def _validate_system_requirements(self) -> bool:
        """Validate system requirements."""
        # Check Python version
        current_version = sys.version_info[:2]
        if current_version < self.min_python_version:
            print(f"❌ Python {self.min_python_version[0]}.{self.min_python_version[1]}+ required, "
                  f"found {current_version[0]}.{current_version[1]}")
            return False

        if current_version < self.recommended_python_version:
            print(f"⚠️ Python {self.recommended_python_version[0]}.{self.recommended_python_version[1]}+ "
                  f"recommended for best performance")

        # Check required tools
        missing_tools = []
        for tool in self.required_tools:
            if not shutil.which(tool):
                missing_tools.append(tool)

        if missing_tools:
            print(f"❌ Missing required tools: {', '.join(missing_tools)}")
            return False

        # Check optional tools
        missing_optional = []
        for tool in self.optional_tools:
            if not shutil.which(tool):
                missing_optional.append(tool)

        if missing_optional:
            print(f"⚠️ Missing optional tools (some features may be limited): {', '.join(missing_optional)}")

        return True

    def _create_project_directories(self) -> None:
        """Create necessary project directories."""
        directories = [
            self.config_path,
            self.data_path,
            self.logs_path,
            self.results_path,
            self.project_root / "docs" / "build",
            self.project_root / "tests" / "fixtures",
            self.project_root / "experiments" / "configs",
            self.project_root / "analysis" / "cache"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created: {directory}")

    def _setup_virtual_environment(self, force_reinstall: bool = False) -> bool:
        """Set up Python virtual environment."""
        if self.venv_path.exists() and not force_reinstall:
            print(f"🐍 Virtual environment already exists at {self.venv_path}")
            return True

        if force_reinstall and self.venv_path.exists():
            print("🗑️ Removing existing virtual environment")
            shutil.rmtree(self.venv_path)

        try:
            print(f"🐍 Creating virtual environment at {self.venv_path}")
            venv.create(self.venv_path, with_pip=True, upgrade_deps=True)

            # Upgrade pip in the virtual environment
            pip_path = self._get_venv_pip_path()
            subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)

            return True
        except Exception as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return False

    def _install_dependencies(self, include_optional: bool = True, force_reinstall: bool = False) -> bool:
        """Install Python dependencies."""
        pip_path = self._get_venv_pip_path()

        # Install base requirements
        print("📦 Installing base requirements...")
        for requirement in self.base_requirements:
            if not self._install_package(pip_path, requirement, force_reinstall):
                print(f"❌ Failed to install {requirement}")
                return False

        # Install optional requirements
        if include_optional:
            print("📦 Installing optional requirements...")
            for requirement in self.optional_requirements:
                if not self._install_package(pip_path, requirement, force_reinstall, optional=True):
                    print(f"⚠️ Failed to install optional package {requirement}")

        # Install project in development mode
        print("📦 Installing project in development mode...")
        try:
            subprocess.run([
                str(pip_path), "install", "-e", "."
            ], cwd=self.project_root, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install project: {e}")
            return False

        return True

    def _install_package(self, pip_path: Path, package: str, force_reinstall: bool = False, optional: bool = False) -> bool:
        """Install a single package."""
        cmd = [str(pip_path), "install"]
        if force_reinstall:
            cmd.append("--force-reinstall")
        cmd.append(package)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"✅ Installed: {package}")
                return True
            else:
                if not optional:
                    print(f"❌ Failed to install {package}: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            if not optional:
                print(f"❌ Timeout installing {package}")
            return False
        except Exception as e:
            if not optional:
                print(f"❌ Error installing {package}: {e}")
            return False

    def _setup_configuration(self) -> None:
        """Set up configuration files."""
        # Create default configuration
        default_config = {
            "simulation": {
                "world_width": 100,
                "world_height": 100,
                "cell_count": 100,
                "max_timesteps": 1000,
                "random_seed": 42
            },
            "visualization": {
                "plot_interval": 10,
                "save_plots": True,
                "animation": True
            },
            "performance": {
                "enable_profiling": False,
                "log_level": "INFO",
                "parallel_processing": True
            },
            "data": {
                "save_trajectory": True,
                "compression": True,
                "output_format": "hdf5"
            }
        }

        config_file = self.config_path / "default.yaml"
        if not config_file.exists():
            import yaml
            with open(config_file, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
            print(f"⚙️ Created default configuration: {config_file}")

        # Create logging configuration
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                },
                "detailed": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "standard",
                    "stream": "ext://sys.stdout"
                },
                "file": {
                    "class": "logging.FileHandler",
                    "level": "DEBUG",
                    "formatter": "detailed",
                    "filename": str(self.logs_path / "morphogenesis.log")
                }
            },
            "loggers": {
                "morphogenesis": {
                    "level": "DEBUG",
                    "handlers": ["console", "file"],
                    "propagate": False
                }
            },
            "root": {
                "level": "INFO",
                "handlers": ["console"]
            }
        }

        logging_config_file = self.config_path / "logging.yaml"
        if not logging_config_file.exists():
            import yaml
            with open(logging_config_file, 'w') as f:
                yaml.dump(logging_config, f, default_flow_style=False, indent=2)
            print(f"📝 Created logging configuration: {logging_config_file}")

    def _install_development_tools(self) -> bool:
        """Install development tools."""
        pip_path = self._get_venv_pip_path()

        dev_tools = [
            "pre-commit>=2.15.0",
            "commitizen>=2.20.0",
            "bandit>=1.7.0",
            "safety>=1.10.0"
        ]

        success = True
        for tool in dev_tools:
            if not self._install_package(pip_path, tool, optional=True):
                success = False

        # Set up pre-commit hooks
        try:
            precommit_path = self.venv_path / "bin" / "pre-commit"
            if precommit_path.exists():
                subprocess.run([str(precommit_path), "install"],
                             cwd=self.project_root, check=True, capture_output=True)
                print("🪝 Pre-commit hooks installed")
        except Exception as e:
            print(f"⚠️ Failed to set up pre-commit hooks: {e}")

        return success

    def _validate_installation(self) -> bool:
        """Validate the installation."""
        python_path = self._get_venv_python_path()

        # Test basic imports
        test_imports = [
            "numpy",
            "scipy",
            "matplotlib",
            "pandas",
            "asyncio",
            "aiofiles",
            "yaml",
            "click"
        ]

        print("🔍 Testing imports...")
        for module in test_imports:
            try:
                result = subprocess.run([
                    str(python_path), "-c", f"import {module}; print(f'{module} OK')"
                ], capture_output=True, text=True, timeout=10)

                if result.returncode == 0:
                    print(f"✅ {module}")
                else:
                    print(f"❌ {module}: {result.stderr}")
                    return False
            except Exception as e:
                print(f"❌ {module}: {e}")
                return False

        # Test project imports
        print("🔍 Testing project imports...")
        project_modules = [
            "core.data.types",
            "core.agents.cell_agent",
            "core.coordination.coordinator"
        ]

        for module in project_modules:
            try:
                result = subprocess.run([
                    str(python_path), "-c", f"from {module} import *; print(f'{module} OK')"
                ], capture_output=True, text=True, timeout=10, cwd=self.project_root)

                if result.returncode == 0:
                    print(f"✅ {module}")
                else:
                    print(f"❌ {module}: {result.stderr}")
                    return False
            except Exception as e:
                print(f"❌ {module}: {e}")
                return False

        return True

    def _get_venv_python_path(self) -> Path:
        """Get path to Python executable in virtual environment."""
        if sys.platform == "win32":
            return self.venv_path / "Scripts" / "python.exe"
        else:
            return self.venv_path / "bin" / "python"

    def _get_venv_pip_path(self) -> Path:
        """Get path to pip executable in virtual environment."""
        if sys.platform == "win32":
            return self.venv_path / "Scripts" / "pip.exe"
        else:
            return self.venv_path / "bin" / "pip"

    def cleanup_environment(self) -> None:
        """Clean up environment (remove virtual environment and generated files)."""
        print("🧹 Cleaning up environment...")

        if self.venv_path.exists():
            print(f"🗑️ Removing virtual environment: {self.venv_path}")
            shutil.rmtree(self.venv_path)

        # Clean up generated directories
        cleanup_paths = [
            self.project_root / "__pycache__",
            self.project_root / ".pytest_cache",
            self.project_root / ".mypy_cache",
            self.project_root / "build",
            self.project_root / "dist",
            self.project_root / "*.egg-info"
        ]

        for path in cleanup_paths:
            if path.exists():
                if path.is_file():
                    path.unlink()
                else:
                    shutil.rmtree(path, ignore_errors=True)
                print(f"🗑️ Removed: {path}")

        print("✅ Environment cleanup complete")

    def print_system_info(self) -> None:
        """Print system information."""
        print("🖥️ System Information:")
        print(f"  Python: {sys.version}")
        print(f"  Platform: {sys.platform}")
        print(f"  Architecture: {os.uname() if hasattr(os, 'uname') else 'Windows'}")
        print(f"  Project root: {self.project_root}")
        print(f"  Virtual env: {self.venv_path}")

        # Check available tools
        print("\n🛠️ Available tools:")
        all_tools = self.required_tools + self.optional_tools
        for tool in all_tools:
            status = "✅" if shutil.which(tool) else "❌"
            print(f"  {tool}: {status}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Set up morphogenesis simulation environment")

    parser.add_argument("--no-optional", action="store_true",
                       help="Skip optional dependencies")
    parser.add_argument("--no-venv", action="store_true",
                       help="Skip virtual environment creation")
    parser.add_argument("--no-dev-tools", action="store_true",
                       help="Skip development tools installation")
    parser.add_argument("--force", action="store_true",
                       help="Force reinstall existing components")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up environment instead of setting up")
    parser.add_argument("--info", action="store_true",
                       help="Print system information")
    parser.add_argument("--project-root", type=str,
                       help="Project root directory")

    args = parser.parse_args()

    # Initialize setup manager
    setup = EnvironmentSetup(args.project_root)

    if args.info:
        setup.print_system_info()
        return

    if args.cleanup:
        setup.cleanup_environment()
        return

    # Set up environment
    success = setup.setup_complete_environment(
        include_optional=not args.no_optional,
        create_venv=not args.no_venv,
        install_dev_tools=not args.no_dev_tools,
        force_reinstall=args.force
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()