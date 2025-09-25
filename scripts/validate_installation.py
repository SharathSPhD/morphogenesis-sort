#!/usr/bin/env python3
"""Installation validation script for morphogenesis simulation.

This script thoroughly validates the installation, dependencies, and
functionality of the morphogenesis simulation package.
"""

import sys
import subprocess
import importlib
import asyncio
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class InstallationValidator:
    """Comprehensive installation validation system."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.logger = self._setup_logging()
        self.validation_results = {
            "python_environment": {},
            "system_requirements": {},
            "dependencies": {},
            "project_modules": {},
            "functionality_tests": {},
            "performance_tests": {},
            "integration_tests": {}
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for validation."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        return logging.getLogger(__name__)

    async def validate_complete_installation(self, verbose: bool = False) -> Tuple[bool, Dict[str, Any]]:
        """Run complete installation validation.

        Args:
            verbose: Enable verbose output

        Returns:
            Tuple of (success, validation_results)
        """
        self.logger.info("🔍 Starting complete installation validation...")

        try:
            # Step 1: Python environment validation
            self.logger.info("📋 Step 1: Validating Python environment")
            python_ok = await self._validate_python_environment()
            self._log_result("Python environment", python_ok)

            # Step 2: System requirements validation
            self.logger.info("🖥️ Step 2: Validating system requirements")
            system_ok = await self._validate_system_requirements()
            self._log_result("System requirements", system_ok)

            # Step 3: Dependencies validation
            self.logger.info("📦 Step 3: Validating dependencies")
            deps_ok = await self._validate_dependencies()
            self._log_result("Dependencies", deps_ok)

            # Step 4: Project modules validation
            self.logger.info("🐍 Step 4: Validating project modules")
            modules_ok = await self._validate_project_modules()
            self._log_result("Project modules", modules_ok)

            # Step 5: Functionality tests
            self.logger.info("⚙️ Step 5: Running functionality tests")
            func_ok = await self._run_functionality_tests()
            self._log_result("Functionality tests", func_ok)

            # Step 6: Performance tests
            self.logger.info("⚡ Step 6: Running performance tests")
            perf_ok = await self._run_performance_tests()
            self._log_result("Performance tests", perf_ok)

            # Step 7: Integration tests
            self.logger.info("🔧 Step 7: Running integration tests")
            integ_ok = await self._run_integration_tests()
            self._log_result("Integration tests", integ_ok)

            # Overall validation result
            all_results = [python_ok, system_ok, deps_ok, modules_ok, func_ok, perf_ok, integ_ok]
            overall_success = all(all_results)

            self.validation_results["overall_success"] = overall_success
            self.validation_results["validation_timestamp"] = datetime.now().isoformat()

            # Print summary
            self._print_validation_summary(overall_success)

            return overall_success, self.validation_results

        except Exception as e:
            self.logger.error(f"❌ Validation failed with exception: {e}")
            if verbose:
                traceback.print_exc()
            return False, self.validation_results

    async def _validate_python_environment(self) -> bool:
        """Validate Python environment."""
        try:
            # Check Python version
            version_info = sys.version_info
            required_version = (3, 8)
            recommended_version = (3, 10)

            version_ok = version_info >= required_version
            self.validation_results["python_environment"]["version"] = {
                "current": f"{version_info.major}.{version_info.minor}.{version_info.micro}",
                "required": f"{required_version[0]}.{required_version[1]}+",
                "recommended": f"{recommended_version[0]}.{recommended_version[1]}+",
                "meets_requirement": version_ok,
                "meets_recommendation": version_info >= recommended_version
            }

            if not version_ok:
                self.logger.error(f"❌ Python {required_version[0]}.{required_version[1]}+ required, "
                                f"found {version_info.major}.{version_info.minor}")
                return False

            if version_info < recommended_version:
                self.logger.warning(f"⚠️ Python {recommended_version[0]}.{recommended_version[1]}+ "
                                  f"recommended for best performance")

            # Check virtual environment
            in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
            self.validation_results["python_environment"]["virtual_environment"] = {
                "active": in_venv,
                "path": sys.prefix if in_venv else None
            }

            if not in_venv:
                self.logger.warning("⚠️ Not running in virtual environment")

            # Check pip
            try:
                import pip
                pip_version = pip.__version__
                self.validation_results["python_environment"]["pip"] = {
                    "available": True,
                    "version": pip_version
                }
            except ImportError:
                self.logger.error("❌ pip not available")
                self.validation_results["python_environment"]["pip"] = {"available": False}
                return False

            return True

        except Exception as e:
            self.logger.error(f"❌ Python environment validation failed: {e}")
            return False

    async def _validate_system_requirements(self) -> bool:
        """Validate system requirements."""
        try:
            import platform
            import shutil

            # System information
            system_info = {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "python_implementation": platform.python_implementation()
            }
            self.validation_results["system_requirements"]["system_info"] = system_info

            # Required tools
            required_tools = ["git", "pip"]
            optional_tools = ["ffmpeg", "graphviz", "dot"]

            tools_status = {}
            all_required_found = True

            for tool in required_tools:
                tool_path = shutil.which(tool)
                is_available = tool_path is not None
                tools_status[tool] = {
                    "required": True,
                    "available": is_available,
                    "path": tool_path
                }

                if not is_available:
                    self.logger.error(f"❌ Required tool not found: {tool}")
                    all_required_found = False

            for tool in optional_tools:
                tool_path = shutil.which(tool)
                is_available = tool_path is not None
                tools_status[tool] = {
                    "required": False,
                    "available": is_available,
                    "path": tool_path
                }

                if not is_available:
                    self.logger.warning(f"⚠️ Optional tool not found: {tool}")

            self.validation_results["system_requirements"]["tools"] = tools_status

            # Memory check
            try:
                import psutil
                memory_info = psutil.virtual_memory()
                memory_gb = memory_info.total / (1024**3)

                self.validation_results["system_requirements"]["memory"] = {
                    "total_gb": round(memory_gb, 2),
                    "available_gb": round(memory_info.available / (1024**3), 2),
                    "sufficient": memory_gb >= 4.0  # Minimum 4GB recommended
                }

                if memory_gb < 2.0:
                    self.logger.warning(f"⚠️ Low system memory: {memory_gb:.1f}GB (4GB+ recommended)")
                elif memory_gb < 4.0:
                    self.logger.warning(f"⚠️ Limited system memory: {memory_gb:.1f}GB (4GB+ recommended)")

            except ImportError:
                self.logger.warning("⚠️ Cannot check system memory (psutil not available)")
                self.validation_results["system_requirements"]["memory"] = {"available": False}

            # Disk space check
            try:
                import shutil
                disk_usage = shutil.disk_usage(self.project_root)
                free_gb = disk_usage.free / (1024**3)

                self.validation_results["system_requirements"]["disk"] = {
                    "free_gb": round(free_gb, 2),
                    "sufficient": free_gb >= 1.0  # Minimum 1GB free space
                }

                if free_gb < 0.5:
                    self.logger.warning(f"⚠️ Low disk space: {free_gb:.1f}GB free")

            except Exception:
                self.logger.warning("⚠️ Cannot check disk space")
                self.validation_results["system_requirements"]["disk"] = {"available": False}

            return all_required_found

        except Exception as e:
            self.logger.error(f"❌ System requirements validation failed: {e}")
            return False

    async def _validate_dependencies(self) -> bool:
        """Validate package dependencies."""
        dependencies = {
            "required": [
                "numpy", "scipy", "matplotlib", "pandas", "asyncio",
                "aiofiles", "yaml", "click", "tqdm", "psutil"
            ],
            "optional": [
                "h5py", "numba", "jupyter", "plotly", "dash",
                "pytest", "pytest-asyncio", "black", "flake8", "mypy"
            ]
        }

        all_required_ok = True
        deps_status = {}

        # Test required dependencies
        for dep in dependencies["required"]:
            try:
                if dep == "yaml":
                    import yaml
                elif dep == "asyncio":
                    import asyncio
                else:
                    importlib.import_module(dep)

                # Get version if possible
                try:
                    module = importlib.import_module(dep)
                    version = getattr(module, '__version__', 'unknown')
                except:
                    version = 'unknown'

                deps_status[dep] = {
                    "required": True,
                    "available": True,
                    "version": version,
                    "import_successful": True
                }

            except ImportError as e:
                self.logger.error(f"❌ Required dependency missing: {dep}")
                deps_status[dep] = {
                    "required": True,
                    "available": False,
                    "error": str(e),
                    "import_successful": False
                }
                all_required_ok = False

        # Test optional dependencies
        for dep in dependencies["optional"]:
            try:
                importlib.import_module(dep)

                try:
                    module = importlib.import_module(dep)
                    version = getattr(module, '__version__', 'unknown')
                except:
                    version = 'unknown'

                deps_status[dep] = {
                    "required": False,
                    "available": True,
                    "version": version,
                    "import_successful": True
                }

            except ImportError:
                self.logger.info(f"ℹ️ Optional dependency not available: {dep}")
                deps_status[dep] = {
                    "required": False,
                    "available": False,
                    "import_successful": False
                }

        self.validation_results["dependencies"] = deps_status
        return all_required_ok

    async def _validate_project_modules(self) -> bool:
        """Validate project module imports."""
        project_modules = [
            "core.data.types",
            "core.data.state",
            "core.data.actions",
            "core.data.serialization",
            "core.agents.cell_agent",
            "core.agents.behaviors.config",
            "core.agents.behaviors.sorting_cell",
            "core.agents.behaviors.morphogen_cell",
            "core.agents.behaviors.adaptive_cell",
            "core.coordination.coordinator",
            "core.coordination.scheduler",
            "core.coordination.spatial_index",
            "core.coordination.conflict_resolver",
            "core.metrics.collector",
            "core.metrics.counters",
            "analysis.scientific_analysis"
        ]

        modules_status = {}
        all_modules_ok = True

        for module_name in project_modules:
            try:
                module = importlib.import_module(module_name)

                # Try to access key classes/functions
                module_content = dir(module)
                has_content = len(module_content) > 5  # Basic content check

                modules_status[module_name] = {
                    "import_successful": True,
                    "has_content": has_content,
                    "content_count": len(module_content)
                }

            except ImportError as e:
                self.logger.error(f"❌ Project module import failed: {module_name}")
                self.logger.error(f"   Error: {e}")
                modules_status[module_name] = {
                    "import_successful": False,
                    "error": str(e)
                }
                all_modules_ok = False

            except Exception as e:
                self.logger.warning(f"⚠️ Project module issue: {module_name} - {e}")
                modules_status[module_name] = {
                    "import_successful": True,
                    "has_issue": True,
                    "issue": str(e)
                }

        self.validation_results["project_modules"] = modules_status
        return all_modules_ok

    async def _run_functionality_tests(self) -> bool:
        """Run basic functionality tests."""
        tests_passed = 0
        total_tests = 0

        # Test 1: Data types creation
        total_tests += 1
        try:
            from core.data.types import Position, CellParameters, WorldParameters, create_cell_id

            # Test Position
            pos = Position(1.0, 2.0)
            assert pos.x == 1.0 and pos.y == 2.0
            assert pos.distance_to(Position(0.0, 0.0)) > 0

            # Test CellParameters
            params = CellParameters()
            assert params.validate()

            # Test WorldParameters
            world_params = WorldParameters()
            assert world_params.validate()

            # Test cell ID creation
            cell_id = create_cell_id(42)
            assert int(cell_id) == 42

            tests_passed += 1
            self.validation_results["functionality_tests"]["data_types"] = {"passed": True}

        except Exception as e:
            self.logger.error(f"❌ Data types test failed: {e}")
            self.validation_results["functionality_tests"]["data_types"] = {
                "passed": False, "error": str(e)
            }

        # Test 2: Cell agent creation
        total_tests += 1
        try:
            from core.data.types import CellID, CellType, CellState
            from core.data.state import CellData
            from core.agents.cell_agent import CellBehaviorConfig, StandardCellAgent

            # Create test cell data
            cell_id = CellID(1)
            cell_data = CellData(
                cell_id=cell_id,
                position=Position(0.0, 0.0),
                cell_type=CellType.STANDARD,
                cell_state=CellState.ACTIVE,
                sort_value=1.0,
                age=0
            )

            # Create behavior config
            behavior_config = CellBehaviorConfig()

            # Create cell parameters
            cell_params = CellParameters()

            # Create agent (but don't run it)
            agent = StandardCellAgent(cell_id, cell_data, behavior_config, cell_params)
            assert agent.cell_id == cell_id

            tests_passed += 1
            self.validation_results["functionality_tests"]["cell_agent"] = {"passed": True}

        except Exception as e:
            self.logger.error(f"❌ Cell agent test failed: {e}")
            self.validation_results["functionality_tests"]["cell_agent"] = {
                "passed": False, "error": str(e)
            }

        # Test 3: Serialization
        total_tests += 1
        try:
            from core.data.serialization import AsyncDataSerializer
            import tempfile
            import os

            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                serializer = AsyncDataSerializer(temp_dir)

                # Test data
                test_data = {"test": "value", "number": 42}

                # Serialize and deserialize
                metadata = await serializer.serialize_data(test_data, "test")
                loaded_data = await serializer.deserialize_data("test", dict)

                assert loaded_data["test"] == "value"
                assert loaded_data["number"] == 42

            tests_passed += 1
            self.validation_results["functionality_tests"]["serialization"] = {"passed": True}

        except Exception as e:
            self.logger.error(f"❌ Serialization test failed: {e}")
            self.validation_results["functionality_tests"]["serialization"] = {
                "passed": False, "error": str(e)
            }

        success_rate = tests_passed / total_tests if total_tests > 0 else 0
        self.validation_results["functionality_tests"]["summary"] = {
            "tests_passed": tests_passed,
            "total_tests": total_tests,
            "success_rate": success_rate
        }

        return success_rate >= 0.8  # 80% pass rate required

    async def _run_performance_tests(self) -> bool:
        """Run basic performance tests."""
        try:
            import time
            import numpy as np

            performance_results = {}

            # Test 1: Numpy operations
            start_time = time.time()
            large_array = np.random.random((1000, 1000))
            result = np.dot(large_array, large_array.T)
            numpy_time = time.time() - start_time

            performance_results["numpy_matrix_multiply"] = {
                "time_seconds": numpy_time,
                "acceptable": numpy_time < 5.0  # Should complete in under 5 seconds
            }

            # Test 2: Async operations
            start_time = time.time()
            await asyncio.sleep(0.1)  # Simulate async operation
            async_time = time.time() - start_time

            performance_results["async_operations"] = {
                "time_seconds": async_time,
                "acceptable": 0.08 <= async_time <= 0.15  # Should be close to 0.1 seconds
            }

            # Test 3: Memory usage (if psutil available)
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)

                performance_results["memory_usage"] = {
                    "memory_mb": memory_mb,
                    "acceptable": memory_mb < 500  # Should use less than 500MB
                }
            except ImportError:
                performance_results["memory_usage"] = {"available": False}

            self.validation_results["performance_tests"] = performance_results

            # Check if all available tests passed
            all_acceptable = all(
                test_result.get("acceptable", True)
                for test_result in performance_results.values()
                if "acceptable" in test_result
            )

            return all_acceptable

        except Exception as e:
            self.logger.error(f"❌ Performance tests failed: {e}")
            self.validation_results["performance_tests"] = {
                "failed": True, "error": str(e)
            }
            return False

    async def _run_integration_tests(self) -> bool:
        """Run integration tests."""
        try:
            # Test 1: Configuration loading
            config_test_passed = await self._test_configuration_loading()

            # Test 2: Experiment setup (without running)
            experiment_test_passed = await self._test_experiment_setup()

            # Test 3: Analysis components
            analysis_test_passed = await self._test_analysis_components()

            integration_results = {
                "configuration_loading": {"passed": config_test_passed},
                "experiment_setup": {"passed": experiment_test_passed},
                "analysis_components": {"passed": analysis_test_passed}
            }

            self.validation_results["integration_tests"] = integration_results

            return all([config_test_passed, experiment_test_passed, analysis_test_passed])

        except Exception as e:
            self.logger.error(f"❌ Integration tests failed: {e}")
            self.validation_results["integration_tests"] = {
                "failed": True, "error": str(e)
            }
            return False

    async def _test_configuration_loading(self) -> bool:
        """Test configuration loading."""
        try:
            from core.agents.behaviors.config import (
                SortingConfig, MorphogenConfig, AdaptiveConfig
            )

            # Test default configurations
            sorting_config = SortingConfig()
            assert sorting_config.validate()

            morphogen_config = MorphogenConfig()
            assert morphogen_config.validate()

            adaptive_config = AdaptiveConfig()
            assert adaptive_config.validate()

            return True
        except Exception as e:
            self.logger.error(f"Configuration loading test failed: {e}")
            return False

    async def _test_experiment_setup(self) -> bool:
        """Test experiment setup without running."""
        try:
            from experiments.experiment_config import ExperimentConfig

            # Create test experiment configuration
            config = ExperimentConfig(
                name="test_experiment",
                description="Test configuration",
                cell_count=10,
                world_width=50,
                world_height=50,
                max_timesteps=10,
                random_seed=42
            )

            assert config.cell_count == 10
            assert config.name == "test_experiment"

            return True
        except Exception as e:
            self.logger.error(f"Experiment setup test failed: {e}")
            return False

    async def _test_analysis_components(self) -> bool:
        """Test analysis components."""
        try:
            from analysis.scientific_analysis import ScientificAnalyzer

            # Create analyzer instance
            analyzer = ScientificAnalyzer()

            # Test basic functionality (without requiring actual data)
            assert hasattr(analyzer, 'analyze_trajectory')
            assert hasattr(analyzer, 'compute_metrics')

            return True
        except Exception as e:
            self.logger.error(f"Analysis components test failed: {e}")
            return False

    def _log_result(self, test_name: str, success: bool) -> None:
        """Log test result."""
        if success:
            self.logger.info(f"✅ {test_name}: PASSED")
        else:
            self.logger.error(f"❌ {test_name}: FAILED")

    def _print_validation_summary(self, overall_success: bool) -> None:
        """Print validation summary."""
        print("\n" + "="*60)
        print("🔍 MORPHOGENESIS INSTALLATION VALIDATION SUMMARY")
        print("="*60)

        status_emoji = "✅" if overall_success else "❌"
        status_text = "PASSED" if overall_success else "FAILED"

        print(f"\n{status_emoji} Overall Status: {status_text}")

        # Print detailed results
        categories = [
            ("Python Environment", "python_environment"),
            ("System Requirements", "system_requirements"),
            ("Dependencies", "dependencies"),
            ("Project Modules", "project_modules"),
            ("Functionality Tests", "functionality_tests"),
            ("Performance Tests", "performance_tests"),
            ("Integration Tests", "integration_tests")
        ]

        for category_name, category_key in categories:
            if category_key in self.validation_results:
                category_data = self.validation_results[category_key]

                # Determine category status
                if category_key == "functionality_tests":
                    category_success = category_data.get("summary", {}).get("success_rate", 0) >= 0.8
                elif isinstance(category_data, dict) and "failed" in category_data:
                    category_success = not category_data["failed"]
                else:
                    # For other categories, check if there are any failures
                    category_success = not any(
                        (isinstance(v, dict) and not v.get("available", True) and v.get("required", False))
                        or (isinstance(v, dict) and not v.get("import_successful", True))
                        for v in category_data.values()
                        if isinstance(v, dict)
                    )

                emoji = "✅" if category_success else "❌"
                print(f"  {emoji} {category_name}")

        print("\n" + "="*60)

        if overall_success:
            print("🎉 Installation validation completed successfully!")
            print("✨ Your morphogenesis simulation environment is ready to use!")
        else:
            print("⚠️ Installation validation found issues.")
            print("🔧 Please address the failed components and run validation again.")

        print("\n💡 Next steps:")
        if overall_success:
            print("  1. Run example experiment: python scripts/run_experiments.py sorting")
            print("  2. Explore configuration: python scripts/run_experiments.py --create-example sorting")
            print("  3. Check documentation in docs/ directory")
        else:
            print("  1. Fix the issues shown above")
            print("  2. Run setup: python scripts/setup_environment.py")
            print("  3. Re-run validation: python scripts/validate_installation.py")

    async def run_specific_validation(self, validation_type: str) -> bool:
        """Run specific validation test."""
        validation_methods = {
            "python": self._validate_python_environment,
            "system": self._validate_system_requirements,
            "dependencies": self._validate_dependencies,
            "modules": self._validate_project_modules,
            "functionality": self._run_functionality_tests,
            "performance": self._run_performance_tests,
            "integration": self._run_integration_tests
        }

        if validation_type not in validation_methods:
            self.logger.error(f"Unknown validation type: {validation_type}")
            return False

        self.logger.info(f"Running {validation_type} validation...")
        method = validation_methods[validation_type]
        result = await method()
        self._log_result(f"{validation_type} validation", result)

        return result

    def save_validation_report(self, output_path: Optional[Path] = None) -> Path:
        """Save validation report to file."""
        if output_path is None:
            output_path = self.project_root / "validation_report.json"

        import json
        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)

        self.logger.info(f"Validation report saved: {output_path}")
        return output_path


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate morphogenesis installation")
    parser.add_argument("--specific", type=str,
                       choices=["python", "system", "dependencies", "modules",
                               "functionality", "performance", "integration"],
                       help="Run specific validation test")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--save-report", type=str,
                       help="Save validation report to file")

    args = parser.parse_args()

    # Initialize validator
    validator = InstallationValidator()

    # Run specific validation if requested
    if args.specific:
        success = await validator.run_specific_validation(args.specific)
        sys.exit(0 if success else 1)

    # Run complete validation
    success, results = await validator.validate_complete_installation(args.verbose)

    # Save report if requested
    if args.save_report:
        report_path = validator.save_validation_report(Path(args.save_report))

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())