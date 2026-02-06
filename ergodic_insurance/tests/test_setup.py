"""Test to verify project setup is working correctly."""

from pathlib import Path

import pytest


def test_project_structure_exists():
    """Test that the basic project structure exists."""
    project_root = Path(__file__).parent.parent

    assert project_root.exists()
    # No src directory in this project structure
    assert (project_root / "tests").exists()
    # notebooks directory is at the root level, not in ergodic_insurance
    assert (project_root / "data" / "parameters").exists()


def test_configuration_files_exist():
    """Test that configuration files are present."""
    package_root = Path(__file__).parent.parent
    repo_root = package_root.parent

    # Package-level config files
    assert (package_root / "pytest.ini").exists()
    assert (package_root / "requirements.txt").exists()
    assert (package_root / "README.md").exists()
    # Repo-level config files (pyproject.toml consolidated to repo root)
    assert (repo_root / "pyproject.toml").exists()


def test_package_imports():
    """Test that the package can be imported."""
    try:
        import ergodic_insurance
        from ergodic_insurance._version import __version__ as source_version

        assert ergodic_insurance.__version__ == source_version
    except ImportError:
        pytest.skip("Package not installed in editable mode yet")


@pytest.mark.unit
def test_pytest_markers():
    """Test that pytest markers are properly configured.

    This test verifies that the pytest marker system is functioning
    correctly by checking that this test runs when unit tests are selected.
    """
    # Verify that pytest module is imported and functional
    assert pytest.__version__ is not None
    # Verify that the mark decorator exists and can be accessed
    assert hasattr(pytest.mark, "unit")
    assert hasattr(pytest.mark, "slow")
    assert hasattr(pytest.mark, "integration")


@pytest.mark.parametrize(
    "value,expected",
    [
        (1, 1),
        (2, 2),
        (3, 3),
    ],
)
def test_parametrize_working(value, expected):
    """Test that parametrized tests work correctly."""
    assert value == expected
