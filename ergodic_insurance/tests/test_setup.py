"""Test to verify project setup is working correctly."""

from pathlib import Path

import pytest


def test_project_structure_exists():
    """Test that the basic project structure exists."""
    project_root = Path(__file__).parent.parent

    assert project_root.exists()
    assert (project_root / "src").exists()
    assert (project_root / "tests").exists()
    assert (project_root / "notebooks").exists()
    assert (project_root / "data" / "parameters").exists()


def test_configuration_files_exist():
    """Test that configuration files are present."""
    project_root = Path(__file__).parent.parent

    assert (project_root / "pytest.ini").exists()
    assert (project_root / "pyproject.toml").exists()
    assert (project_root / "requirements.txt").exists()
    assert (project_root / "setup.py").exists()
    assert (project_root / "README.md").exists()


def test_package_imports():
    """Test that the package can be imported."""
    try:
        import src

        assert src.__version__ == "0.1.0"
    except ImportError:
        pytest.skip("Package not installed in editable mode yet")


@pytest.mark.unit
def test_pytest_markers():
    """Test that pytest markers are working."""
    assert True


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
