"""Scenario management system for batch processing simulations.

This module provides a framework for managing multiple simulation scenarios,
parameter sweeps, and configuration variations for comprehensive analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
from itertools import product
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator

from .config import Config
from .monte_carlo import SimulationConfig


class ScenarioType(Enum):
    """Types of scenario generation methods."""

    SINGLE = "single"
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    CUSTOM = "custom"
    SENSITIVITY = "sensitivity"


class ParameterSpec(BaseModel):
    """Specification for parameter variations in scenarios.

    Attributes:
        name: Parameter name (dot notation for nested params)
        values: List of values for grid search
        min_value: Minimum value for random search
        max_value: Maximum value for random search
        n_samples: Number of samples for random search
        distribution: Distribution type for random sampling
        base_value: Base value for sensitivity analysis
        variation_pct: Percentage variation for sensitivity
    """

    name: str = Field(description="Parameter name with dot notation")
    values: Optional[List[Any]] = Field(default=None, description="Explicit values")
    min_value: Optional[float] = Field(default=None, description="Min for random search")
    max_value: Optional[float] = Field(default=None, description="Max for random search")
    n_samples: int = Field(default=10, description="Samples for random search")
    distribution: str = Field(default="uniform", description="Distribution type")
    base_value: Optional[Any] = Field(default=None, description="Base value")
    variation_pct: float = Field(default=0.1, description="Variation percentage")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate parameter name format."""
        if not v:
            raise ValueError("Parameter name cannot be empty")
        return v

    def generate_values(self, method: ScenarioType) -> List[Any]:
        """Generate parameter values based on method.

        Args:
            method: Scenario generation method

        Returns:
            List of parameter values
        """
        if method == ScenarioType.GRID_SEARCH and self.values:
            return self.values
        elif method == ScenarioType.RANDOM_SEARCH:
            if self.min_value is not None and self.max_value is not None:
                if self.distribution == "uniform":
                    return list(np.random.uniform(self.min_value, self.max_value, self.n_samples))
                elif self.distribution == "log":
                    return list(
                        np.exp(
                            np.random.uniform(
                                np.log(self.min_value), np.log(self.max_value), self.n_samples
                            )
                        )
                    )
        elif method == ScenarioType.SENSITIVITY and self.base_value is not None:
            variations = [-self.variation_pct, 0, self.variation_pct]
            return [self.base_value * (1 + v) for v in variations]

        return self.values or [self.base_value]


@dataclass
class ScenarioConfig:
    """Configuration for a single scenario.

    Attributes:
        scenario_id: Unique identifier for the scenario
        name: Human-readable scenario name
        description: Detailed description
        base_config: Base configuration object
        simulation_config: Simulation configuration
        parameter_overrides: Parameter overrides to apply
        tags: Tags for categorization
        priority: Execution priority (lower = higher priority)
        created_at: Creation timestamp
        metadata: Additional metadata
    """

    scenario_id: str
    name: str
    description: str = ""
    base_config: Optional[Config] = None
    simulation_config: Optional[SimulationConfig] = None
    parameter_overrides: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    priority: int = 100
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize scenario with defaults."""
        if not self.scenario_id:
            self.scenario_id = self.generate_id()
        if not self.simulation_config:
            self.simulation_config = SimulationConfig()

    def generate_id(self) -> str:
        """Generate unique scenario ID from configuration.

        Returns:
            Unique scenario identifier
        """
        # Create hash from key configuration elements
        key_data = {
            "name": self.name,
            "overrides": self.parameter_overrides,
            "created": str(self.created_at),
        }
        hash_str = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()[:8]
        return f"scenario_{hash_str}"

    def apply_overrides(self, config: Any) -> Any:
        """Apply parameter overrides to configuration.

        Args:
            config: Configuration object to modify

        Returns:
            Modified configuration
        """
        for param_path, value in self.parameter_overrides.items():
            parts = param_path.split(".")
            obj = config

            # Navigate to the parameter
            for part in parts[:-1]:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                elif isinstance(obj, dict) and part in obj:
                    obj = obj[part]
                else:
                    break

            # Set the value
            final_part = parts[-1]
            if hasattr(obj, final_part):
                setattr(obj, final_part, value)
            elif isinstance(obj, dict):
                obj[final_part] = value

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert scenario to dictionary representation.

        Returns:
            Dictionary representation
        """
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "parameter_overrides": self.parameter_overrides,
            "tags": list(self.tags),
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


class ScenarioManager:
    """Manager for creating and organizing simulation scenarios."""

    def __init__(self):
        """Initialize scenario manager."""
        self.scenarios: List[ScenarioConfig] = []
        self.scenario_index: Dict[str, ScenarioConfig] = {}

    def create_scenario(
        self,
        name: str,
        base_config: Optional[Config] = None,
        simulation_config: Optional[SimulationConfig] = None,
        parameter_overrides: Optional[Dict[str, Any]] = None,
        description: str = "",
        tags: Optional[Set[str]] = None,
        priority: int = 100,
    ) -> ScenarioConfig:
        """Create a single scenario.

        Args:
            name: Scenario name
            base_config: Base configuration
            simulation_config: Simulation configuration
            parameter_overrides: Parameters to override
            description: Scenario description
            tags: Scenario tags
            priority: Execution priority

        Returns:
            Created scenario configuration
        """
        scenario = ScenarioConfig(
            scenario_id="",  # Will be generated
            name=name,
            description=description,
            base_config=base_config,
            simulation_config=simulation_config or SimulationConfig(),
            parameter_overrides=parameter_overrides or {},
            tags=tags or set(),
            priority=priority,
        )

        self.add_scenario(scenario)
        return scenario

    def add_scenario(self, scenario: ScenarioConfig) -> None:
        """Add scenario to manager.

        Args:
            scenario: Scenario to add
        """
        if scenario.scenario_id in self.scenario_index:
            # Check for duplicate
            existing = self.scenario_index[scenario.scenario_id]
            if existing.parameter_overrides == scenario.parameter_overrides:
                return  # Skip duplicate

        self.scenarios.append(scenario)
        self.scenario_index[scenario.scenario_id] = scenario

    def create_grid_search(
        self,
        name_template: str,
        parameter_specs: List[ParameterSpec],
        base_config: Optional[Config] = None,
        simulation_config: Optional[SimulationConfig] = None,
        tags: Optional[Set[str]] = None,
    ) -> List[ScenarioConfig]:
        """Create scenarios for grid search over parameters.

        Args:
            name_template: Template for scenario names
            parameter_specs: Parameter specifications
            base_config: Base configuration
            simulation_config: Simulation configuration
            tags: Common tags for all scenarios

        Returns:
            List of created scenarios
        """
        scenarios = []

        # Generate value combinations
        param_names = [spec.name for spec in parameter_specs]
        param_values = [spec.generate_values(ScenarioType.GRID_SEARCH) for spec in parameter_specs]

        # Create scenarios for each combination
        for i, values in enumerate(product(*param_values)):
            overrides = dict(zip(param_names, values))

            # Format scenario name
            param_str = "_".join(
                f"{k.split('.')[-1]}={v:.3g}" if isinstance(v, float) else f"{k.split('.')[-1]}={v}"
                for k, v in overrides.items()
            )
            name = name_template.format(params=param_str, index=i)

            scenario = self.create_scenario(
                name=name,
                base_config=base_config,
                simulation_config=simulation_config,
                parameter_overrides=overrides,
                description=f"Grid search scenario {i+1}",
                tags=(tags or set()) | {"grid_search"},
                priority=50,  # Higher priority for systematic search
            )
            scenarios.append(scenario)

        return scenarios

    def create_random_search(
        self,
        name_template: str,
        parameter_specs: List[ParameterSpec],
        n_scenarios: int,
        base_config: Optional[Config] = None,
        simulation_config: Optional[SimulationConfig] = None,
        tags: Optional[Set[str]] = None,
        seed: Optional[int] = None,
    ) -> List[ScenarioConfig]:
        """Create scenarios for random search over parameters.

        Args:
            name_template: Template for scenario names
            parameter_specs: Parameter specifications
            n_scenarios: Number of scenarios to generate
            base_config: Base configuration
            simulation_config: Simulation configuration
            tags: Common tags for all scenarios
            seed: Random seed for reproducibility

        Returns:
            List of created scenarios
        """
        if seed is not None:
            np.random.seed(seed)

        scenarios = []

        for i in range(n_scenarios):
            overrides = {}
            for spec in parameter_specs:
                values = spec.generate_values(ScenarioType.RANDOM_SEARCH)
                if values:
                    overrides[spec.name] = np.random.choice(values)

            name = name_template.format(index=i)

            scenario = self.create_scenario(
                name=name,
                base_config=base_config,
                simulation_config=simulation_config,
                parameter_overrides=overrides,
                description=f"Random search scenario {i+1}",
                tags=(tags or set()) | {"random_search"},
                priority=75,  # Medium priority
            )
            scenarios.append(scenario)

        return scenarios

    def create_sensitivity_analysis(
        self,
        base_name: str,
        parameter_specs: List[ParameterSpec],
        base_config: Optional[Config] = None,
        simulation_config: Optional[SimulationConfig] = None,
        tags: Optional[Set[str]] = None,
    ) -> List[ScenarioConfig]:
        """Create scenarios for sensitivity analysis.

        Args:
            base_name: Base name for scenarios
            parameter_specs: Parameters to vary
            base_config: Base configuration
            simulation_config: Simulation configuration
            tags: Common tags for all scenarios

        Returns:
            List of created scenarios
        """
        scenarios = []

        # Create baseline scenario
        baseline = self.create_scenario(
            name=f"{base_name}_baseline",
            base_config=base_config,
            simulation_config=simulation_config,
            parameter_overrides={},
            description="Baseline scenario",
            tags=(tags or set()) | {"sensitivity", "baseline"},
            priority=25,  # Highest priority
        )
        scenarios.append(baseline)

        # Create sensitivity scenarios
        for spec in parameter_specs:
            values = spec.generate_values(ScenarioType.SENSITIVITY)
            for i, value in enumerate(values):
                if i == 1:  # Skip middle value (baseline)
                    continue

                direction = "high" if i > 1 else "low"
                name = f"{base_name}_{spec.name.replace('.', '_')}_{direction}"

                scenario = self.create_scenario(
                    name=name,
                    base_config=base_config,
                    simulation_config=simulation_config,
                    parameter_overrides={spec.name: value},
                    description=f"Sensitivity: {spec.name} {direction}",
                    tags=(tags or set()) | {"sensitivity", direction},
                    priority=30,
                )
                scenarios.append(scenario)

        return scenarios

    def get_scenarios_by_tag(self, tag: str) -> List[ScenarioConfig]:
        """Get scenarios with specific tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of matching scenarios
        """
        return [s for s in self.scenarios if tag in s.tags]

    def get_scenarios_by_priority(self, max_priority: int = 100) -> List[ScenarioConfig]:
        """Get scenarios up to priority threshold.

        Args:
            max_priority: Maximum priority value (inclusive)

        Returns:
            Sorted list of scenarios
        """
        filtered = [s for s in self.scenarios if s.priority <= max_priority]
        return sorted(filtered, key=lambda x: x.priority)

    def clear_scenarios(self) -> None:
        """Clear all scenarios."""
        self.scenarios.clear()
        self.scenario_index.clear()

    def export_scenarios(self, path: Union[str, Path]) -> None:
        """Export scenarios to JSON file.

        Args:
            path: Output file path
        """
        path = Path(path)
        data = {
            "scenarios": [s.to_dict() for s in self.scenarios],
            "metadata": {
                "n_scenarios": len(self.scenarios),
                "exported_at": datetime.now().isoformat(),
            },
        }

        with open(path, "w") as f:
            json.dumps(data, indent=2, default=str)

    def import_scenarios(self, path: Union[str, Path]) -> None:
        """Import scenarios from JSON file.

        Args:
            path: Input file path
        """
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)

        for scenario_dict in data.get("scenarios", []):
            # Reconstruct scenario
            scenario = ScenarioConfig(
                scenario_id=scenario_dict["scenario_id"],
                name=scenario_dict["name"],
                description=scenario_dict.get("description", ""),
                parameter_overrides=scenario_dict.get("parameter_overrides", {}),
                tags=set(scenario_dict.get("tags", [])),
                priority=scenario_dict.get("priority", 100),
            )
            # Parse datetime
            if "created_at" in scenario_dict:
                scenario.created_at = datetime.fromisoformat(scenario_dict["created_at"])

            self.add_scenario(scenario)
