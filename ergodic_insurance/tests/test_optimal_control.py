"""Tests for optimal control module.

Author: Alex Filiakov
Date: 2025-01-26
"""

import numpy as np
import pandas as pd
import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.hjb_solver import (
    ControlVariable,
    ExpectedWealth,
    HJBProblem,
    HJBSolver,
    HJBSolverConfig,
    LogUtility,
    StateSpace,
    StateVariable,
)
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.optimal_control import (
    ControlMode,
    ControlSpace,
    HJBFeedbackControl,
    OptimalController,
    StaticControl,
    TimeVaryingControl,
    create_hjb_controller,
)


class TestControlSpace:
    """Test ControlSpace class."""

    def test_control_space_creation(self):
        """Test creating control space."""
        space = ControlSpace(limits=[(1e6, 5e7), (5e7, 1e8)], retentions=[(1e5, 1e6), (1e6, 5e6)])

        assert len(space.limits) == 2
        assert len(space.retentions) == 2
        assert len(space.coverage_percentages) == 2
        assert space.coverage_percentages[0] == (0.8, 1.0)

    def test_control_space_validation(self):
        """Test control space validation."""
        # Mismatched lengths
        with pytest.raises(ValueError, match="must have same number of layers"):
            ControlSpace(limits=[(1e6, 5e7)], retentions=[(1e5, 1e6), (1e6, 5e6)])

        # Invalid limit bounds
        with pytest.raises(ValueError, match="Invalid limit bounds"):
            ControlSpace(limits=[(5e7, 1e6)], retentions=[(1e5, 1e6)])  # Min > Max

        # Invalid retention bounds
        with pytest.raises(ValueError, match="Invalid retention bounds"):
            ControlSpace(limits=[(1e6, 5e7)], retentions=[(1e6, 1e5)])  # Min > Max

        # Invalid coverage bounds
        with pytest.raises(ValueError, match="Invalid coverage percentage bounds"):
            ControlSpace(
                limits=[(1e6, 5e7)],
                retentions=[(1e5, 1e6)],
                coverage_percentages=[(1.5, 2.0)],  # > 1
            )

    def test_dimensions(self):
        """Test dimension calculation."""
        # 2 layers with limits and retentions
        space = ControlSpace(limits=[(1e6, 5e7), (5e7, 1e8)], retentions=[(1e5, 1e6), (1e6, 5e6)])
        assert space.get_dimensions() == 6  # 2 limits + 2 retentions + 2 coverages

        # Add reinsurance
        space = ControlSpace(
            limits=[(1e6, 5e7)], retentions=[(1e5, 1e6)], reinsurance_limits=[(1e8, 5e8)]
        )
        assert space.get_dimensions() == 4  # 1 limit + 1 retention + 1 coverage + 1 reinsurance

    def test_array_conversion(self):
        """Test converting to/from array format."""
        space = ControlSpace(limits=[(1e6, 5e7), (5e7, 1e8)], retentions=[(1e5, 1e6), (1e6, 5e6)])

        # To array
        limits = [2e7, 7e7]
        retentions = [5e5, 3e6]
        coverages = [0.9, 0.95]

        array = space.to_array(limits, retentions, coverages)
        assert len(array) == 6
        assert array[0] == 2e7
        assert array[2] == 5e5
        assert array[4] == 0.9

        # From array
        params = space.from_array(array)
        assert params["limits"] == limits
        assert params["retentions"] == retentions
        assert params["coverages"] == coverages


class TestStaticControl:
    """Test StaticControl strategy."""

    def test_static_control_creation(self):
        """Test creating static control."""
        control = StaticControl(limits=[1e7, 5e7], retentions=[1e6, 5e6], coverages=[0.9, 0.95])

        assert control.limits == [1e7, 5e7]
        assert control.retentions == [1e6, 5e6]
        assert control.coverages == [0.9, 0.95]

    def test_get_control(self):
        """Test getting control parameters."""
        control = StaticControl(limits=[1e7], retentions=[1e6])

        # Should return same values regardless of state/time
        params1 = control.get_control({"wealth": 1e7}, time=0)
        params2 = control.get_control({"wealth": 5e7}, time=10)

        assert params1 == params2
        assert params1["limits"] == [1e7]
        assert params1["retentions"] == [1e6]
        assert params1["coverages"] == [1.0]

    def test_update(self):
        """Test that update does nothing for static control."""
        control = StaticControl([1e7], [1e6])

        # Update should not change anything
        control.update({"wealth": 1e7}, {"loss": 1e5})

        params = control.get_control({}, 0)
        assert params["limits"] == [1e7]


class TestTimeVaryingControl:
    """Test TimeVaryingControl strategy."""

    def test_time_varying_creation(self):
        """Test creating time-varying control."""
        control = TimeVaryingControl(
            time_schedule=[0, 5, 10],
            limits_schedule=[[1e7], [2e7], [3e7]],
            retentions_schedule=[[1e6], [2e6], [3e6]],
        )

        assert len(control.time_schedule) == 3
        assert control.limits_schedule.shape == (3, 1)
        assert control.retentions_schedule.shape == (3, 1)

    def test_time_interpolation(self):
        """Test control interpolation over time."""
        control = TimeVaryingControl(
            time_schedule=[0, 10],
            limits_schedule=[[1e7], [2e7]],
            retentions_schedule=[[1e6], [2e6]],
        )

        # At t=0
        params = control.get_control({}, time=0)
        assert params["limits"] == [1e7]
        assert params["retentions"] == [1e6]

        # At t=5 (midpoint)
        params = control.get_control({}, time=5)
        assert params["limits"] == [1.5e7]
        assert params["retentions"] == [1.5e6]

        # At t=10
        params = control.get_control({}, time=10)
        assert params["limits"] == [2e7]
        assert params["retentions"] == [2e6]

        # Beyond schedule
        params = control.get_control({}, time=15)
        assert params["limits"] == [2e7]  # Should use last value

    def test_validation(self):
        """Test schedule validation."""
        with pytest.raises(ValueError, match="Schedule lengths must match"):
            TimeVaryingControl(
                time_schedule=[0, 1],
                limits_schedule=[[1e7], [2e7], [3e7]],  # Wrong length
                retentions_schedule=[[1e6], [2e6]],
            )


class TestHJBFeedbackControl:
    """Test HJBFeedbackControl strategy."""

    @pytest.fixture
    def simple_hjb_solver(self):
        """Create a simple solved HJB problem."""
        # Create simple 2D problem
        state_space = StateSpace(
            [StateVariable("wealth", 1e6, 1e8, 5), StateVariable("time", 0, 10, 3)]
        )

        control_variables = [
            ControlVariable("limit", 1e6, 5e7, 3),
            ControlVariable("retention", 1e5, 1e6, 3),
        ]

        def test_dynamics(state, control, time):
            """Simple test dynamics."""
            # Return zero drift for all state variables
            return np.zeros_like(state)

        def test_running_cost(state, control, time):
            """Simple test running cost."""
            # Return zeros with the correct shape (without the state dimension)
            return np.zeros(state.shape[:-1])

        problem = HJBProblem(
            state_space=state_space,
            control_variables=control_variables,
            utility_function=LogUtility(),
            dynamics=test_dynamics,
            running_cost=test_running_cost,
            time_horizon=10,
        )

        config = HJBSolverConfig(max_iterations=2, verbose=False)  # Just enough to initialize

        solver = HJBSolver(problem, config)
        solver.solve()

        return solver

    def test_feedback_control_creation(self, simple_hjb_solver):
        """Test creating HJB feedback control."""
        control_space = ControlSpace(limits=[(1e6, 5e7)], retentions=[(1e5, 1e6)])

        control = HJBFeedbackControl(simple_hjb_solver, control_space)

        assert control.hjb_solver == simple_hjb_solver
        assert control.control_space == control_space
        assert control.interpolators is not None

    def test_feedback_control_extraction(self, simple_hjb_solver):
        """Test extracting feedback control."""
        control_space = ControlSpace(limits=[(1e6, 5e7)], retentions=[(1e5, 1e6)])

        control = HJBFeedbackControl(simple_hjb_solver, control_space)

        # Get control at specific state
        state = {"assets": 5e7, "time": 5}
        params = control.get_control(state, time=5)

        assert "limits" in params
        assert "retentions" in params
        assert "coverages" in params
        assert len(params["limits"]) == 1
        assert len(params["retentions"]) == 1

    def test_state_mapping(self, simple_hjb_solver):
        """Test custom state mapping."""
        control_space = ControlSpace(limits=[(1e6, 5e7)], retentions=[(1e5, 1e6)])

        # Custom mapping function
        def custom_mapping(state):
            return np.array([state.get("custom_wealth", 1e7), 0])

        control = HJBFeedbackControl(simple_hjb_solver, control_space, state_mapping=custom_mapping)

        # Should use custom mapping
        state = {"custom_wealth": 2e7}
        params = control.get_control(state, time=0)

        assert params is not None
        assert "limits" in params

    def test_interpolation_dimension_handling(self, simple_hjb_solver):
        """Test that HJB feedback control handles state dimensions correctly.

        This is a regression test for an issue where the interpolator
        received incorrectly shaped states.
        """
        control_space = ControlSpace(limits=[(1e6, 5e7)], retentions=[(1e5, 1e6)])
        control = HJBFeedbackControl(simple_hjb_solver, control_space)

        # Test various state dictionary formats
        test_states = [
            {"assets": 2e7, "time": 1.0},  # Standard format
            {"wealth": 3e7, "time": 2.0},  # Alternative naming
            {"equity": 1.5e7},  # Missing time (should default)
            {"assets": 4e7, "time": 0.5, "extra_key": 100},  # Extra keys
        ]

        for state in test_states:
            # Should not raise dimension errors
            params = control.get_control(state, time=0)
            assert params is not None
            assert "limits" in params
            assert "retentions" in params
            assert len(params["limits"]) == 1
            assert len(params["retentions"]) == 1


class TestOptimalController:
    """Test OptimalController class."""

    @pytest.fixture
    def manufacturer(self):
        """Create a test manufacturer."""
        config = ManufacturerConfig(
            initial_assets=1e7,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.6,
        )
        return WidgetManufacturer(config)

    def test_controller_creation(self):
        """Test creating optimal controller."""
        strategy = StaticControl([1e7], [1e6])
        control_space = ControlSpace(limits=[(1e6, 5e7)], retentions=[(1e5, 1e6)])

        controller = OptimalController(strategy, control_space)

        assert controller.strategy == strategy
        assert controller.control_space == control_space
        assert controller.control_history == []
        assert controller.state_history == []

    def test_apply_control(self, manufacturer):
        """Test applying control to create insurance program."""
        strategy = StaticControl([2e7, 3e7], [1e6, 5e6])
        control_space = ControlSpace(
            limits=[(1e6, 5e7), (5e7, 1e8)], retentions=[(1e5, 1e6), (1e6, 1e7)]
        )

        controller = OptimalController(strategy, control_space)

        # Apply control
        program = controller.apply_control(manufacturer, time=0)

        assert program is not None
        assert len(program.layers) == 2
        assert program.layers[0].limit == 2e7
        assert program.layers[0].attachment_point == 1e6
        assert program.layers[1].limit == 3e7
        assert program.layers[1].attachment_point == 5e6

        # Check history
        assert len(controller.control_history) == 1
        assert len(controller.state_history) == 1

    def test_state_extraction(self, manufacturer):
        """Test extracting state from manufacturer."""
        strategy = StaticControl([1e7], [1e6])
        control_space = ControlSpace(limits=[(1e6, 5e7)], retentions=[(1e5, 1e6)])

        controller = OptimalController(strategy, control_space)

        # Extract state
        state = controller._extract_state(manufacturer)

        assert "assets" in state
        assert "equity" in state
        assert "wealth" in state
        assert "debt" in state
        assert state["assets"] > 0
        assert state["equity"] > 0

    def test_performance_summary(self, manufacturer):
        """Test getting performance summary."""
        strategy = StaticControl([1e7], [1e6])
        control_space = ControlSpace(limits=[(1e6, 5e7)], retentions=[(1e5, 1e6)])

        controller = OptimalController(strategy, control_space)

        # Apply control multiple times
        for i in range(3):
            controller.apply_control(manufacturer, time=i)
            controller.update_outcome({"loss": i * 1e5, "cost": i * 1e4})

        # Get summary
        summary = controller.get_performance_summary()

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 3
        assert "step" in summary.columns
        assert any("state_" in col for col in summary.columns)
        assert any("control_" in col for col in summary.columns)
        assert any("outcome_" in col for col in summary.columns)

    def test_reset(self, manufacturer):
        """Test resetting controller history."""
        strategy = StaticControl([1e7], [1e6])
        control_space = ControlSpace(limits=[(1e6, 5e7)], retentions=[(1e5, 1e6)])

        controller = OptimalController(strategy, control_space)

        # Apply control and add history
        controller.apply_control(manufacturer, time=0)
        controller.update_outcome({"loss": 1e5})

        assert len(controller.control_history) == 1
        assert len(controller.outcome_history) == 1

        # Reset
        controller.reset()

        assert len(controller.control_history) == 0
        assert len(controller.state_history) == 0
        assert len(controller.outcome_history) == 0


class TestCreateHJBController:
    """Test the convenience function for creating HJB controllers."""

    @pytest.fixture
    def manufacturer(self):
        """Create a test manufacturer."""
        config = ManufacturerConfig(
            initial_assets=1e7,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.6,
        )
        return WidgetManufacturer(config)

    @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
    def test_create_log_utility_controller(self, manufacturer):
        """Test creating controller with log utility."""
        controller = create_hjb_controller(manufacturer, simulation_years=1, utility_type="log")

        assert isinstance(controller, OptimalController)
        assert isinstance(controller.strategy, HJBFeedbackControl)
        assert controller.control_space is not None

        # Test that it can apply control
        program = controller.apply_control(manufacturer, time=0)
        assert program is not None
        assert len(program.layers) > 0

    @pytest.mark.slow
    def test_create_power_utility_controller(self, manufacturer):
        """Test creating controller with power utility."""
        controller = create_hjb_controller(
            manufacturer, simulation_years=1, utility_type="power", risk_aversion=3.0
        )

        assert isinstance(controller, OptimalController)
        assert isinstance(controller.strategy, HJBFeedbackControl)

    @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
    def test_create_linear_utility_controller(self, manufacturer):
        """Test creating controller with linear utility."""
        controller = create_hjb_controller(manufacturer, simulation_years=1, utility_type="linear")

        assert isinstance(controller, OptimalController)
        assert isinstance(controller.strategy, HJBFeedbackControl)

    def test_invalid_utility_type(self, manufacturer):
        """Test error handling for invalid utility type."""
        with pytest.raises(ValueError, match="Unknown utility type"):
            create_hjb_controller(manufacturer, simulation_years=1, utility_type="invalid")

    def test_hjb_controller_with_manufacturer_integration(self, manufacturer):
        """Test that HJB controller integrates properly with manufacturer.

        This is a regression test to ensure the HJB feedback control
        correctly handles manufacturer state extraction and interpolation.
        """
        # Create a minimal HJB problem
        state_space = StateSpace(
            [StateVariable("wealth", 1e6, 1e8, 3), StateVariable("time", 0, 5, 3)]
        )

        control_variables = [
            ControlVariable("limit", 1e6, 3e7, 3),
            ControlVariable("retention", 1e5, 5e6, 3),
        ]

        def dynamics(state, control, time):
            drift = np.zeros_like(state)
            drift[..., 0] = state[..., 0] * 0.05
            drift[..., 1] = 1.0
            return drift

        def running_cost(state, control, time):
            return np.zeros(state.shape[:-1])

        problem = HJBProblem(
            state_space=state_space,
            control_variables=control_variables,
            utility_function=LogUtility(),
            dynamics=dynamics,
            running_cost=running_cost,
            time_horizon=5.0,
        )

        config = HJBSolverConfig(max_iterations=2, verbose=False)
        solver = HJBSolver(problem, config)
        solver.solve()

        # Create controller
        control_space = ControlSpace(limits=[(1e6, 3e7)], retentions=[(1e5, 5e6)])
        hjb_strategy = HJBFeedbackControl(solver, control_space)
        controller = OptimalController(hjb_strategy, control_space)

        # Should be able to apply control without dimension errors
        insurance = controller.apply_control(manufacturer, time=0)
        assert insurance is not None
        assert hasattr(insurance, "layers")
        assert len(insurance.layers) > 0
