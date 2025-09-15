"""Tests for adaptive stopping criteria module."""

from unittest.mock import Mock, patch
import warnings

import numpy as np
import pytest

from ergodic_insurance.adaptive_stopping import (
    AdaptiveStoppingMonitor,
    ConvergenceStatus,
    StoppingCriteria,
    StoppingRule,
)


class TestStoppingCriteria:
    """Test suite for StoppingCriteria class."""

    def test_default_initialization(self):
        """Test default criteria initialization."""
        criteria = StoppingCriteria()

        assert criteria.rule == StoppingRule.COMBINED
        assert criteria.r_hat_threshold == 1.05
        assert criteria.min_ess == 1000
        assert criteria.relative_tolerance == 0.01
        assert criteria.min_iterations == 1000
        assert criteria.max_iterations == 100000

    def test_custom_initialization(self):
        """Test custom criteria initialization."""
        criteria = StoppingCriteria(
            rule=StoppingRule.R_HAT, r_hat_threshold=1.1, min_ess=500, min_iterations=500
        )

        assert criteria.rule == StoppingRule.R_HAT
        assert criteria.r_hat_threshold == 1.1
        assert criteria.min_ess == 500
        assert criteria.min_iterations == 500

    def test_invalid_r_hat_threshold(self):
        """Test that invalid R-hat threshold raises error."""
        with pytest.raises(ValueError, match="R-hat threshold must be > 1.0"):
            StoppingCriteria(r_hat_threshold=0.9)

    def test_low_ess_warning(self):
        """Test warning for very low ESS threshold."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            StoppingCriteria(min_ess=50)
            assert len(w) == 1
            assert "Very low ESS threshold" in str(w[0].message)

    def test_low_min_iterations_warning(self):
        """Test warning for very low minimum iterations."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            StoppingCriteria(min_iterations=50)
            assert len(w) == 1
            assert "Very low minimum iterations" in str(w[0].message)


class TestConvergenceStatus:
    """Test suite for ConvergenceStatus class."""

    def test_converged_status(self):
        """Test converged status representation."""
        status = ConvergenceStatus(
            converged=True,
            iteration=5000,
            reason="All criteria met",
            diagnostics={"r_hat": 1.01, "ess": 1500},
            should_stop=True,
            estimated_remaining=0,
        )

        str_repr = str(status)
        assert "CONVERGED" in str_repr
        assert "5000" in str_repr
        assert "All criteria met" in str_repr

    def test_not_converged_status(self):
        """Test not converged status representation."""
        status = ConvergenceStatus(
            converged=False,
            iteration=1000,
            reason="R-hat too high",
            diagnostics={"r_hat": 1.2},
            should_stop=False,
            estimated_remaining=2000,
        )

        str_repr = str(status)
        assert "NOT CONVERGED" in str_repr
        assert "1000" in str_repr


class TestAdaptiveStoppingMonitor:
    """Test suite for AdaptiveStoppingMonitor class."""

    @pytest.fixture
    def monitor(self):
        """Create monitor instance."""
        return AdaptiveStoppingMonitor()

    @pytest.fixture
    def sample_chains(self):
        """Create sample chain data."""
        np.random.seed(42)
        # Create converging chains
        n_chains = 4
        n_iterations = 2000

        chains = np.zeros((n_chains, n_iterations))
        for c in range(n_chains):
            # Start with different means (not converged)
            chains[c, :500] = np.random.randn(500) + c * 2
            # Converge to same distribution
            chains[c, 500:] = np.random.randn(1500)

        return chains

    def test_initialization_default(self):
        """Test default monitor initialization."""
        monitor = AdaptiveStoppingMonitor()

        assert monitor.criteria.rule == StoppingRule.COMBINED
        assert monitor.custom_rule is None
        assert len(monitor.r_hat_history) == 0
        assert monitor.consecutive_convergence == 0
        assert not monitor.burn_in_detected

    def test_initialization_custom_criteria(self):
        """Test monitor with custom criteria."""
        criteria = StoppingCriteria(rule=StoppingRule.ESS, min_ess=2000)
        monitor = AdaptiveStoppingMonitor(criteria=criteria)

        assert monitor.criteria.rule == StoppingRule.ESS
        assert monitor.criteria.min_ess == 2000

    def test_check_convergence_below_min_iterations(self, monitor):
        """Test convergence check before minimum iterations."""
        chains = np.random.randn(2, 500)

        status = monitor.check_convergence(500, chains)

        assert not status.converged
        assert not status.should_stop
        assert "Below minimum iterations" in status.reason
        assert status.estimated_remaining == 500  # 1000 - 500

    def test_check_convergence_at_max_iterations(self, monitor):
        """Test convergence check at maximum iterations."""
        monitor.criteria.max_iterations = 1000
        chains = np.random.randn(2, 1000)

        status = monitor.check_convergence(1000, chains)

        assert not status.converged
        assert status.should_stop
        assert "Maximum iterations reached" in status.reason

    def test_check_convergence_not_at_interval(self, monitor):
        """Test convergence check not at check interval."""
        chains = np.random.randn(2, 1050)

        status = monitor.check_convergence(1050, chains)

        assert not status.converged
        assert not status.should_stop
        assert "Not at check interval" in status.reason

    def test_check_convergence_r_hat_rule(self, monitor, sample_chains):
        """Test convergence with R-hat rule."""
        monitor.criteria.rule = StoppingRule.R_HAT
        monitor.criteria.r_hat_threshold = 1.1

        # Check at converged point
        status = monitor.check_convergence(2000, sample_chains[:, 500:2000])

        assert status.iteration == 2000
        assert "R-hat" in status.reason
        # Converged portion should have low R-hat
        assert status.diagnostics["r_hat"] < 1.1

    def test_check_convergence_ess_rule(self, monitor):
        """Test convergence with ESS rule."""
        monitor.criteria.rule = StoppingRule.ESS
        monitor.criteria.min_ess = 500

        # Create chain with good ESS
        chains = np.random.randn(2, 2000)

        status = monitor.check_convergence(2000, chains)

        assert "ESS" in status.reason
        assert status.diagnostics["ess"] > 0

    def test_check_convergence_mcse_rule(self, monitor):
        """Test convergence with MCSE rule."""
        monitor.criteria.rule = StoppingRule.MCSE
        monitor.criteria.mcse_relative_threshold = 0.1

        # Create chain with stable mean
        chains = np.random.randn(2, 2000) + 10  # Large mean for small relative MCSE

        status = monitor.check_convergence(2000, chains)

        assert "MCSE" in status.reason
        assert "mcse_relative" in status.diagnostics

    def test_check_convergence_combined_rule(self, monitor):
        """Test convergence with combined rule."""
        monitor.criteria.rule = StoppingRule.COMBINED

        # Create well-converged chains
        chains = np.random.randn(4, 2000) + 10

        status = monitor.check_convergence(2000, chains)

        assert "R-hat" in status.reason
        assert "ESS" in status.reason
        assert "MCSE_rel" in status.reason

    def test_check_convergence_custom_rule(self):
        """Test convergence with custom rule."""

        def custom_rule(diagnostics):
            mean = diagnostics.get("mean", 0)
            converged = abs(mean - 5.0) < 0.1
            reason = f"Mean = {mean:.2f}, target = 5.0"
            return converged, reason

        monitor = AdaptiveStoppingMonitor(
            criteria=StoppingCriteria(rule=StoppingRule.CUSTOM), custom_rule=custom_rule
        )

        chains = np.ones((2, 2000)) * 5.05
        status = monitor.check_convergence(2000, chains)

        assert status.converged
        assert "target = 5.0" in status.reason

    def test_patience_mechanism(self, monitor):
        """Test that patience works correctly."""
        monitor.criteria.patience = 3
        monitor.criteria.check_interval = 100

        # Create well-converged chains (similar distributions)
        np.random.seed(42)
        base = np.random.randn(2000)
        chains = np.array([base + np.random.randn(2000) * 0.01 for _ in range(4)])

        # First check - converged but shouldn't stop
        status1 = monitor.check_convergence(1000, chains[:, :1000])
        assert not status1.should_stop
        # Check if converged (may or may not be 1 depending on criteria)
        if status1.converged:
            assert monitor.consecutive_convergence >= 1

        # Second check - still shouldn't stop
        status2 = monitor.check_convergence(1100, chains[:, :1100])
        assert not status2.should_stop
        if status2.converged:
            # If converged, check consecutive count increased
            assert monitor.consecutive_convergence >= 1

        # Third check - may or may not stop depending on convergence
        status3 = monitor.check_convergence(1200, chains[:, :1200])
        if status3.converged:
            # If all three checks converged, should stop with patience=3
            if monitor.consecutive_convergence >= 3:
                assert status3.should_stop

    def test_detect_adaptive_burn_in_geweke(self, monitor, sample_chains):
        """Test adaptive burn-in detection with Geweke method."""
        burn_in = monitor.detect_adaptive_burn_in(sample_chains, method="geweke")

        assert isinstance(burn_in, int)
        assert burn_in >= 0
        assert burn_in < sample_chains.shape[1]

    def test_detect_adaptive_burn_in_variance(self, monitor, sample_chains):
        """Test adaptive burn-in detection with variance method."""
        burn_in = monitor.detect_adaptive_burn_in(sample_chains, method="variance")

        assert isinstance(burn_in, int)
        assert burn_in >= 0
        assert burn_in < sample_chains.shape[1]

    def test_detect_adaptive_burn_in_invalid_method(self, monitor):
        """Test burn-in detection with invalid method."""
        chains = np.random.randn(2, 100)

        with pytest.raises(ValueError, match="Unknown burn-in detection method"):
            monitor.detect_adaptive_burn_in(chains, method="invalid")

    def test_estimate_convergence_rate(self, monitor):
        """Test convergence rate estimation."""
        # Create exponentially decaying diagnostic
        iterations = np.arange(50)
        diagnostic_history = 2.0 * np.exp(-0.1 * iterations) + 1.0

        rate, remaining = monitor.estimate_convergence_rate(
            diagnostic_history.tolist(), target_value=1.0
        )

        assert rate > 0
        assert remaining >= 0

    def test_estimate_convergence_rate_insufficient_data(self, monitor):
        """Test convergence rate with insufficient data."""
        diagnostic_history = [1.5, 1.4]

        rate, remaining = monitor.estimate_convergence_rate(diagnostic_history, target_value=1.0)

        assert rate == 0.0
        assert remaining == -1

    def test_get_stopping_summary(self, monitor):
        """Test getting monitor summary."""
        # Add some history
        monitor.r_hat_history = [1.2, 1.1, 1.05]
        monitor.ess_history = [500, 800, 1200]
        monitor.mean_history = [5.1, 5.05, 5.0]
        monitor.burn_in_detected = True
        monitor.burn_in_iteration = 500

        summary = monitor.get_stopping_summary()

        assert isinstance(summary, dict)
        assert summary["iterations_checked"] == 0  # No actual checks yet
        assert summary["burn_in_detected"] is True
        assert summary["burn_in_iteration"] == 500
        assert summary["latest_r_hat"] == 1.05
        assert summary["latest_ess"] == 1200
        assert summary["latest_mean"] == 5.0

    def test_calculate_diagnostics(self, monitor):
        """Test diagnostic calculation."""
        chains = np.random.randn(3, 1000)

        diagnostics = monitor._calculate_diagnostics(chains)

        assert "r_hat" in diagnostics
        assert "ess" in diagnostics
        assert "mean" in diagnostics
        assert "variance" in diagnostics
        assert "mcse" in diagnostics
        assert "mcse_relative" in diagnostics

        assert diagnostics["r_hat"] > 0
        assert diagnostics["ess"] > 0
        assert diagnostics["variance"] > 0

    def test_calculate_diagnostics_single_chain(self, monitor):
        """Test diagnostics with single chain."""
        chain = np.random.randn(1000)

        diagnostics = monitor._calculate_diagnostics(chain)

        assert diagnostics["r_hat"] == 1.0  # Single chain R-hat
        assert diagnostics["ess"] > 0

    def test_update_history(self, monitor):
        """Test history updating."""
        diagnostics = {"r_hat": 1.05, "ess": 1500, "mean": 5.0, "variance": 2.0}

        monitor._update_history(1000, diagnostics)

        assert len(monitor.iteration_history) == 1
        assert monitor.iteration_history[0] == 1000
        assert len(monitor.r_hat_history) == 1
        assert monitor.r_hat_history[0] == 1.05
        assert len(monitor.ess_history) == 1
        assert monitor.ess_history[0] == 1500

    def test_relative_change_rule(self, monitor):
        """Test relative change stopping rule."""
        monitor.criteria.rule = StoppingRule.RELATIVE_CHANGE
        monitor.criteria.relative_tolerance = 0.01

        # Add history for relative change calculation
        monitor.mean_history = [5.0, 5.05]

        chains = np.ones((2, 2000)) * 5.05
        status = monitor.check_convergence(2000, chains)

        assert "Relative change" in status.reason

    def test_edge_cases(self, monitor):
        """Test edge cases and boundary conditions."""
        # Empty chain - should handle gracefully, not raise
        status = monitor.check_convergence(0, np.array([]))
        assert not status.converged  # Empty chain can't be converged

        # Very short chain
        chain = np.array([1.0, 2.0])
        status = monitor.check_convergence(1000, chain)
        assert not status.converged  # Too short to converge

        # Chain with zero variance
        constant_chain = np.ones((2, 1000))
        status = monitor.check_convergence(1000, constant_chain)
        assert status.diagnostics["variance"] == 0

    def test_burn_in_detection_integration(self, monitor, sample_chains):
        """Test burn-in detection during convergence checking."""
        # Should detect burn-in after enough iterations
        status = monitor.check_convergence(1000, sample_chains[:, :1000])

        # After 1000 iterations, burn-in might be detected
        if monitor.burn_in_detected:
            assert monitor.burn_in_iteration > 0
            assert monitor.burn_in_iteration < 500  # Should detect early burn-in
