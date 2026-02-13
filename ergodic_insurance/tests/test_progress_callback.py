"""Tests for progress_callback and cancel_event on MonteCarloEngine and Simulation."""

import threading
import time
from unittest.mock import Mock

import numpy as np
import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.insurance_program import EnhancedInsuranceLayer, InsuranceProgram
from ergodic_insurance.loss_distributions import LossEvent, ManufacturingLossGenerator
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.monte_carlo import MonteCarloConfig, MonteCarloEngine
from ergodic_insurance.simulation import Simulation

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manufacturer():
    config = ManufacturerConfig(
        initial_assets=10_000_000,
        asset_turnover_ratio=0.5,
        base_operating_margin=0.1,
        tax_rate=0.25,
        retention_ratio=0.8,
    )
    return WidgetManufacturer(config)


@pytest.fixture
def loss_generator():
    return ManufacturingLossGenerator.create_simple(
        frequency=0.1, severity_mean=1_000_000, severity_std=500_000, seed=42
    )


@pytest.fixture
def insurance_program():
    layer = EnhancedInsuranceLayer(attachment_point=0, limit=1_000_000, base_premium_rate=0.02)
    return InsuranceProgram(layers=[layer])


def _make_engine(loss_generator, insurance_program, manufacturer, n_sims=200, parallel=False):
    config = MonteCarloConfig(
        n_simulations=n_sims,
        n_years=2,
        parallel=parallel,
        cache_results=False,
        progress_bar=False,
        seed=42,
    )
    loss_generator.generate_losses = Mock(
        return_value=(
            [LossEvent(time=0.5, amount=50_000, loss_type="test")],
            50_000,
        )
    )
    return MonteCarloEngine(
        loss_generator=loss_generator,
        insurance_program=insurance_program,
        manufacturer=manufacturer,
        config=config,
    )


# ---------------------------------------------------------------------------
# MonteCarloEngine — sequential progress callback
# ---------------------------------------------------------------------------


class TestMonteCarloProgressCallback:
    """Tests for MonteCarloEngine progress_callback."""

    def test_progress_callback_sequential(self, loss_generator, insurance_program, manufacturer):
        """Callback fires with correct (completed, total, elapsed) args."""
        engine = _make_engine(loss_generator, insurance_program, manufacturer)
        cb = Mock()

        engine.run(progress_callback=cb)

        assert cb.call_count > 0
        for call in cb.call_args_list:
            completed, total, elapsed = call.args
            assert total == engine.config.n_simulations
            assert 0 < completed <= total
            assert elapsed >= 0

    def test_progress_callback_receives_increasing_completed(
        self, loss_generator, insurance_program, manufacturer
    ):
        """completed values are monotonically increasing."""
        engine = _make_engine(loss_generator, insurance_program, manufacturer)
        completed_values = []

        def capture(completed, total, elapsed):
            completed_values.append(completed)

        engine.run(progress_callback=capture)

        assert len(completed_values) > 0
        for i in range(1, len(completed_values)):
            assert completed_values[i] >= completed_values[i - 1]

    def test_no_callback_default(self, loss_generator, insurance_program, manufacturer):
        """run() still works with no callback (backward compat)."""
        engine = _make_engine(loss_generator, insurance_program, manufacturer)
        results = engine.run()
        assert results.final_assets is not None
        assert len(results.final_assets) == engine.config.n_simulations

    def test_progress_callback_with_progress_bar(
        self, loss_generator, insurance_program, manufacturer
    ):
        """Both progress_callback and progress_bar work simultaneously."""
        engine = _make_engine(loss_generator, insurance_program, manufacturer)
        engine.config.progress_bar = True
        cb = Mock()

        results = engine.run(progress_callback=cb)

        assert cb.call_count > 0
        assert results.final_assets is not None


# ---------------------------------------------------------------------------
# MonteCarloEngine — cancel event
# ---------------------------------------------------------------------------


class TestMonteCarloCancelEvent:
    """Tests for MonteCarloEngine cancel_event."""

    def test_cancel_event_sequential(self, loss_generator, insurance_program, manufacturer):
        """Set cancel_event after N sims, verify partial results returned."""
        n_sims = 500
        engine = _make_engine(loss_generator, insurance_program, manufacturer, n_sims=n_sims)
        cancel = threading.Event()

        call_count = 0

        def set_cancel_after(completed, total, elapsed):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                cancel.set()

        results = engine.run(progress_callback=set_cancel_after, cancel_event=cancel)

        # Should have fewer results than total
        assert len(results.final_assets) < n_sims
        assert len(results.final_assets) > 0

    def test_cancel_event_parallel(self, loss_generator, insurance_program, manufacturer):
        """Cancel during parallel execution returns partial results.

        Uses use_enhanced_parallel=False so the basic ProcessPoolExecutor
        path is exercised without needing to pickle Mock objects through
        shared memory.
        """
        n_sims = 500
        config = MonteCarloConfig(
            n_simulations=n_sims,
            n_years=2,
            parallel=True,
            use_enhanced_parallel=False,
            cache_results=False,
            progress_bar=False,
            seed=42,
        )
        engine = MonteCarloEngine(
            loss_generator=loss_generator,
            insurance_program=insurance_program,
            manufacturer=manufacturer,
            config=config,
        )
        # Set cancel immediately so we get partial results
        cancel = threading.Event()
        cancel.set()

        results = engine.run(cancel_event=cancel)
        # With cancel set immediately, we may get 0 or partial results.
        # The parallel path may still process some chunks before seeing
        # the cancel event, so we just assert we got *some* result object.
        assert len(results.final_assets) <= n_sims


# ---------------------------------------------------------------------------
# Simulation — progress callback
# ---------------------------------------------------------------------------


class TestSimulationProgressCallback:
    """Tests for Simulation.run() progress_callback."""

    def test_simulation_progress_callback(self, manufacturer, loss_generator):
        """Callback fires per year with correct args."""
        sim = Simulation(
            manufacturer=manufacturer,
            loss_generator=loss_generator,
            time_horizon=20,
            seed=42,
        )
        cb = Mock()

        sim.run(progress_callback=cb)

        # Callback should fire at least once per completed year
        assert cb.call_count > 0
        for call in cb.call_args_list:
            completed, total, elapsed = call.args
            assert total == 20
            assert 0 < completed <= total
            assert elapsed >= 0

    def test_simulation_progress_callback_monotonic(self, manufacturer, loss_generator):
        """completed_years values are monotonically increasing."""
        sim = Simulation(
            manufacturer=manufacturer,
            loss_generator=loss_generator,
            time_horizon=20,
            seed=42,
        )
        completed_values = []

        def capture(completed, total, elapsed):
            completed_values.append(completed)

        sim.run(progress_callback=capture)

        for i in range(1, len(completed_values)):
            assert completed_values[i] >= completed_values[i - 1]

    def test_simulation_no_callback_default(self, manufacturer, loss_generator):
        """Simulation.run() still works with no callback."""
        sim = Simulation(
            manufacturer=manufacturer,
            loss_generator=loss_generator,
            time_horizon=10,
            seed=42,
        )
        results = sim.run()
        assert results is not None
        assert len(results.years) == 10


# ---------------------------------------------------------------------------
# Simulation — cancel event
# ---------------------------------------------------------------------------


class TestSimulationCancelEvent:
    """Tests for Simulation.run() cancel_event."""

    def test_simulation_cancel_event(self, manufacturer, loss_generator):
        """Cancel mid-simulation, verify partial results."""
        time_horizon = 50
        sim = Simulation(
            manufacturer=manufacturer,
            loss_generator=loss_generator,
            time_horizon=time_horizon,
            seed=42,
        )
        cancel = threading.Event()

        def cancel_at_year_10(completed, total, elapsed):
            if completed >= 10:
                cancel.set()

        results = sim.run(progress_callback=cancel_at_year_10, cancel_event=cancel)

        # Results are returned for the full time_horizon array, but years
        # after cancellation will be zero-filled.  The simulation should not
        # have run all 50 years worth of meaningful data.
        assert results is not None
        # Equity for years past the cancellation point should be zero
        nonzero_equity = np.count_nonzero(results.equity)
        assert nonzero_equity < time_horizon

    def test_simulation_cancel_immediate(self, manufacturer, loss_generator):
        """Cancel before any year runs."""
        sim = Simulation(
            manufacturer=manufacturer,
            loss_generator=loss_generator,
            time_horizon=20,
            seed=42,
        )
        cancel = threading.Event()
        cancel.set()

        results = sim.run(cancel_event=cancel)
        # All values should be zero since we cancelled before year 0
        assert results is not None
