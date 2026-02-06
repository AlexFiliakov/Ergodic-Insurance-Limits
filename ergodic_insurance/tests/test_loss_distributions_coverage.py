"""Coverage tests for loss_distributions.py targeting specific uncovered lines.

Missing lines: 288, 294-298, 302-306, 332, 339, 342, 345, 351, 361-363,
430, 486, 922, 973, 975, 1061
"""

import numpy as np
import pytest

from ergodic_insurance.loss_distributions import LossData, LossEvent, ManufacturingLossGenerator


class TestLossEventPostInit:
    """Tests for LossEvent.__post_init__ (lines 288)."""

    def test_timestamp_alternative_name(self):
        """Line 288: timestamp sets time when time is default."""
        event = LossEvent(amount=1000, timestamp=5.5)
        assert event.time == 5.5

    def test_event_type_alternative_name(self):
        """Lines 289-290: event_type sets loss_type when loss_type is default."""
        event = LossEvent(amount=1000, event_type="catastrophic")
        assert event.loss_type == "catastrophic"

    def test_both_alternative_names(self):
        """Both timestamp and event_type override defaults."""
        event = LossEvent(amount=500, timestamp=3.0, event_type="large")
        assert event.time == 3.0
        assert event.loss_type == "large"


class TestLossEventOrdering:
    """Tests for LossEvent ordering methods (lines 294-298, 302-306)."""

    def test_le_with_number(self):
        """Lines 294-295: __le__ comparison with int/float."""
        event = LossEvent(amount=1000)
        assert event <= 1500
        assert event <= 1000
        assert not event <= 500  # pylint: disable=unnecessary-negation

    def test_le_with_loss_event(self):
        """Lines 296-297: __le__ comparison with another LossEvent."""
        event1 = LossEvent(amount=1000)
        event2 = LossEvent(amount=2000)
        assert event1 <= event2
        assert not event2 <= event1  # pylint: disable=unnecessary-negation

    def test_le_with_unsupported_type(self):
        """Line 298: __le__ returns NotImplemented for unsupported types."""
        event = LossEvent(amount=1000)
        result = event.__le__("not_a_number")  # pylint: disable=unnecessary-dunder-call
        assert result is NotImplemented

    def test_lt_with_number(self):
        """Lines 302-303: __lt__ comparison with int/float."""
        event = LossEvent(amount=1000)
        assert event < 1500
        assert not event < 1000  # pylint: disable=unnecessary-negation
        assert not event < 500  # pylint: disable=unnecessary-negation

    def test_lt_with_loss_event(self):
        """Lines 304-305: __lt__ comparison with another LossEvent."""
        event1 = LossEvent(amount=1000)
        event2 = LossEvent(amount=2000)
        assert event1 < event2
        assert not event2 < event1  # pylint: disable=unnecessary-negation

    def test_lt_with_unsupported_type(self):
        """Line 306: __lt__ returns NotImplemented for unsupported types."""
        event = LossEvent(amount=1000)
        result = event.__lt__("not_a_number")  # pylint: disable=unnecessary-dunder-call
        assert result is NotImplemented


class TestLossDataValidation:
    """Tests for LossData.validate() (lines 332, 339, 342, 345, 351)."""

    def test_mismatched_timestamps_and_amounts(self):
        """Line 332: timestamps length != loss_amounts length returns False."""
        data = LossData(
            timestamps=np.array([1.0, 2.0]),
            loss_amounts=np.array([100.0]),
        )
        assert data.validate() is False

    def test_mismatched_loss_types_length(self):
        """Line 339: loss_types length mismatch returns False."""
        data = LossData(
            timestamps=np.array([1.0, 2.0]),
            loss_amounts=np.array([100.0, 200.0]),
            loss_types=["attritional"],  # Only 1 element instead of 2
        )
        assert data.validate() is False

    def test_mismatched_claim_ids_length(self):
        """Line 342: claim_ids length mismatch returns False."""
        data = LossData(
            timestamps=np.array([1.0, 2.0]),
            loss_amounts=np.array([100.0, 200.0]),
            claim_ids=["claim1"],  # Only 1 element instead of 2
        )
        assert data.validate() is False

    def test_mismatched_development_factors_length(self):
        """Line 345: development_factors length mismatch returns False."""
        data = LossData(
            timestamps=np.array([1.0, 2.0]),
            loss_amounts=np.array([100.0, 200.0]),
            development_factors=np.array([1.0]),  # Only 1 element instead of 2
        )
        assert data.validate() is False

    def test_negative_loss_amounts(self):
        """Lines 348-351: Negative loss amounts return False."""
        data = LossData(
            timestamps=np.array([1.0, 2.0]),
            loss_amounts=np.array([100.0, -50.0]),
        )
        assert data.validate() is False

    def test_negative_timestamps(self):
        """Lines 349-350: Negative timestamps return False."""
        data = LossData(
            timestamps=np.array([-1.0, 2.0]),
            loss_amounts=np.array([100.0, 200.0]),
        )
        assert data.validate() is False

    def test_valid_data_returns_true(self):
        """Valid data returns True."""
        data = LossData(
            timestamps=np.array([1.0, 2.0]),
            loss_amounts=np.array([100.0, 200.0]),
            loss_types=["attritional", "large"],
            claim_ids=["c1", "c2"],
        )
        assert data.validate() is True


class TestLossDataToErgodicFormat:
    """Tests for LossData.to_ergodic_format() (lines 361-363)."""

    def test_to_ergodic_format(self):
        """Lines 361-363: Converts to ErgodicData format."""
        data = LossData(
            timestamps=np.array([1.0, 2.0, 3.0]),
            loss_amounts=np.array([100.0, 200.0, 300.0]),
            metadata={"test": "value"},
        )
        ergodic_data = data.to_ergodic_format()
        assert len(ergodic_data.time_series) == 3
        assert len(ergodic_data.values) == 3


class TestLossDataFromLossEvents:
    """Tests for LossData.from_loss_events() (line 430)."""

    def test_empty_events_returns_empty_data(self):
        """Line 430: Empty list returns default LossData."""
        data = LossData.from_loss_events([])
        assert len(data.timestamps) == 0
        assert len(data.loss_amounts) == 0

    def test_from_loss_events(self):
        """Convert LossEvent list to LossData."""
        events = [
            LossEvent(amount=1000, time=1.5, loss_type="attritional"),
            LossEvent(amount=5000, time=0.5, loss_type="large"),
            LossEvent(amount=2000, time=3.0, loss_type="catastrophic"),
        ]
        data = LossData.from_loss_events(events)
        assert len(data.timestamps) == 3
        # Should be sorted by time
        assert data.timestamps[0] == 0.5
        assert data.timestamps[1] == 1.5
        assert data.timestamps[2] == 3.0


class TestLossDataCalculateStatistics:
    """Tests for LossData.calculate_statistics() (line 486)."""

    def test_empty_data_statistics(self):
        """Line 486: Empty loss data returns zero statistics."""
        data = LossData()
        stats = data.calculate_statistics()
        assert stats["count"] == 0
        assert stats["total"] == 0.0
        assert stats["mean"] == 0.0


class TestManufacturingLossGeneratorReseed:
    """Tests for ManufacturingLossGenerator.reseed() (line 922)."""

    def test_reseed_with_gpd_generator(self):
        """Line 922: Reseed also reseeds GPD generator when present."""
        gen = ManufacturingLossGenerator(
            attritional_params={"base_frequency": 5.0, "severity_mean": 50000, "severity_cv": 0.8},
            large_params={"base_frequency": 0.5, "severity_mean": 500000, "severity_cv": 1.5},
            catastrophic_params={
                "base_frequency": 0.05,
                "severity_alpha": 2.5,
                "severity_xm": 1000000,
            },
            extreme_params={
                "threshold_percentile": 0.95,
                "severity_shape": 0.3,
                "severity_scale": 100000,
            },
            seed=42,
        )
        gen.reseed(123)
        # GPD generator should be reseeded without error
        losses1, _ = gen.generate_losses(duration=1.0, revenue=10_000_000)
        gen.reseed(123)
        losses2, _ = gen.generate_losses(duration=1.0, revenue=10_000_000)
        # Same seed should produce same number of losses
        assert len(losses1) == len(losses2)


class TestCreateSimpleValidation:
    """Tests for ManufacturingLossGenerator.create_simple() validation (lines 973, 975)."""

    def test_negative_severity_mean_raises(self):
        """Line 973: Negative severity_mean raises ValueError."""
        with pytest.raises(ValueError, match="severity_mean must be positive"):
            ManufacturingLossGenerator.create_simple(
                frequency=0.1, severity_mean=-100, severity_std=50
            )

    def test_negative_severity_std_raises(self):
        """Line 975: Negative severity_std raises ValueError."""
        with pytest.raises(ValueError, match="severity_std must be non-negative"):
            ManufacturingLossGenerator.create_simple(
                frequency=0.1, severity_mean=1000, severity_std=-50
            )


class TestExtremeValueTransformation:
    """Tests for extreme value loss transformation (line 1061)."""

    def test_extreme_value_transformation_removes_from_categories(self):
        """Line 1061: Extreme losses are removed from original categories."""
        gen = ManufacturingLossGenerator(
            attritional_params={
                "base_frequency": 50.0,
                "severity_mean": 100000,
                "severity_cv": 2.0,
            },
            large_params={"base_frequency": 5.0, "severity_mean": 500000, "severity_cv": 2.0},
            catastrophic_params={
                "base_frequency": 1.0,
                "severity_alpha": 2.0,
                "severity_xm": 200000,
            },
            extreme_params={
                "threshold_percentile": 0.50,  # Low threshold to trigger
                "severity_shape": 0.3,
                "severity_scale": 100000,
            },
            seed=42,
        )
        losses, stats = gen.generate_losses(
            duration=1.0, revenue=50_000_000, include_catastrophic=True
        )
        # Should not crash and returns losses
        assert isinstance(losses, list)
        assert isinstance(stats, dict)
