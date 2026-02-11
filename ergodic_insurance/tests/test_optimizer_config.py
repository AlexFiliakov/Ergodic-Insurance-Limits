"""Tests for BusinessOptimizerConfig and DecisionEngineConfig Pydantic models.

Validates that the dataclass-to-Pydantic conversion (Issue #471) preserves
default values, enables field validation, and supports serialization round-trips.
"""

from pydantic import ValidationError
import pytest

from ergodic_insurance.config.optimizer import (
    BusinessOptimizerConfig,
    DecisionEngineConfig,
)


class TestBusinessOptimizerConfigDefaults:
    """Verify all default values are preserved after Pydantic conversion."""

    def test_defaults(self):
        cfg = BusinessOptimizerConfig()
        assert cfg.base_roe == 0.15
        assert cfg.protection_benefit_factor == 0.05
        assert cfg.roe_noise_std == 0.1
        assert cfg.base_bankruptcy_risk == 0.02
        assert cfg.max_risk_reduction == 0.015
        assert cfg.premium_burden_risk_factor == 0.5
        assert cfg.time_risk_constant == 20.0
        assert cfg.base_growth_rate == 0.10
        assert cfg.growth_boost_factor == 0.03
        assert cfg.premium_drag_factor == 0.5
        assert cfg.asset_growth_factor == 0.8
        assert cfg.equity_growth_factor == 1.1
        assert cfg.risk_transfer_benefit_rate == 0.05
        assert cfg.risk_reduction_value == 0.03
        assert cfg.stability_value == 0.02
        assert cfg.growth_enablement_value == 0.03
        assert cfg.assumed_volatility == 0.20
        assert cfg.volatility_reduction_factor == 0.05
        assert cfg.min_volatility == 0.05
        assert cfg.seed == 42

    def test_custom_overrides(self):
        cfg = BusinessOptimizerConfig(
            base_roe=0.20,
            seed=123,
            time_risk_constant=30.0,
            equity_growth_factor=1.5,
        )
        assert cfg.base_roe == 0.20
        assert cfg.seed == 123
        assert cfg.time_risk_constant == 30.0
        assert cfg.equity_growth_factor == 1.5
        # Unchanged defaults
        assert cfg.base_growth_rate == 0.10


class TestBusinessOptimizerConfigValidation:
    """Verify Pydantic field constraints reject invalid values."""

    @pytest.mark.parametrize(
        "field_name, bad_value",
        [
            ("base_roe", -0.1),
            ("base_roe", 1.5),
            ("protection_benefit_factor", -0.01),
            ("roe_noise_std", -0.5),
            ("base_bankruptcy_risk", 1.1),
            ("max_risk_reduction", -0.001),
            ("premium_burden_risk_factor", -1.0),
            ("time_risk_constant", 0.0),
            ("time_risk_constant", -5.0),
            ("base_growth_rate", -0.1),
            ("base_growth_rate", 1.01),
            ("growth_boost_factor", -0.01),
            ("premium_drag_factor", -0.1),
            ("asset_growth_factor", -0.5),
            ("equity_growth_factor", -0.1),
            ("risk_transfer_benefit_rate", -0.01),
            ("risk_transfer_benefit_rate", 1.1),
            ("risk_reduction_value", -0.01),
            ("stability_value", 1.5),
            ("growth_enablement_value", -0.001),
            ("assumed_volatility", 0.0),
            ("assumed_volatility", 1.5),
            ("volatility_reduction_factor", -0.1),
            ("min_volatility", 0.0),
            ("min_volatility", 1.5),
            ("seed", -1),
        ],
    )
    def test_rejects_out_of_range(self, field_name, bad_value):
        with pytest.raises(ValidationError):
            BusinessOptimizerConfig(**{field_name: bad_value})

    @pytest.mark.parametrize(
        "field_name, good_value",
        [
            ("base_roe", 0.0),
            ("base_roe", 1.0),
            ("time_risk_constant", 0.001),
            ("premium_burden_risk_factor", 5.0),
            ("asset_growth_factor", 2.0),
            ("equity_growth_factor", 0.0),
            ("seed", 0),
        ],
    )
    def test_accepts_boundary_values(self, field_name, good_value):
        cfg = BusinessOptimizerConfig(**{field_name: good_value})
        assert getattr(cfg, field_name) == good_value


class TestDecisionEngineConfigDefaults:
    """Verify all default values are preserved after Pydantic conversion."""

    def test_defaults(self):
        cfg = DecisionEngineConfig()
        assert cfg.base_growth_rate == 0.08
        assert cfg.volatility_reduction_factor == 0.3
        assert cfg.max_volatility_reduction == 0.15
        assert cfg.growth_benefit_factor == 0.5
        assert cfg.loss_cv == 0.5
        assert cfg.default_optimization_weights == {
            "growth": 0.4,
            "risk": 0.4,
            "cost": 0.2,
        }
        assert cfg.layer_attachment_thresholds == (5_000_000, 25_000_000)

    def test_custom_overrides(self):
        cfg = DecisionEngineConfig(
            loss_cv=0.8,
            default_optimization_weights={"growth": 0.6, "risk": 0.3, "cost": 0.1},
            layer_attachment_thresholds=(3_000_000, 20_000_000),
        )
        assert cfg.loss_cv == 0.8
        assert cfg.default_optimization_weights == {
            "growth": 0.6,
            "risk": 0.3,
            "cost": 0.1,
        }
        assert cfg.layer_attachment_thresholds == (3_000_000, 20_000_000)


class TestDecisionEngineConfigValidation:
    """Verify Pydantic field constraints reject invalid values."""

    @pytest.mark.parametrize(
        "field_name, bad_value",
        [
            ("base_growth_rate", -0.1),
            ("base_growth_rate", 1.5),
            ("volatility_reduction_factor", -0.01),
            ("max_volatility_reduction", 1.1),
            ("growth_benefit_factor", -0.5),
            ("loss_cv", 0.0),
            ("loss_cv", -1.0),
        ],
    )
    def test_rejects_out_of_range(self, field_name, bad_value):
        with pytest.raises(ValidationError):
            DecisionEngineConfig(**{field_name: bad_value})

    @pytest.mark.parametrize(
        "field_name, good_value",
        [
            ("base_growth_rate", 0.0),
            ("base_growth_rate", 1.0),
            ("volatility_reduction_factor", 0.0),
            ("max_volatility_reduction", 0.0),
            ("growth_benefit_factor", 0.0),
            ("loss_cv", 0.001),
        ],
    )
    def test_accepts_boundary_values(self, field_name, good_value):
        cfg = DecisionEngineConfig(**{field_name: good_value})
        assert getattr(cfg, field_name) == good_value


class TestSerializationRoundTrip:
    """Verify .model_dump() and .model_validate() work correctly."""

    def test_business_optimizer_config_roundtrip(self):
        original = BusinessOptimizerConfig(base_roe=0.20, seed=99)
        dumped = original.model_dump()
        restored = BusinessOptimizerConfig.model_validate(dumped)
        assert restored == original
        assert restored.base_roe == 0.20
        assert restored.seed == 99

    def test_decision_engine_config_roundtrip(self):
        original = DecisionEngineConfig(
            loss_cv=0.75,
            layer_attachment_thresholds=(1_000_000, 10_000_000),
        )
        dumped = original.model_dump()
        restored = DecisionEngineConfig.model_validate(dumped)
        assert restored == original
        assert restored.loss_cv == 0.75
        assert restored.layer_attachment_thresholds == (1_000_000, 10_000_000)

    def test_business_optimizer_config_model_dump_keys(self):
        cfg = BusinessOptimizerConfig()
        dumped = cfg.model_dump()
        assert "base_roe" in dumped
        assert "seed" in dumped
        assert len(dumped) == 20  # All 20 fields present

    def test_decision_engine_config_model_dump_keys(self):
        cfg = DecisionEngineConfig()
        dumped = cfg.model_dump()
        assert "loss_cv" in dumped
        assert "default_optimization_weights" in dumped
        assert "layer_attachment_thresholds" in dumped
        assert len(dumped) == 7  # All 7 fields present

    def test_model_json_schema_generation(self):
        """Verify JSON schema can be generated for API documentation."""
        boc_schema = BusinessOptimizerConfig.model_json_schema()
        assert "properties" in boc_schema
        assert "base_roe" in boc_schema["properties"]

        dec_schema = DecisionEngineConfig.model_json_schema()
        assert "properties" in dec_schema
        assert "loss_cv" in dec_schema["properties"]


class TestDefaultFactoryIsolation:
    """Verify mutable defaults are properly isolated between instances."""

    def test_dict_default_isolation(self):
        """Each instance should get its own copy of the default dict."""
        cfg1 = DecisionEngineConfig()
        cfg2 = DecisionEngineConfig()
        assert cfg1.default_optimization_weights is not cfg2.default_optimization_weights
        assert cfg1.default_optimization_weights == cfg2.default_optimization_weights
