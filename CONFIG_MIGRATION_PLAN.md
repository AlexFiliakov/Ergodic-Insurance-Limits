# Configuration System Migration Plan

## Executive Summary
Migrate from 12 scattered YAML configuration files to a simplified 3-tier system (profiles/modules/presets) to improve usability, maintainability, and discoverability.

## Migration Timeline
- **Phase 1 (Week 1)**: Foundation & Tools - Create new structure and migration utilities
- **Phase 2 (Week 2)**: Core Implementation - Build new ConfigManager with backward compatibility
- **Phase 3 (Week 3)**: Code Migration - Update all Python code and tests
- **Phase 4 (Week 4)**: Documentation & Cleanup - Update notebooks, docs, and remove legacy code

## Impact Analysis

### Files Requiring Updates
- **14 Python modules** using ConfigLoader or load_config
- **26 Python modules** using config dataclasses directly
- **12 test modules** with configuration dependencies
- **1 Jupyter notebook** with config loading
- **4 example scripts** demonstrating configuration
- **Multiple documentation files** referencing configuration

### Current Dependencies
```
Primary Users:
- decision_engine.py (uses ConfigLoader, pricing scenarios)
- insurance.py, insurance_program.py (load insurance configs)
- simulation.py, monte_carlo.py (use SimulationConfig)
- All test files (create configs for testing)
- All example scripts (demonstrate config usage)
```

## Detailed Task List

### Phase 1: Foundation & Tools (Week 1)

#### 1.1 Create New Directory Structure
- [ ] Create `data/config/profiles/` directory
- [ ] Create `data/config/modules/` directory
- [ ] Create `data/config/presets/` directory
- [ ] Create `data/config/profiles/custom/` for user configs
- [ ] Add `.gitignore` for custom profiles

#### 1.2 Build Migration Tools
- [ ] Create `src/config_migrator.py` with:
  - [ ] `convert_legacy_configs()` - Convert existing YAMLs
  - [ ] `create_profile_from_scenario()` - Generate profiles
  - [ ] `validate_migration()` - Ensure data integrity
  - [ ] `generate_migration_report()` - Document changes
- [ ] Create `tests/test_config_migrator.py`
- [ ] Write migration script `scripts/migrate_configs.py`

#### 1.3 Design New Configuration Schema
- [ ] Create enhanced Pydantic models in `src/config_v2.py`:
  - [ ] `ProfileMetadata` - Profile information
  - [ ] `ConfigV2` - New unified config class
  - [ ] `ModuleConfig` - Base class for modules
  - [ ] `PresetLibrary` - Preset management
- [ ] Create comprehensive validation rules
- [ ] Add schema documentation

### Phase 2: Core Implementation (Week 2)

#### 2.1 Implement ConfigManager
- [ ] Create `src/config_manager.py` with:
  - [ ] `load_profile()` - Main loading method
  - [ ] `with_overrides()` - Dynamic overrides
  - [ ] `with_preset()` - Apply presets
  - [ ] `validate()` - Comprehensive validation
  - [ ] `list_profiles()` - Discovery method
  - [ ] `_deep_merge()` - Config merging logic
  - [ ] `_resolve_includes()` - Module inclusion
  - [ ] `_apply_inheritance()` - Profile extension
- [ ] Add caching mechanism for performance
- [ ] Implement helpful error messages

#### 2.2 Create Default Configurations
- [ ] Convert `baseline.yaml` → `profiles/default.yaml`
- [ ] Convert `conservative.yaml` → `profiles/conservative.yaml`
- [ ] Convert `optimistic.yaml` → `profiles/aggressive.yaml`
- [ ] Extract insurance configs → `modules/insurance.yaml`
- [ ] Extract loss configs → `modules/losses.yaml`
- [ ] Extract simulation configs → `modules/simulation.yaml`
- [ ] Create market presets → `presets/market_conditions.yaml`
- [ ] Create layer presets → `presets/layer_structures.yaml`
- [ ] Create risk presets → `presets/risk_scenarios.yaml`

#### 2.3 Backward Compatibility Layer
- [ ] Create `src/config_compat.py`:
  - [ ] Adapter to support old ConfigLoader interface
  - [ ] Deprecation warnings for old methods
  - [ ] Mapping from old to new structure
- [ ] Update `src/config_loader.py` to use ConfigManager internally
- [ ] Add compatibility tests

### Phase 3: Code Migration (Week 3)

#### 3.1 Update Core Modules
- [ ] Update `src/manufacturer.py`:
  - [ ] Use new config structure
  - [ ] Update constructor signatures
- [ ] Update `src/simulation.py`:
  - [ ] Use ConfigManager
  - [ ] Remove old config loading
- [ ] Update `src/monte_carlo.py`:
  - [ ] Adapt to new config format
- [ ] Update `src/decision_engine.py`:
  - [ ] Replace ConfigLoader with ConfigManager
  - [ ] Update pricing scenario loading
- [ ] Update `src/insurance.py` and `src/insurance_program.py`:
  - [ ] Use new insurance module config
  - [ ] Update layer structure loading

#### 3.2 Update Test Suite
- [ ] Update `tests/conftest.py`:
  - [ ] Create fixtures for new config system
  - [ ] Provide test profiles
- [ ] Update `tests/test_config.py`:
  - [ ] Test new ConfigManager
  - [ ] Test profile loading
  - [ ] Test preset application
- [ ] Update each test file (12 files):
  - [ ] Replace ManufacturerConfig creation
  - [ ] Use test profiles instead of manual config
  - [ ] Update assertions for new structure
- [ ] Create integration tests for migration
- [ ] Ensure 100% test coverage maintained

#### 3.3 Update Example Scripts
- [ ] Update `examples/demo_manufacturer.py`
- [ ] Update `examples/demo_collateral_management.py`
- [ ] Update `examples/demo_claim_development.py`
- [ ] Update `examples/demo_stochastic.py`
- [ ] Add new example: `examples/demo_config_system.py`

### Phase 4: Documentation & Cleanup (Week 4)

#### 4.1 Update Jupyter Notebooks
- [ ] Update `notebooks/11_pareto_analysis.ipynb`:
  - [ ] Use new config loading
  - [ ] Update documentation cells
- [ ] Review all other notebooks for config references
- [ ] Create new notebook: `notebooks/12_configuration_guide.ipynb`

#### 4.2 Update Documentation
- [ ] Update `README.md`:
  - [ ] New configuration system overview
  - [ ] Updated quick start guide
- [ ] Update `CLAUDE.md`:
  - [ ] New config structure documentation
  - [ ] Updated examples
- [ ] Update Sphinx documentation:
  - [ ] `docs/getting_started.rst`
  - [ ] `docs/user_guide/quick_start.rst`
  - [ ] `docs/user_guide/running_analysis.rst`
  - [ ] `docs/api/config.rst`
  - [ ] `docs/api/config_loader.rst`
- [ ] Create migration guide: `docs/migration_guide.md`

#### 4.3 Cleanup & Finalization
- [ ] Remove deprecated code (after grace period)
- [ ] Archive old YAML files to `data/parameters_legacy/`
- [ ] Update `.gitignore` patterns
- [ ] Run full test suite
- [ ] Run performance benchmarks
- [ ] Update CI/CD configuration if needed

## Validation Checklist

### Pre-Migration Validation
- [ ] Backup all existing configurations
- [ ] Document current config usage patterns
- [ ] Create rollback script
- [ ] Test migration tools thoroughly

### During Migration
- [ ] Validate each converted profile loads correctly
- [ ] Ensure all parameters are preserved
- [ ] Test backward compatibility layer
- [ ] Monitor test coverage

### Post-Migration Validation
- [ ] All tests passing (100% coverage maintained)
- [ ] All notebooks run successfully
- [ ] All examples work correctly
- [ ] Documentation is complete and accurate
- [ ] Performance benchmarks meet or exceed current
- [ ] No deprecation warnings in core code

## Rollback Plan

### Immediate Rollback (Phase 1-2)
1. Git revert migration commits
2. Restore original `data/parameters/` directory
3. No code changes needed yet

### Partial Rollback (Phase 3)
1. Keep new system but restore compatibility layer
2. Revert code changes module by module
3. Run compatibility tests

### Full Rollback (Phase 4)
1. Restore from pre-migration backup tag
2. Document lessons learned
3. Plan alternative approach

## Success Metrics
- **Code Reduction**: 30% fewer lines in config management
- **Load Time**: <100ms for full profile load
- **Test Coverage**: Maintain 100%
- **User Feedback**: Positive response from first users
- **Documentation**: Complete migration guide with examples

## Risk Mitigation
- **Risk**: Breaking existing functionality
  - **Mitigation**: Comprehensive backward compatibility layer
- **Risk**: Performance degradation
  - **Mitigation**: Caching and lazy loading
- **Risk**: User confusion
  - **Mitigation**: Clear migration guide and examples
- **Risk**: Test failures
  - **Mitigation**: Gradual migration with continuous testing

## Communication Plan
1. **Week 0**: Announce migration plan to stakeholders
2. **Week 1**: Share foundation progress and tools
3. **Week 2**: Demo new ConfigManager
4. **Week 3**: Request testing from early adopters
5. **Week 4**: Announce completion with guide

## Long-term Benefits
1. **Simplified Onboarding**: New users understand system faster
2. **Reduced Maintenance**: Fewer files to manage
3. **Better Testing**: Easier to create test configurations
4. **Enhanced Features**: Preset system enables rapid experimentation
5. **Improved Documentation**: Self-documenting profiles

## Next Steps
1. Review and approve this plan
2. Create migration branch: `feature/config-migration`
3. Begin Phase 1 implementation
4. Set up weekly progress reviews
