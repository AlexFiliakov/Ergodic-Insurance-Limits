# Configuration Migration Task Tracker

## Quick Start Commands
```bash
# Create feature branch
git checkout -b feature/config-migration

# Run migration tools (after Phase 1)
python scripts/migrate_configs.py --validate

# Test backward compatibility (after Phase 2)
pytest tests/test_config_compat.py -v

# Run full validation (after Phase 4)
python scripts/validate_migration.py --full
```

## Phase 1: Foundation & Tools (Days 1-5)

### Day 1: Setup New Structure
```bash
# Create directory structure
mkdir -p ergodic_insurance/data/config/{profiles,modules,presets}
mkdir -p ergodic_insurance/data/config/profiles/custom
mkdir -p ergodic_insurance/scripts
```

- [ ] Create directories
- [ ] Add .gitignore for custom profiles
- [ ] Create README in each directory explaining purpose
- [ ] Commit: "feat: create new config directory structure"

### Day 2: Migration Tools
```python
# ergodic_insurance/src/config_migrator.py
class ConfigMigrator:
    def __init__(self):
        self.legacy_dir = Path("data/parameters")
        self.new_dir = Path("data/config")

    def convert_baseline(self) -> Dict:
        """Convert baseline.yaml to new format"""

    def extract_modules(self) -> None:
        """Extract module configs from legacy files"""

    def create_presets(self) -> None:
        """Generate preset libraries"""
```

- [ ] Implement ConfigMigrator class
- [ ] Write unit tests for migrator
- [ ] Create CLI script for migration
- [ ] Test with baseline.yaml
- [ ] Commit: "feat: implement config migration tools"

### Day 3: New Configuration Models
```python
# ergodic_insurance/src/config_v2.py
from pydantic import BaseModel, Field

class ProfileMetadata(BaseModel):
    name: str
    description: str
    extends: Optional[str] = None
    includes: List[str] = []
    presets: Dict[str, str] = {}

class ConfigV2(BaseModel):
    """Unified configuration model"""
    profile: ProfileMetadata
    manufacturer: ManufacturerConfig
    insurance: InsuranceConfig
    simulation: SimulationConfig
    # ... other sections
```

- [ ] Design new Pydantic models
- [ ] Add validation methods
- [ ] Create factory methods
- [ ] Write comprehensive tests
- [ ] Commit: "feat: create new configuration models"

### Day 4-5: Convert Legacy Configs
- [ ] Run migrator on all 12 YAML files
- [ ] Validate converted configs
- [ ] Create initial profiles (default, conservative, aggressive)
- [ ] Extract and organize modules
- [ ] Generate preset libraries
- [ ] Document conversion mappings
- [ ] Commit: "feat: convert legacy configurations"

## Phase 2: Core Implementation (Days 6-10)

### Day 6: ConfigManager Implementation
```python
# ergodic_insurance/src/config_manager.py
class ConfigManager:
    def load_profile(self, name: str = "default") -> ConfigV2:
        """Load configuration profile with all dependencies"""

    def with_preset(self, preset_type: str, preset_name: str):
        """Apply a preset to current config"""
```

- [ ] Implement core ConfigManager
- [ ] Add profile inheritance
- [ ] Add module inclusion
- [ ] Add preset application
- [ ] Commit: "feat: implement ConfigManager"

### Day 7: Caching and Performance
- [ ] Add LRU cache for loaded configs
- [ ] Implement lazy loading for modules
- [ ] Add config validation caching
- [ ] Performance benchmarks
- [ ] Commit: "perf: add config caching"

### Day 8: Backward Compatibility
```python
# ergodic_insurance/src/config_compat.py
class LegacyConfigAdapter:
    """Adapter to support old ConfigLoader interface"""

    def load(self, config_name: str, **kwargs) -> Config:
        # Map old interface to new ConfigManager
        warnings.warn("ConfigLoader is deprecated, use ConfigManager",
                     DeprecationWarning)
```

- [ ] Create compatibility adapter
- [ ] Update ConfigLoader to use adapter
- [ ] Add deprecation warnings
- [ ] Test with existing code
- [ ] Commit: "feat: add backward compatibility layer"

### Day 9-10: Testing New System
- [ ] Unit tests for ConfigManager
- [ ] Integration tests for profile loading
- [ ] Test preset application
- [ ] Test inheritance and includes
- [ ] Benchmark performance
- [ ] Commit: "test: comprehensive config system tests"

## Phase 3: Code Migration (Days 11-15)

### Day 11: Core Module Updates
Update these files to use ConfigManager:
- [ ] `src/manufacturer.py` - Use new config structure
- [ ] `src/simulation.py` - Replace ConfigLoader
- [ ] `src/monte_carlo.py` - Update config usage
- [ ] `src/decision_engine.py` - New pricing scenario loading
- [ ] Commit: "refactor: update core modules for new config"

### Day 12: Insurance Module Updates
- [ ] `src/insurance.py` - Use insurance module config
- [ ] `src/insurance_program.py` - Load from presets
- [ ] `src/loss_distributions.py` - Use loss module config
- [ ] `src/claim_generator.py` - Update config references
- [ ] Commit: "refactor: update insurance modules"

### Day 13: Test Suite Updates
- [ ] Update `conftest.py` with new fixtures
- [ ] Update `test_config.py` for new system
- [ ] Update `test_manufacturer.py`
- [ ] Update `test_simulation.py`
- [ ] Update `test_insurance*.py` files
- [ ] Commit: "test: update test suite for new config"

### Day 14: Example Scripts
- [ ] Update all 4 example scripts
- [ ] Create `demo_config_system.py`
- [ ] Test all examples
- [ ] Commit: "docs: update example scripts"

### Day 15: Integration Testing
- [ ] Run full test suite
- [ ] Check coverage remains at 100%
- [ ] Run performance benchmarks
- [ ] Fix any issues
- [ ] Commit: "test: full integration validation"

## Phase 4: Documentation & Cleanup (Days 16-20)

### Day 16: Jupyter Notebooks
- [ ] Update `11_pareto_analysis.ipynb`
- [ ] Review all notebooks for config usage
- [ ] Create `12_configuration_guide.ipynb`
- [ ] Test all notebooks
- [ ] Commit: "docs: update Jupyter notebooks"

### Day 17: Documentation Updates
- [ ] Update README.md
- [ ] Update CLAUDE.md
- [ ] Create migration guide
- [ ] Update Sphinx docs
- [ ] Commit: "docs: comprehensive documentation update"

### Day 18: User Guide
- [ ] Write configuration best practices
- [ ] Create preset creation guide
- [ ] Document common patterns
- [ ] Add troubleshooting section
- [ ] Commit: "docs: add user guides"

### Day 19: Cleanup
- [ ] Remove deprecated code (optional)
- [ ] Archive legacy configs
- [ ] Clean up imports
- [ ] Update .gitignore
- [ ] Commit: "chore: cleanup legacy code"

### Day 20: Final Validation
- [ ] Run complete test suite
- [ ] Run all notebooks
- [ ] Test all examples
- [ ] Performance benchmarks
- [ ] Create release notes
- [ ] Commit: "chore: final migration validation"

## Validation Scripts

### Quick Validation
```bash
# After each phase
python -c "from ergodic_insurance.src.config_manager import ConfigManager;
           cm = ConfigManager();
           config = cm.load_profile('default');
           print('✓ Config loads successfully')"
```

### Full Test Suite
```bash
# Run after major changes
pytest ergodic_insurance/tests/ -v --cov=ergodic_insurance --cov-report=term-missing
```

### Notebook Validation
```bash
# Test all notebooks
jupyter nbconvert --execute ergodic_insurance/notebooks/*.ipynb
```

## Progress Tracking

### Week 1 Milestones
- [ ] New directory structure created
- [ ] Migration tools functional
- [ ] All configs converted
- [ ] New models defined

### Week 2 Milestones
- [ ] ConfigManager working
- [ ] Backward compatibility verified
- [ ] Performance acceptable
- [ ] All tests passing

### Week 3 Milestones
- [ ] All code migrated
- [ ] Examples updated
- [ ] 100% test coverage
- [ ] Integration validated

### Week 4 Milestones
- [ ] Documentation complete
- [ ] Notebooks updated
- [ ] Cleanup done
- [ ] Ready for merge

## Emergency Procedures

### If Tests Fail
1. Check backward compatibility layer
2. Verify config conversion accuracy
3. Review deprecation warnings
4. Rollback specific module if needed

### If Performance Degrades
1. Check caching implementation
2. Profile config loading
3. Optimize merge operations
4. Consider lazy loading

### If Notebooks Break
1. Use compatibility mode temporarily
2. Update imports incrementally
3. Test with simplified configs
4. Document workarounds

## Definition of Done
- [ ] All 4 phases complete
- [ ] 100% test coverage maintained
- [ ] All notebooks run without errors
- [ ] Documentation fully updated
- [ ] Performance benchmarks pass
- [ ] No critical bugs
- [ ] Deprecation warnings added
- [ ] Migration guide published
- [ ] Code review approved
- [ ] Merged to main branch

## Post-Migration Tasks
- [ ] Monitor for issues (1 week)
- [ ] Gather user feedback
- [ ] Remove compatibility layer (after 1 month)
- [ ] Archive legacy code
- [ ] Update training materials
- [ ] Celebrate! 🎉
