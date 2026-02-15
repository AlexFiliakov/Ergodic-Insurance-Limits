# Claude Code Agent Team Objectives and Structure

## Objectives (ranked by priority)

1. **Prune tautological tests** — tests that always pass regardless of code
   changes. These include tests with no real assertions, tests that only assert
   on mocked return values they themselves set up, tests asserting constants or
   type checks that the type system already enforces, and tests where the
   "expected" value is computed by the same code under test.

2. **Eliminate redundancy** — near-duplicate tests covering the same code paths.
   Find tests that are copy-pasted with trivial variations, tests that exercise
   identical branches through different but equivalent inputs, and parametrizable
   tests that should be collapsed into `@pytest.mark.parametrize`.

3. **Fix slow tests** — tests with unnecessary overhead. Look for tests that
   create expensive fixtures they barely use, tests doing real I/O or network
   calls that should be mocked, tests with `time.sleep()` or busy-wait loops,
   excessive setup/teardown that could be session- or module-scoped, and tests
   that import heavy modules unnecessarily.

4. **Stabilize flaky tests** — tests with intermittent failures. Identify tests
   with timing dependencies, order-dependent tests that rely on shared mutable
   state, tests that depend on filesystem or environment specifics, and tests
   with floating-point comparisons using exact equality instead of
   `pytest.approx()`.

5. **General improvements** — anything else that improves the suite:
   - Add `@pytest.mark.slow` and `@pytest.mark.integration` markers where missing
   - Standardize fixture patterns (prefer factories over complex object construction)
   - Add missing `__str__`/`__repr__` assertions for key domain objects
   - Ensure parametrized tests have descriptive IDs
   - Flag tests with no docstring or unclear intent for review

## Team Structure

Create a team with 4 teammates. Each teammate works on a distinct analysis pass
across the entire test suite. The lead (you) handles the initial survey and
coordinates.

### Phase 0: Survey (Lead only, before spawning teammates)

Before creating any teammates, do the following:

1. Run `pytest --co -q 2>/dev/null | tail -5` to get the total test count.
2. Run `find ergodic_insurance/tests/ -name "test_*.py" -o -name "*_test.py" | head -50` to map
   the directory structure.
3. Run `find ergodic_insurance/tests/ -name "conftest.py"` to map shared fixtures.
4. Run `cat ergodic_insurance/tests/conftest.py` (and any subdirectory conftest files) to understand
   shared fixture patterns.
5. Run `git log --oneline -20 -- ergodic_insurance/tests/` to understand recent test evolution.
6. Try `pytest --co -q 2>/dev/null | wc -l` to get precise count.
7. Build a mapping of test files → source modules they test.
8. Identify the 10 largest test files by line count:
   `find ergodic_insurance/tests/ -name "*.py" | xargs wc -l | sort -rn | head -15`

Save this survey as `ergodic_insurance/tests/REFACTOR_SURVEY.md` — teammates will read it for
context.

### Teammate 1: Tautology Hunter

```
Your name is tautology-hunter. You specialize in finding tests that provide no
real value because they always pass regardless of code correctness.

Read `ergodic_insurance/tests/REFACTOR_SURVEY.md` for context on the test suite structure.

Work through EVERY test file systematically. For each test function, check:

1. **Mock-only assertions**: Does the test mock a dependency and then only assert
   that the mock was called? If it never checks the actual output or side effect
   of the code under test, it's tautological.

2. **Self-fulfilling assertions**: Does the test compute the expected value using
   the same function it's testing? e.g., `assert compute(x) == compute(x)` or
   tests that call the implementation to generate the expected value.

3. **Trivial assertions**: `assert True`, `assert result is not None` (when the
   function always returns something), `assert isinstance(obj, MyClass)` where
   the constructor was just called.

4. **Dead assertions**: Tests where the assertion can never fail because of how
   the test is structured (e.g., asserting inside a `try` block that catches the
   assertion error).

5. **Vacuous parametrize**: Parametrized tests where all parameter sets test the
   same branch.

For each tautological test found, categorize it:
- **DELETE**: No salvageable value. Record the test name and why.
- **REWRITE**: Has the right intent but wrong assertion. Write the corrected version.
- **REVIEW**: Unclear — might be intentionally testing something subtle. Flag for
  human review with your analysis.

Create `ergodic_insurance/tests/TAUTOLOGY_REPORT.md` with your findings organized by file.
Then implement the DELETE and REWRITE changes directly. Mark REVIEW items with
a `# TODO(tautology-review): <reason>` comment.

Send a message to team-lead when you finish each major directory (ergodic_insurance/tests/test_*
files, then ergodic_insurance/tests/integration/).
```

### Teammate 2: Redundancy Analyst

```
Your name is redundancy-analyst. You specialize in finding duplicate and
near-duplicate tests that can be consolidated.

Read `ergodic_insurance/tests/REFACTOR_SURVEY.md` for context on the test suite structure.

Work through EVERY test file systematically. Look for:

1. **Copy-paste duplicates**: Tests in different files (or the same file) that
   are structurally identical with only variable names or values changed. Use
   AST-level comparison, not just string matching — two tests with different
   variable names but identical structure are duplicates.

2. **Parametrize candidates**: Groups of 3+ tests that test the same function
   with different inputs. Collapse these into `@pytest.mark.parametrize` with
   descriptive IDs. Example:
   ```python
   # BEFORE (3 separate tests)
   def test_calc_premium_small(): assert calc_premium(1000) == 50
   def test_calc_premium_medium(): assert calc_premium(5000) == 200
   def test_calc_premium_large(): assert calc_premium(10000) == 350

   # AFTER (1 parametrized test)
   @pytest.mark.parametrize("limit,expected", [
       pytest.param(1000, 50, id="small-limit"),
       pytest.param(5000, 200, id="medium-limit"),
       pytest.param(10000, 350, id="large-limit"),
   ])
   def test_calc_premium(limit, expected):
       assert calc_premium(limit) == expected
   ```

3. **Overlapping coverage**: Tests in different files that exercise the exact same
   code path. This requires reading both the tests AND the source. Two tests that
   call different functions but those functions are just wrappers around the same
   logic — one test is redundant.

4. **Fixture duplication**: Fixtures defined in multiple conftest.py files or test
   files that create the same objects. Consolidate into the appropriate conftest
   scope.

Create `ergodic_insurance/tests/REDUNDANCY_REPORT.md` with your findings. For each cluster of
duplicates, show the original tests and your consolidated replacement.
Then implement the consolidations directly.

Send a message to team-lead when you finish each major directory.
```

### Teammate 3: Performance Optimizer

```
Your name is perf-optimizer. You specialize in making tests run faster without
reducing their value.

Read `ergodic_insurance/tests/REFACTOR_SURVEY.md` for context on the test suite structure.

Your analysis has two phases:

**Phase A: Static analysis (do this first for all files)**

Scan every test file for these performance anti-patterns:

1. **Fixture scope**: Fixtures that create expensive objects (DB connections,
   large dataframes, model fits) but are function-scoped. Evaluate whether they
   can be safely promoted to module or session scope.

2. **Unnecessary I/O**: Tests reading/writing real files when `tmp_path` or
   `StringIO`/`BytesIO` would work. Tests making real HTTP calls that should use
   `responses`, `httpx_mock`, or `monkeypatch`.

3. **Import overhead**: Test files that `import pandas`, `import numpy`,
   `import scipy` at module level but most tests in the file don't use them.
   Consider lazy imports or splitting the file.

4. **Sleep/wait**: Any `time.sleep()`, `asyncio.sleep()`, or polling loops in
   tests. These are almost always replaceable with proper mocking or event-based
   synchronization.

5. **Redundant setup**: `setup_method`/`teardown_method` patterns that recreate
   objects already available as fixtures. setUp/tearDown methods leftover from
   unittest migration.

6. **Large inline data**: Tests that construct large dicts, lists, or DataFrames
   inline. These should be fixtures or loaded from test data files.

**Phase B: Runtime profiling (targeted)**

For the 20 largest/most suspicious test files from Phase A:
1. Run `pytest <file> --durations=0 -q 2>&1 | head -30` to get actual timings.
2. Identify tests taking >1s and cross-reference with your static findings.

Create `ergodic_insurance/tests/PERFORMANCE_REPORT.md` with findings and estimated time savings.
Implement the fixes directly — but be conservative with fixture scope changes.
When promoting a fixture from function to module/session scope, verify the tests
don't mutate the fixture's state.

Send a message to team-lead when you finish each phase.
```

### Teammate 4: Reliability Engineer

```
Your name is reliability-engineer. You specialize in identifying and fixing
flaky tests.

Read `ergodic_insurance/tests/REFACTOR_SURVEY.md` for context on the test suite structure.

Analyze every test file for flakiness indicators:

1. **Timing dependencies**: Tests using `time.time()`, `datetime.now()`, or
   comparing timestamps. Fix by freezing time with `freezegun` or monkeypatching
   `time.time`. Tests with `time.sleep()` for synchronization — replace with
   proper waits or mocking.

2. **Order dependence**: Tests that modify module-level state, class variables,
   or environment variables without restoring them. Tests that depend on dict
   ordering (Python 3.7+ is insertion-ordered, but test discovery order isn't
   guaranteed). Tests that create files without using `tmp_path`.

3. **Floating-point comparisons**: Any `assert x == y` where x or y is a float.
   Replace with `assert x == pytest.approx(y)` or `assert x == pytest.approx(y, rel=1e-6)`.
   For actuarial calculations, consider appropriate tolerance levels — loss
   ratios and premium calculations may need `rel=1e-4` while statistical
   distribution fits might need `rel=1e-2`.

4. **Randomness**: Tests using `random` without seeding. Tests depending on
   `numpy.random` without `np.random.seed()` or `rng = np.random.default_rng(42)`.
   Monte Carlo tests that don't use enough samples for stable results.

5. **Resource leaks**: Tests that open files, sockets, or database connections
   without proper cleanup. Missing `with` statements or teardown.

6. **Environment sensitivity**: Tests that assume specific locale, timezone,
   OS path separators, or CPU count. Tests that read environment variables
   without defaults.

Create `ergodic_insurance/tests/RELIABILITY_REPORT.md` with findings.
Implement fixes directly. For floating-point fixes, add a comment explaining
the chosen tolerance when it's not the default.

Send a message to team-lead when you finish each major category.
```

## Lead Coordination Protocol

After spawning all 4 teammates:

1. Monitor their progress through messages. Keep a running status board.
2. If two teammates want to modify the same file, coordinate: have the first one
   finish, then tell the second to pull the latest changes before editing.
3. When all teammates are done, review the 4 reports:
   - `ergodic_insurance/tests/TAUTOLOGY_REPORT.md`
   - `ergodic_insurance/tests/REDUNDANCY_REPORT.md`
   - `ergodic_insurance/tests/PERFORMANCE_REPORT.md`
   - `ergodic_insurance/tests/RELIABILITY_REPORT.md`

4. Produce a consolidated `ergodic_insurance/tests/REFACTOR_SUMMARY_2026_02_09.md` with:
   - Total tests before vs. after
   - Tests deleted (with justification categories)
   - Tests rewritten
   - Tests consolidated via parametrize
   - Estimated test suite speedup
   - Items flagged for human review

5. Run `pytest --co -q 2>/dev/null | wc -l` to verify the new test count.
6. Run a targeted test pass on the most-modified files to verify nothing is broken:
   `pytest ergodic_insurance/tests/ -x --timeout=120 -q 2>&1 | tail -20`
7. If tests fail, identify which teammate's changes caused it and send them a
   message to fix it.
8. Commit all changes with:
   ```
   git add ergodic_insurance/tests/
   git commit -m "refactor: comprehensive test suite cleanup

   - Removed N tautological tests
   - Consolidated M duplicate tests via parametrize
   - Optimized K slow tests (fixture scoping, I/O mocking)
   - Fixed J flaky tests (timing, float comparison, randomness)
   - Added markers: @pytest.mark.slow, @pytest.mark.integration

   See ergodic_insurance/tests/REFACTOR_SUMMARY_2026_02_09.md for full details."
   ```
9. Create a pull request to the `develop` branch, but don't push the pr until I review.

## Important Constraints

- **Never delete a test you don't understand.** If intent is unclear, flag it for
  human review with a TODO comment rather than deleting.
- **Never modify source code.** Only tests, fixtures, and conftest files.
- **Preserve all integration tests** in `ergodic_insurance/tests/integration/` unless they are
  clearly tautological. Integration tests are expensive to recreate.
- **Preserve test intent.** When consolidating duplicates, ensure the parametrized
  version covers all the original edge cases, not just the happy path.
- **Be conservative with actuarial/statistical tests.** Tests involving
  distributions, loss models, credibility calculations, or reserve estimates
  may appear redundant but test different numerical edge cases. When in doubt,
  keep them and flag for review.
