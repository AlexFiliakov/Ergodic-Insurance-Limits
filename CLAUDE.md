This project implements a framework for assessing insurance purchasing decisions and helps optimize insurance retentions and limits for businesses using ergodic theory (time-average) rather than traditional ensemble approaches. The framework aims to demonstrate how insurance transforms from a cost center to a growth enabler when analyzed through time averages.

The model is intended to be used with public financial data to model a real business, simulate an insurance market cycle, and optimize long-term business growth by selecting the right risk management strategy.

## Key Objectives
1. Build a complete simulation framework for ergodic insurance optimization
2. Generate compelling evidence for blog posts demonstrating ergodic advantages
3. Provide actuaries, CFOs, and risk managers with practical Python tools for insurance decision-making
4. Validate that optimal insurance premiums can exceed expected losses by 200-500% while enhancing growth

Primary Language: Python 3.12+
Version Control: Never commit directly to `main` branch - use feature branches and make pull requests to `develop` branch
Code Quality: Run formatters and linters before committing

## Git Configuration
- **User**: Alex Filiakov
- **Email**: alexfiliakov@gmail.com
- **Repository**: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits

## Commit Messages & Version Bumps
This project uses [python-semantic-release](https://python-semantic-release.readthedocs.io/) to automatically version and release on pushes to `main`. It reads **only first-parent commits** on `main`, so the **merge commit message** is what determines the version bump — not the individual commits inside a PR branch.

### Conventional Commits format (required)
All commits and PR titles must follow [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Version Bump | Example |
|---|---|---|
| `fix:` | Patch (0.0.x) | `fix: correct sign error in penalty calculation` |
| `perf:` | Patch (0.0.x) | `perf: remove deepcopy from worker hot loop` |
| `feat:` | Minor (0.x.0) | `feat: add NOL carryforward tracking per ASC 740` |
| `feat!:` or `BREAKING CHANGE:` | Major (x.0.0) | `feat!: redesign Config API` |
| `docs:`, `ci:`, `test:`, `chore:`, `refactor:`, `style:` | No release | `ci: add pip caching to workflows` |

### PR titles to `main` (critical)
When merging a PR into `main`, GitHub uses the **PR title** as the merge commit message. Because semantic-release only sees first-parent commits, this title is the **sole input** for version detection. Always:

1. **Title the PR with a conventional commit prefix** matching the highest-impact change:
   - If the PR contains any `feat:` commits, title it `feat: <summary>`
   - If the PR contains only `fix:`/`perf:` commits, title it `fix: <summary>` or `perf: <summary>`
   - If the PR contains a breaking change, title it `feat!: <summary>`
2. **Never use non-standard prefixes** like `Release:`, `Merge:`, or `Update:` — these produce no version bump.

### PR titles to `develop`
PRs into `develop` do not trigger releases, so the title format is less critical. However, using conventional commit prefixes consistently is still recommended for readability.

### `major_on_zero = false`
While the project is pre-1.0, breaking changes bump minor (not major). This is configured in `pyproject.toml` under `[tool.semantic_release]`.

You have access to the `log_activity` tool. Use it to record your activities after every activity that is relevant for the project. This helps track development progress and understand what has been done.

## When Starting Work
1. Review this file and the sprint documents in `simone/`
2. Check current git status and recent commits
3. Run tests to ensure everything is working
4. Check the todo items in sprint documents for next tasks
5. Use the TodoWrite tool to track your work progress

## Documentation Layout
The documentation is split between three parts:
- "ergodic_insurance\docs\tutorials\" which has the tutorials on specific important features.
- "ergodic_insurance\docs\user_guide\" which has the Quick Start guide and a general overview of the project.
- "ergodic_insurance\notebooks\" which provides specific and comprehensive examples of how to use the code.
