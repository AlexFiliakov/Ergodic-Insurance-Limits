# Useful Prompts

## Basic issue fix

Use subagents to parallelize the work whenever possible and resolve issue # . Ensure all relevant tests pass, but skip full-suite validation because it times out.

## High-Powered Debug

Create an agent team to review my Python package for actuarial/risk management use. Spawn five reviewers:
- One focused on performance optimization opportunities
- One validating financial implementation and GAAP adherence
- One checking the mathematical implementation correctness
- One reviewing API usability for actuarial and risk management professionals
- One validating existing GitHub issues for accuracy and completeness

Have each teammate:
1. Review their specific domain independently
2. Document findings with file paths and line numbers
3. Create detailed GitHub issues for bugs and enhancements
4. Include competing implementation approaches with pros/cons
5. Recommend the best course of action with justification
6. Validate every aspect of its assigned domain. Include edge cases, unusual patterns, and potential future issues. If any area lacks coverage or validation, document it as an issue.
7. Each reviewer should identify at least 10 actionable issues in their domain. If fewer issues are found, explain why the codebase is already well-optimized in that area.
8. Each reviewer should maintain their own Markdown file in `\docs` documenting:
  - Areas reviewed so far
  - Number of issues identified
  - Remaining areas to cover

Each issue should be implementation-ready with:
- Clear problem statement
- Specific code locations affected
- Alternative solutions evaluated
- Recommended approach with reasoning
- Acceptance criteria for completion

This package is used in production financial systems. A thorough review is critical to prevent calculation errors that could affect compliance and financial reporting.

---
