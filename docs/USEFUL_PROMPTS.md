# Useful Prompts

## Basic issue fix

Use subagents to parallelize the work whenever possible and resolve issue # . Ensure all relevant tests pass, but skip full-suite validation because it times out.

## More convoluted issue fix

Use subagents to parallelize the work whenever possible and resolve issue #. Ensure all relevant tests pass, but skip full-suite validation because it times out. Work in a new branch. Plan, execute, review. Commit and make a pull request, but don't push to main.

## Research Failing Test

See pull request # which has failing tests. Investigate the failing tests thoroughly to come up with clear resolution plans. Where multiple approaches are viable, clearly explain each approach and make a recommendation with justification. Report your findings and recommended fix approaches in a comment on the pull request.

## Fix Failing Test From Comments

See pull request # which has failing tests. The resolution approach is described in the pull request comments. Use parallel subagents as needed to fix the failing test, then test the repairs, but don't run a full test-suite because it will time out. Once relevant tests pass, commit and push the fixes.
