# Useful Prompts

## Basic issue fix

Use subagents to parallelize the work whenever possible and resolve issue # . Ensure all relevant tests pass, but skip full-suite validation because it times out.

## More convoluted issue fix

Use subagents to parallelize the work whenever possible and resolve issue #. Ensure all relevant tests pass, but skip full-suite validation because it times out. Work in a new branch. Plan, execute, review. Commit and make a pull request, but don't push to main.

## Research Failing Test

See pull request # which has a failing test. Investigate the failing test thoroughly to come up with clear resolution plans. Where multiple approaches are viable, clearly explain each approach and make a recommendation with justification. Report your findings and recommended fix approaches in a comment on the pull request.
