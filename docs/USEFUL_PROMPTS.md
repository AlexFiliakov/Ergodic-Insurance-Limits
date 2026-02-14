# Useful Prompts

## Basic issue fix

Use subagents to parallelize the work whenever possible and resolve issue # . Ensure all relevant tests pass, but skip full-suite validation because it times out.

## More convoluted issue fix

Use subagents to parallelize the work whenever possible and resolve issue #. Ensure all relevant tests pass, but skip full-suite validation because it times out. Work in a new branch. Plan, execute, review. Commit and make a pull request, but don't push to main.

## Research Failing Test

See pull request # which has failing tests. Investigate the failing tests thoroughly to come up with clear resolution plans. Where multiple approaches are viable, clearly explain each approach and make a recommendation with justification. Report your findings and recommended fix approaches in a comment on the pull request.

## Fix Failing Test From Comments

See pull request # which has failing tests. The resolution approach is described in the pull request comments. Use parallel subagents as needed to fix the failing test, then test the repairs, but don't run a full test-suite because it will time out. Once relevant tests pass, commit and push the fixes.

## Backlog Prioritization

Review the GitHub issues backlog for issues labeled "priority-high" and create an implementation roadmap using the Mikado Method for parallel implementation on Claude Opus 4.6 models. I will be running up to 6 Claude Code sessions in parallel, so partition the work into batches of tasks up to 6 in parallel that won't touch the same files, with more critical items being done first. Your output should be a roadmap in `docs\reviews\ROADMAP_2026_02_12.md`. Each batch of issues should begin with a section heading summarizing the overall issues in the batch in a one-sentence heading. The section heading should be followed by a table of parallel tasks consisting of the following columns: | Issue Number | Change Title | Files Affected |

This should be followed by a brief description of why these tasks can be done in parallel, and why execute the batch in this particular order.

After that, to help me configure the Claude sections, follow up the explanations with a format as follows, which will be the code I will run manually and the prompts I'll use to execute your recommended batch issues:

"""
git worktree add C:/worktrees/<issue 1 branch name> -b bugfix/<issue 1 branch name>
git worktree add C:/worktrees/<issue 2 branch name> -b bugfix/<issue 2 branch name>
git worktree add C:/worktrees/<issue 3 branch name> -b bugfix/<issue 3 branch name>
git worktree add C:/worktrees/<issue 4 branch name> -b bugfix/<issue 4 branch name>
...

cd "C:/worktrees/<issue 1 branch name>";claude --dangerously-skip-permissions
cd "C:/worktrees/<issue 2 branch name>";claude --dangerously-skip-permissions
cd "C:/worktrees/<issue 3 branch name>";claude --dangerously-skip-permissions
cd "C:/worktrees/<issue 4 branch name>";claude --dangerously-skip-permissions
...

Make sure your branch is up to date with `develop` before starting work. Use subagents to parallelize the work whenever possible and resolve issue #<issue 1 number>. Ensure all relevant tests pass, but skip full-suite validation because it times out. Work in the current branch, which is part of a git worktree. Plan, execute, review. Commit and create a pull request to the `develop` branch, but don't push the pr until I review.

Make sure your branch is up to date with `develop` before starting work. Use subagents to parallelize the work whenever possible and resolve issue #<issue 1 number>. Ensure all relevant tests pass, but skip full-suite validation because it times out. Work in the current branch, which is part of a git worktree. Plan, execute, review. Commit and create a pull request to the `develop` branch, but don't push the pr until I review.

Make sure your branch is up to date with `develop` before starting work. Use subagents to parallelize the work whenever possible and resolve issue #<issue 1 number>. Ensure all relevant tests pass, but skip full-suite validation because it times out. Work in the current branch, which is part of a git worktree. Plan, execute, review. Commit and create a pull request to the `develop` branch, but don't push the pr until I review.

Make sure your branch is up to date with `develop` before starting work. Use subagents to parallelize the work whenever possible and resolve issue #<issue 1 number>. Ensure all relevant tests pass, but skip full-suite validation because it times out. Work in the current branch, which is part of a git worktree. Plan, execute, review. Commit and create a pull request to the `develop` branch, but don't push the pr until I review.

...
"""

When a Claude team is recommended for an issue, note it in your report at the end of the batch with the full team prompt clearly specified.
