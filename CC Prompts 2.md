work on issue #46 "Hamilton-Jacobi-Bellman Solver for Optimal Control". Ask me clarifying questions, then proceed to resolve the issue. Make sure that all new and existing tests pass before closing the issue.

---

put the configuration plan as you outlined in `CONFIG_MIGRATION_PLAN.md` and `CONFIG_MIGRATION_TASKS.md` into a GitHub issue. Put all the work into a single GitHub issue labeled "enhancement" and "priority-high" and give it an appropriate title.

---

Update all docstrings in modules, submodules, classes, methods, and functions to aid development and understandability of the code. Adhere to the Google docstring style. Then update README.md and CLAUDE.md with the latest directory structures. Then update Sphinx documentation templates for the newest code structure, ensuring the documentation clearly references the refactored configuration with clear and practical demos.

---

Fix the following sphinx `make html` warnings. Ultrathink to plan, make a list of todos, and then execute:


Fix the following sphinx `make html` warnings. Ultrathink to plan, make a list of todos, and then execute:


Fix the following sphinx `make html` warnings. Ultrathink to plan, make a list of todos, and then execute:


---

work on issue #48 "Design and Implement Enhanced Parallel Simulation Architecture". Ask me clarifying questions, then ultrathink and create an execution list, and then proceed to resolve the issue. Make sure that all new and existing tests pass. Update all docstrings in modules, submodules, classes, methods, and functions to aid development and understandability of the code. Adhere to the Google docstring style. Then update README.md and CLAUDE.md with the latest directory structures. Then update Sphinx documentation templates for the newest code structure, ensuring the documentation clearly references the refactored configuration with clear and practical demos. Once everything is thoroughly completed, create a pull request.

---

work to complete GitHub issue #122 "Add Insurance Pricing Module with Market Cycle Support" and patch tests and jupyter notebooks as you go along, making sure all tests pass with high quality validation (avoiding things like always-true tests, or excessive mocks). Provide solid documentation using Google style docstrings. Ask me clarifying questions, ultrathink to make a plan, review the plan for completeness, then execute.

---

Auto-update failed · Try claude doctor
or npm i -g @anthropic-ai/claude-code

---

improve the test coverage of the following files to at least 90%, writing high quality tests that don't overly rely on mocks or have always-true condition, and other cheat tests:
...

---

Add some visuals to "docs/theory/05_statistical_methods.md" by creating these new visuals through "docs/theory/generate_visuals.py". I think each major section should have a relevant visual, preferably using the visuals we developed previously in earlier issues, such as the convergence diagnostics.

---

review the full project, excluding documentation and user guides, and determine whether it's ready to ship as v0.1.0. All the initial functionality should be implemented per the GitHub closed issues, and the only open issues that remain are documentation, which I will work on once the project code is stable. Do you see any features that were overlooked, any implementation oversights or bugs, or any tests that were fibbed (passing tests without really testing anything)? Please compile a state of the project and whether it's ready for v0.1.0

---



Fix the following test issues: "" has a lot of failing tests
Fix the following test issues: "" has a lot of failing tests
Fix the following test issues: "" has a lot of failing tests









Fix the following test issues:




Fix the following test issues:




Fix the following test issues:




Fix the following test issues:




Fix the following test issues:




Fix the following test issues:




Fix the following test issues:




Fix the following test issues:












Fix the following hook issues so I can commit, or suppress them if you think a fix will reduce readability:




Fix the following hook issues so I can commit, or suppress them if you think a fix will reduce readability:




Fix the following hook issues so I can commit, or suppress them if you think a fix will reduce readability:





Fix the following hook issues so I can commit, or suppress them if you think a fix will reduce readability:





Fix the following hook issues so I can commit, or suppress them if you think a fix will reduce readability:











Fix the following mypy issues so I can commit, or suppress them if you think a fix will reduce readability:




Fix the following mypy issues so I can commit, or suppress them if you think a fix will reduce readability:





Fix the following mypy issues so I can commit, or suppress them if you think a fix will reduce readability:












Fix the following pylint issues so I can commit, or suppress them if you think a fix will reduce readability:





Fix the following pylint issues so I can commit, or suppress them if you think a fix will reduce readability:





Fix the following pylint issues so I can commit, or suppress them if you think a fix will reduce readability:





Fix the following pylint issues so I can commit, or suppress them if you think a fix will reduce readability:








---

~/.claude/agents

---

Help me create a Code Base Learning Assistant Agent
This agent helps me understand unfamiliar codebases:
- Creates code maps and architecture diagrams
- Explains complex functions step-by-step
- Identifies design patterns used
- Traces data flow through the application
- Explains business logic in plain English
- Creates onboarding guides for new developers
- Answers "how does this work?" questions
Tell me what to put specifically in the agent description to make it most effective.



Help me create an SEO & Readability Optimizer Agent
This is an SEO and readability expert for blog content.

OPTIMIZATION TASKS:
- Analyze keyword density and placement
- Suggest title tag optimizations (50-60 chars)
- Create meta descriptions (150-160 chars)
- Generate slug/URL recommendations
- Add schema markup suggestions
- Optimize heading structure for featured snippets
- Calculate and improve readability scores
- Suggest internal linking opportunities
- Create XML sitemap entries
- Generate Open Graph tags
- Add Twitter Card metadata

READABILITY IMPROVEMENTS:
- Break up long paragraphs (3-4 sentences max)
- Suggest simpler word alternatives
- Add transition phrases between sections
- Create scannable content with bullet points
- Highlight key takeaways in callout boxes
- Ensure mobile-friendly formatting
Tell me what to put specifically in the agent description to make it most effective.

---

Review the spec documentation at "docs\specs\US_FINANCIAL_STATEMENTS_RESEARCH.md" and plan its integration into the current codebase.

First, analyze the spec and current codebase to understand:
1. Core functionality requirements from the spec
2. Existing codebase architecture and patterns
3. Potential integration points and dependencies

Then ask me clarifying questions about:
- Any unnecessary complexity that can be simplified for parsimonious implementation
- Performance requirements or constraints
- Integration dependencies I haven't documented
- Any ambiguous requirements in the spec

Then, partition the new implementation into coherent chunks that will be testable and create issues based on these chunks.
- Issues should have minimal interdependencies where possible
- Provide an approach for each issue
- If there are multiple viable approaches, provide trade-offs and recommendations
- Provide implementation guidance and what unit tests would be necessary to maintain high code quality
- Your code guidance should come mainly in the form of tests that need to pass
- Provide guidance on a system overhaul and removing legacy implementation without backward compatibility

---

Q1: Implement straight-line depreciation and a flat tax rate to start
Q2: Implement only annual upfront premiums with monthly amortization, no retroactively-rated policies
Q3: Financial statements should be generated annually in a yearly run and monthly in a monthly run. Don't worry about performance yet; keep the current approaches to simulation and parallelization and just bolt on the additional complexity. The maximum realistic simulation horizon is 100 years.
Q4: Financial statements should integrate with existing Monte Carlo simulations. We can generate the statements post-hoc if it's easier, or in real-time if it will help free up memory. Don't worry about importing actual financial data from public companies yet, that will be supplied via input parameters instead of from financial documents. Review the current financial statement creation functions and build from those, enhancing as needed.
Q5: Build a generic framework that can be configured for different industries, but simplify the implementation where industry-specific treatment will be unduly complex and bloated.
Other considerations:
- Implement full double-entry bookkeeping
- I like your other recommendations

---
