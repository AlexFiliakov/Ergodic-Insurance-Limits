# Clauce Code Prompt Planning

Let's discuss and plan a project based on the specification in `Financial Modeling Framework - Spec.md`. Read this file, ask me clarifying questions, and then let's come up with a project roadmap.

## Project Planning Questions

1. Scope & Focus: Are you looking to build a complete simulation framework, or would you prefer to start with a specific component (e.g., the loss modeling, ergodic optimization engine, or insurance layer structuring)?
2. Target Users: Who will use this framework - internal risk managers, actuaries, or is this for broader distribution/SaaS?
3. Data Sources: Do you have historical loss data for calibration, or should we build synthetic data generation capabilities based on the specified parameters?
4. Integration Requirements: Should this integrate with existing financial systems, or operate as a standalone analytical tool?
5. Visualization Needs: How important are interactive dashboards vs. static reports for presenting results?
6. Performance Constraints: The spec mentions 100K-1M Monte Carlo iterations. What's your tolerance for computation time (seconds, minutes, hours)?
7. Priority Features: Which is most critical initially:
- Basic ergodic growth rate calculations
- Multi-layer insurance optimization
- ROE optimization with ruin constraints
- Full Monte Carlo simulation engine
8. Technology Preferences: The existing simone project uses TypeScript. Would you prefer to continue in TypeScript or switch to Python (as mentioned in the spec) for better scientific computing libraries?

### Project Planning Responses

1. I'm looking to build the complete simulation framework and run simulations in order to make some preliminary conclusions.
2. Actuaries who are familiar with Python are the target users. The implementation should be in Python.
3. We should build synthetic data based on the specified parameters. I don't have industry sources yet to calibrate.
4. This is a standalone analytical tool.
5. Compelling static reports are crucial since this will first feed a series of blog posts.
6. No performance constraints, no need to test performance yet.
7. The following order of importance:
- Basic financial model
- Basic ergodic growth rate calculations
- Multi-layer insurance optimization
- ROE optimization with ruin constraints
- Full Monte Carlo simulation engine
8. Definitely prefer Python

