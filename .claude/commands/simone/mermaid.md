# Mermaid Diagram Maintenance Commands

This document provides Claude commands for maintaining and updating the architecture diagrams as the codebase evolves.

## Overview

The architecture documentation uses Mermaid diagrams located in `docs/architecture/`. These diagrams need to be kept in sync with code changes to maintain accurate documentation.

## Available Commands

### 1. Update Class Diagram After Code Changes

When you've modified class structures, use this command to update the relevant diagram:

```
@claude Please update the class diagram in docs/architecture/class_diagrams/core_classes.md to reflect the following changes:
- [List your changes here]
- [e.g., Added new method to WidgetManufacturer]
- [e.g., Renamed InsurancePolicy to Policy]
```

### 2. Add New Module to Architecture

When adding a new module or significant component:

```
@claude Please add the new module [MODULE_NAME] to:
1. The module overview diagram in docs/architecture/module_overview.md
2. The context diagram if it's a major component
3. Create a new class diagram if needed

The module does: [DESCRIPTION]
It depends on: [DEPENDENCIES]
It's used by: [CONSUMERS]
```

### 3. Generate Sequence Diagram for Complex Flow

For documenting complex interaction flows:

```
@claude Please create a sequence diagram showing the interaction flow for [FEATURE/PROCESS]:
- Starting point: [START]
- Key actors: [LIST OF COMPONENTS]
- End result: [OUTCOME]
Save it in docs/architecture/[appropriate_location].md
```

### 4. Validate Diagram Accuracy

To ensure diagrams match current code:

```
@claude Please validate that the diagrams in docs/architecture/ accurately reflect the current codebase:
1. Check class names and methods match actual implementation
2. Verify module dependencies are correct
3. Confirm data flow representations are accurate
4. Report any discrepancies found
```

### 5. Create Component-Specific Diagram

For detailed documentation of a specific component:

```
@claude Please create a detailed class diagram for the [COMPONENT_NAME] module showing:
- All classes and their relationships
- Key methods and properties
- Inheritance hierarchies
- Dependencies on other modules
Save as docs/architecture/class_diagrams/[component_name].md
```

### 6. Update After Refactoring

After significant refactoring:

```
@claude The [MODULE/COMPONENT] has been refactored. Please:
1. Review the changes in [FILE_PATHS]
2. Update all affected diagrams in docs/architecture/
3. List all diagrams that were updated
4. Highlight any architectural improvements
```

### 7. Generate Data Flow Diagram

For visualizing data transformations:

```
@claude Please create a data flow diagram showing how [DATA_TYPE] flows through the system:
- Origin: [WHERE IT COMES FROM]
- Transformations: [PROCESSING STEPS]
- Destination: [WHERE IT ENDS UP]
- Include key data structures at each stage
```

### 8. Document Design Pattern Usage

To document design patterns in the codebase:

```
@claude Please identify and document design patterns used in [MODULE/COMPONENT]:
1. Create a diagram showing the pattern implementation
2. Explain why this pattern was chosen
3. Show the participating classes
4. Save in docs/architecture/patterns/[pattern_name].md
```

## Best Practices

### When to Update Diagrams

Update diagrams when:
- Adding new classes or modules
- Changing class relationships or dependencies
- Modifying significant methods or interfaces
- Refactoring architecture
- Adding new data flows or processes

### Diagram Organization

- **Context Diagram**: Update for major component additions/removals
- **Module Overview**: Update for new modules or dependency changes
- **Class Diagrams**: Update for class structure changes
- **Service Layer**: Update for infrastructure changes

### Naming Conventions

- Use consistent class names matching the code
- Include file names in descriptions (e.g., "manufacturer.py")
- Use clear, descriptive labels for relationships
- Follow Python naming conventions in diagrams

### Documentation Standards

Each diagram should include:
1. Clear title and description
2. Mermaid code block with proper syntax
3. Brief explanation of key components
4. Design patterns used (if applicable)
5. Links to related diagrams

## Automation Tips

### Pre-commit Hook

Consider adding a reminder in your workflow:

```python
# In your development process
"""
REMINDER: If this change affects architecture:
1. Update relevant diagrams in docs/architecture/
2. Run: @claude validate diagram accuracy
"""
```

### Regular Maintenance

Schedule regular diagram reviews:

```
@claude Please perform a monthly architecture documentation review:
1. Validate all diagrams against current code
2. Identify undocumented components
3. Check for outdated information
4. Suggest improvements
```

## Troubleshooting

### Common Issues

1. **Diagram Too Complex**: Break into multiple focused diagrams
2. **Outdated Information**: Run validation command regularly
3. **Missing Components**: Check for new files not in diagrams
4. **Incorrect Relationships**: Verify imports and dependencies

### Getting Help

If you need assistance with diagram updates:

```
@claude I need help updating the architecture diagrams:
- Current issue: [DESCRIBE PROBLEM]
- Affected components: [LIST]
- Attempted solution: [WHAT YOU TRIED]
Please provide guidance on the best approach.
```

## Quick Reference

### Mermaid Syntax Reminders

- Class diagram: `classDiagram`
- Sequence diagram: `sequenceDiagram`
- Flowchart: `flowchart TD` (top-down) or `flowchart LR` (left-right)
- Relationships: `-->` (dependency), `--|>` (inheritance), `--*` (composition)

### File Locations

- Context: `docs/architecture/context_diagram.md`
- Modules: `docs/architecture/module_overview.md`
- Classes: `docs/architecture/class_diagrams/*.md`
- Custom: `docs/diagrams/` (alternative location)

## Example Update Session

```bash
# After adding a new risk analysis module:

@claude I've added a new module called advanced_risk_analyzer.py that:
- Calculates tail risk metrics
- Depends on risk_metrics.py and numpy
- Is used by business_optimizer.py
- Contains classes: TailRiskAnalyzer, ExtremeValueDistribution

Please:
1. Add it to the module overview diagram
2. Create a class diagram for its components
3. Update the context diagram if needed
4. Show how it integrates with existing risk analysis
```

---

*Last Updated: [Auto-updated by Claude]*
*Version: 1.0.0*
