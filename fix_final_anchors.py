#!/usr/bin/env python3
"""Fix remaining anchor formatting issues in documentation."""

import re

def fix_anchors_on_same_line():
    """Fix anchors that are on the same line as headers."""

    # Fix 04_optimization_theory.md
    file1 = "ergodic_insurance/docs/theory/04_optimization_theory.md"
    with open(file1, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix anchors on same line as headers
    fixes = [
        (r'\(pareto-efficiency\)= ## Pareto Efficiency', '(pareto-efficiency)=\n\n## Pareto Efficiency'),
        (r'\(multi-objective-optimization\)= ## Multi-Objective Optimization', '(multi-objective-optimization)=\n\n## Multi-Objective Optimization'),
        (r'\(hamilton-jacobi-bellman-equations\)= ## Hamilton-Jacobi-Bellman Equations', '(hamilton-jacobi-bellman-equations)=\n\n## Hamilton-Jacobi-Bellman Equations'),
        (r'\(convergence-criteria\)= ## Convergence Criteria', '(convergence-criteria)=\n\n## Convergence Criteria'),
    ]

    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content)

    # Fix the numerical-methods anchor that's embedded in code
    # Find the line and fix it
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.strip() == '(numerical-methods)=':
            # Make sure it's properly formatted as an anchor
            if i > 0 and '```' not in lines[i-1]:
                # Good, it's not in a code block
                pass
            elif '970:' in line or 'numerical-methods' in line:
                # It might be in a grep output or code block, extract it
                lines[i] = '(numerical-methods)='
                if i + 1 < len(lines) and not lines[i+1].strip().startswith('#'):
                    lines.insert(i+1, '')
                    lines.insert(i+2, '## Numerical Methods')

    # Also look for the pattern in the middle of text
    content = '\n'.join(lines)
    content = re.sub(r'``` --- \(numerical-methods\)= ## Numerical Methods',
                     '```\n\n---\n\n(numerical-methods)=\n\n## Numerical Methods',
                     content)

    with open(file1, 'w', encoding='utf-8') as f:
        f.write(content)

    # Fix 05_statistical_methods.md
    file2 = "ergodic_insurance/docs/theory/05_statistical_methods.md"
    with open(file2, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix confidence-intervals anchor on same line as header
    content = re.sub(r'\(confidence-intervals\)= ## Confidence Intervals',
                     '(confidence-intervals)=\n\n## Confidence Intervals',
                     content)

    with open(file2, 'w', encoding='utf-8') as f:
        f.write(content)

    print("Fixed anchor formatting issues")

if __name__ == "__main__":
    fix_anchors_on_same_line()
