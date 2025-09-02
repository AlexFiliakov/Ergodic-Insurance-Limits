#!/usr/bin/env python3
"""Fix collapsed headers in documentation."""

# Read the file
with open("ergodic_insurance/docs/theory/04_optimization_theory.md", "r", encoding="utf-8") as f:
    content = f.read()

# Find the collapsed line starting with "## Numerical Methods"
import re

# Replace the problematic collapsed line
pattern = r'## Numerical Methods ### Gradient-Based Methods.*?``` --- \(stochastic-control\)='
replacement = '''## Numerical Methods

### Gradient-Based Methods

#### Gradient Descent

$$
x_{k+1} = x_k - \\alpha_k \\nabla f(x_k)
$$

#### Newton's Method

$$
x_{k+1} = x_k - H_f(x_k)^{-1} \\nabla f(x_k)
$$

#### Quasi-Newton (BFGS)

$$
x_{k+1} = x_k - \\alpha_k B_k^{-1} \\nabla f(x_k)
$$

where $B_k$ approximates the Hessian.

### Derivative-Free Methods

```python
class OptimizationMethods:
    """Compare different optimization methods for insurance problems."""

    def __init__(self, objective, bounds):
        self.objective = objective
        self.bounds = bounds

    def gradient_descent(self, x0, learning_rate=0.01, max_iter=500):
        """Basic gradient descent with numerical gradients."""

        x = x0.copy()
        history = [x.copy()]

        for _ in range(max_iter):
            # Numerical gradient
            grad = self.numerical_gradient(x)

            # Update
            x = x - learning_rate * grad

            # Project onto bounds
            x = np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

            history.append(x.copy())

            # Check convergence
            if np.linalg.norm(grad) < 1e-6:
                # Continue recording the converged value
                for _ in range(max_iter - len(history) + 1):
                    history.append(x.copy())
                break

        return x, history

    def numerical_gradient(self, x, eps=1e-6):
        """Compute gradient using finite differences."""

        grad = np.zeros_like(x)

        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps

            grad[i] = (self.objective(x_plus) - self.objective(x_minus)) / (2 * eps)

        return grad

    def simulated_annealing(self, x0, temp=1.0, cooling=0.99, max_iter=500):
        """Simulated annealing for global optimization."""

        x = x0.copy()
        best_x = x.copy()
        best_f = self.objective(x)

        history = [x.copy()]

        for i in range(max_iter):
            # Generate neighbor
            neighbor = x + np.random.randn(len(x)) * temp
            neighbor = np.clip(neighbor, self.bounds[:, 0], self.bounds[:, 1])

            # Evaluate
            f_neighbor = self.objective(neighbor)
            f_current = self.objective(x)

            # Accept or reject
            delta = f_neighbor - f_current
            if delta < 0 or np.random.rand() < np.exp(-delta / max(temp, 1e-10)):
                x = neighbor

            if f_neighbor < best_f:
                best_x = neighbor.copy()
                best_f = f_neighbor

            # Cool down more gradually
            temp *= cooling

        history.append(best_x.copy())  # Track best found so far

        return best_x, history

    def particle_swarm(self, n_particles=30, max_iter=500):
        """Particle swarm optimization with improved convergence."""

        # Initialize swarm
        particles = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            (n_particles, len(self.bounds))
        )
        velocities = np.random.randn(n_particles, len(self.bounds)) * 0.1

        # Best positions
        p_best = particles.copy()
        p_best_scores = np.array([self.objective(p) for p in particles])

        g_best_idx = np.argmin(p_best_scores)
        g_best = p_best[g_best_idx].copy()
        g_best_score = p_best_scores[g_best_idx]

        history = [g_best.copy()]

        # PSO parameters with better convergence
        w_start = 0.9  # Higher initial inertia
        w_end = 0.4    # Lower final inertia
        c1 = 2.0       # Cognitive parameter
        c2 = 2.0       # Social parameter

        for iteration in range(max_iter):
            # Linear inertia weight decay
            w = w_start - (w_start - w_end) * iteration / max_iter

            # Update all particles
            for i in range(n_particles):
                # Update velocity
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (w * velocities[i] +
                                c1 * r1 * (p_best[i] - particles[i]) +
                                c2 * r2 * (g_best - particles[i]))

                # Limit velocity to prevent divergence
                max_vel = (self.bounds[:, 1] - self.bounds[:, 0]) * 0.1
                velocities[i] = np.clip(velocities[i], -max_vel, max_vel)

                # Update position
                particles[i] = particles[i] + velocities[i]
                particles[i] = np.clip(particles[i], self.bounds[:, 0], self.bounds[:, 1])

                # Update best positions
                score = self.objective(particles[i])
                if score < p_best_scores[i]:
                    p_best[i] = particles[i].copy()
                    p_best_scores[i] = score

                if score < g_best_score:
                    g_best = particles[i].copy()
                    g_best_score = score

            history.append(g_best.copy())

        return g_best, history

    def compare_methods(self, x0):
        """Compare convergence of different methods."""

        # Set random seed for reproducibility
        np.random.seed(42)

        methods = {
            'Gradient Descent': lambda: self.gradient_descent(x0, max_iter=500),
            'Simulated Annealing': lambda: self.simulated_annealing(x0, max_iter=500),
            'Particle Swarm': lambda: self.particle_swarm(max_iter=500)
        }

        results = {}

        for name, method in methods.items():
            np.random.seed(42)  # Reset seed for each method
            solution, history = method()
            results[name] = {
                'solution': solution,
                'value': self.objective(solution),
                'history': history
            }

        return results

    def plot_convergence(self, results):
        """Visualize convergence of different methods."""

        plt.figure(figsize=(12, 6))

        for name, result in results.items():
            history = result['history']
            values = [self.objective(x) for x in history]
            plt.plot(values, label=name, linewidth=2)

        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.title('Convergence Comparison')
        plt.legend()

        # Use regular number formatting instead of scientific notation
        ax = plt.gca()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))

        plt.grid(True, alpha=0.3)
        plt.xlim(0, 500)  # Set x-axis limit to 500

        # Save the figure
        plt.savefig('../../assets/convergence_comparison.png', dpi=100, bbox_inches='tight')
        plt.show()

# Test optimization methods
def insurance_objective(x):
    """Complex insurance optimization objective."""
    retention, limit, deductible = x

    # Expected cost
    cost = 0.02 * limit + 0.01 * deductible + 0.005 / (1 + retention)

    # Risk penalty
    risk = np.exp(-retention) + np.exp(-limit/10)

    # Non-convex component
    complexity = np.sin(retention * 5) * 0.1

    return cost + risk + complexity

bounds = np.array([
    [0, 2],   # retention
    [0, 10],  # limit
    [0, 1]    # deductible
])

opt_methods = OptimizationMethods(insurance_objective, bounds)
x0 = np.array([1, 5, 0.5])

results = opt_methods.compare_methods(x0)
opt_methods.plot_convergence(results)

# Print results
print("\\nOptimization Results:")
print("=" * 50)
for name, result in results.items():
    print(f"\\n{name}:")
    print(f"  Solution: {result['solution']}")
    print(f"  Final Value: {result['value']:.6f}")
    print(f"  Converged to: {insurance_objective(result['solution']):.6f}")

```
#### Sample Output
![Convergence Comparison](../../../assets/convergence_comparison.png)
```

Optimization Results:
==================================================

Gradient Descent:
  Solution: [1.09050224 5.20024017 0.45      ]
  Final Value: 0.967612
  Converged to: 0.967612

Simulated Annealing:
  Solution: [ 2. 10.  0.]
  Final Value: 0.650479
  Converged to: 0.650479

Particle Swarm:
  Solution: [ 2. 10.  0.]
  Final Value: 0.650479
  Converged to: 0.650479

```
---
(stochastic-control)='''

# Use re.DOTALL to match across lines
content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Write back
with open("ergodic_insurance/docs/theory/04_optimization_theory.md", "w", encoding="utf-8") as f:
    f.write(content)

print("Fixed collapsed headers in Numerical Methods section")
