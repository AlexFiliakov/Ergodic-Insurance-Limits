# Optimization Theory for Insurance

## Table of Contents
1. [Constrained Optimization](#constrained-optimization)
2. [Pareto Efficiency](#pareto-efficiency)
3. [Multi-Objective Optimization](#multi-objective-optimization)
4. [Hamilton-Jacobi-Bellman Equations](#hamilton-jacobi-bellman-equations)
5. [Numerical Methods](#numerical-methods)
6. [Stochastic Control](#stochastic-control)
7. [Convergence Criteria](#convergence-criteria)
8. [Practical Implementation](#practical-implementation)

## Constrained Optimization

### General Formulation

The insurance optimization problem:

$$\begin{align}
\max_{x \in \mathcal{X}} &\quad f(x) \\
\text{subject to} &\quad g_i(x) \leq 0, \quad i = 1, ..., m \\
&\quad h_j(x) = 0, \quad j = 1, ..., p
\end{align}$$

where:
- $x$ = Decision variables (retention, limits, premiums)
- $f(x)$ = Objective (growth rate, utility, profit)
- $g_i(x)$ = Inequality constraints (budget, ruin probability)
- $h_j(x)$ = Equality constraints (regulatory requirements)

### Lagrangian Method

Form the Lagrangian:
$$\mathcal{L}(x, \lambda, \mu) = f(x) - \sum_{i=1}^m \lambda_i g_i(x) - \sum_{j=1}^p \mu_j h_j(x)$$

### Karush-Kuhn-Tucker (KKT) Conditions

Necessary conditions for optimality:

1. **Stationarity**: $\nabla_x \mathcal{L} = 0$
2. **Primal feasibility**: $g_i(x) \leq 0$, $h_j(x) = 0$
3. **Dual feasibility**: $\lambda_i \geq 0$
4. **Complementary slackness**: $\lambda_i g_i(x) = 0$

### Insurance Application

```python
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import matplotlib.pyplot as plt

class InsuranceOptimizer:
    """Optimize insurance program with constraints."""

    def __init__(self, initial_wealth, growth_params, loss_dist):
        self.W0 = initial_wealth
        self.growth_params = growth_params
        self.loss_dist = loss_dist

    def objective(self, x):
        """Maximize expected log wealth (negative for minimization)."""
        retention, limit = x[0], x[1]

        # Simulate outcomes
        n_sims = 1000
        final_wealth = []

        for _ in range(n_sims):
            # Base growth
            growth = np.random.normal(
                self.growth_params['mu'],
                self.growth_params['sigma']
            )
            wealth = self.W0 * (1 + growth)

            # Loss and insurance
            loss = self.loss_dist.rvs()
            retained_loss = min(loss, retention)
            covered_loss = min(max(0, loss - retention), limit)

            # Premium (simplified)
            premium = 0.01 * limit + 0.02 * max(0, limit - retention)

            # Final wealth
            wealth = wealth - retained_loss - premium
            final_wealth.append(max(0, wealth))

        # Expected log utility
        positive_wealth = [w for w in final_wealth if w > 0]
        if not positive_wealth:
            return 1e10  # Penalize bankruptcy

        return -np.mean(np.log(positive_wealth))

    def ruin_constraint(self, x):
        """Probability of ruin constraint."""
        retention, limit = x[0], x[1]

        # Simulate ruin probability
        n_sims = 1000
        ruin_count = 0

        for _ in range(n_sims):
            wealth = self.W0
            for year in range(10):  # 10-year horizon
                growth = np.random.normal(
                    self.growth_params['mu'],
                    self.growth_params['sigma']
                )
                wealth *= (1 + growth)

                loss = self.loss_dist.rvs()
                retained_loss = min(loss, retention)
                premium = 0.01 * limit + 0.02 * max(0, limit - retention)

                wealth = wealth - retained_loss - premium

                if wealth <= 0:
                    ruin_count += 1
                    break

        return ruin_count / n_sims  # Should be <= threshold

    def optimize(self, ruin_threshold=0.01, budget=None):
        """Find optimal insurance program."""

        # Initial guess
        x0 = [self.W0 * 0.05, self.W0 * 0.20]  # 5% retention, 20% limit

        # Bounds
        bounds = [
            (0, self.W0 * 0.10),  # Retention: 0 to 10% of wealth
            (0, self.W0 * 0.50)   # Limit: 0 to 50% of wealth
        ]

        # Constraints
        constraints = []

        # Ruin probability constraint
        constraints.append(NonlinearConstraint(
            self.ruin_constraint,
            lb=0,
            ub=ruin_threshold
        ))

        # Budget constraint if specified
        if budget:
            def premium_constraint(x):
                return 0.01 * x[1] + 0.02 * max(0, x[1] - x[0])

            constraints.append(NonlinearConstraint(
                premium_constraint,
                lb=0,
                ub=budget
            ))

        # Optimize
        result = minimize(
            self.objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100}
        )

        return result

# Example optimization
from scipy import stats

optimizer = InsuranceOptimizer(
    initial_wealth=10_000_000,
    growth_params={'mu': 0.08, 'sigma': 0.15},
    loss_dist=stats.lognorm(s=2, scale=100_000)
)

result = optimizer.optimize(ruin_threshold=0.01)

print(f"Optimal retention: ${result.x[0]:,.0f}")
print(f"Optimal limit: ${result.x[1]:,.0f}")
print(f"Expected growth: {-result.fun:.4f}")
```

## Pareto Efficiency

### Definition

A solution is **Pareto efficient** if no objective can be improved without worsening another.

### Pareto Frontier

Set of all Pareto efficient solutions:
$$\mathcal{P} = \{x^* \in \mathcal{X} : \nexists x \in \mathcal{X}, f_i(x) \geq f_i(x^*) \forall i, f_j(x) > f_j(x^*) \text{ for some } j\}$$

### Scalarization Methods

#### Weighted Sum

$$\min_{x} \sum_{i=1}^k w_i f_i(x)$$

where $\sum w_i = 1$, $w_i \geq 0$.

#### Epsilon-Constraint

$$\begin{align}
\min_{x} &\quad f_1(x) \\
\text{s.t.} &\quad f_i(x) \leq \epsilon_i, \quad i = 2, ..., k
\end{align}$$

### Insurance Trade-offs

```python
class ParetoFrontier:
    """Compute Pareto frontier for insurance decisions."""

    def __init__(self, objectives, constraints):
        self.objectives = objectives
        self.constraints = constraints

    def weighted_sum_method(self, weights_grid):
        """Generate Pareto frontier using weighted sum."""

        frontier = []

        for weights in weights_grid:
            # Combined objective
            def combined_objective(x):
                return sum(w * obj(x) for w, obj in
                          zip(weights, self.objectives))

            # Optimize
            result = minimize(
                combined_objective,
                x0=[0.5, 0.5],  # Initial guess
                bounds=[(0, 1), (0, 1)],
                constraints=self.constraints,
                method='SLSQP'
            )

            if result.success:
                # Evaluate all objectives
                obj_values = [obj(result.x) for obj in self.objectives]
                frontier.append({
                    'x': result.x,
                    'objectives': obj_values,
                    'weights': weights
                })

        return frontier

    def epsilon_constraint_method(self, epsilon_grid):
        """Generate Pareto frontier using epsilon-constraint."""

        frontier = []

        for eps in epsilon_grid:
            # Minimize first objective
            def primary_objective(x):
                return self.objectives[0](x)

            # Constrain other objectives
            additional_constraints = []
            for i, obj in enumerate(self.objectives[1:], 1):
                additional_constraints.append(
                    NonlinearConstraint(obj, lb=-np.inf, ub=eps[i-1])
                )

            # Optimize
            all_constraints = self.constraints + additional_constraints

            result = minimize(
                primary_objective,
                x0=[0.5, 0.5],
                bounds=[(0, 1), (0, 1)],
                constraints=all_constraints,
                method='SLSQP'
            )

            if result.success:
                obj_values = [obj(result.x) for obj in self.objectives]
                frontier.append({
                    'x': result.x,
                    'objectives': obj_values,
                    'epsilon': eps
                })

        return frontier

    def plot_frontier(self, frontier, obj_names=['Obj 1', 'Obj 2']):
        """Visualize Pareto frontier."""

        objectives = np.array([f['objectives'] for f in frontier])

        if objectives.shape[1] == 2:
            # 2D plot
            plt.figure(figsize=(8, 6))
            plt.scatter(objectives[:, 0], objectives[:, 1], s=50)
            plt.plot(objectives[:, 0], objectives[:, 1], 'b-', alpha=0.3)
            plt.xlabel(obj_names[0])
            plt.ylabel(obj_names[1])
            plt.title('Pareto Frontier')
            plt.grid(True, alpha=0.3)

            # Annotate some points
            for i in [0, len(frontier)//2, -1]:
                plt.annotate(f'Solution {i}',
                           (objectives[i, 0], objectives[i, 1]),
                           xytext=(5, 5), textcoords='offset points')

        elif objectives.shape[1] == 3:
            # 3D plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2])
            ax.set_xlabel(obj_names[0])
            ax.set_ylabel(obj_names[1])
            ax.set_zlabel(obj_names[2])
            ax.set_title('3D Pareto Frontier')

        plt.show()

# Example: Growth vs Risk trade-off
def growth_objective(x):
    """Negative expected growth (for minimization)."""
    retention, coverage = x
    return -(0.08 - 0.02 * retention + 0.01 * coverage)

def risk_objective(x):
    """Risk measure (VaR)."""
    retention, coverage = x
    return 0.5 * retention - 0.3 * coverage + 0.2

# Create Pareto frontier
objectives = [growth_objective, risk_objective]
constraints = [
    NonlinearConstraint(lambda x: x[0] + x[1], lb=0, ub=1.5)
]

pareto = ParetoFrontier(objectives, constraints)

# Generate frontier
weights_grid = [(w, 1-w) for w in np.linspace(0, 1, 20)]
frontier = pareto.weighted_sum_method(weights_grid)

# Visualize
pareto.plot_frontier(frontier, ['Negative Growth', 'Risk'])
```

## Multi-Objective Optimization

### Problem Formulation

$$\min_{x \in \mathcal{X}} F(x) = [f_1(x), f_2(x), ..., f_k(x)]^T$$

### Dominance Relations

Solution $x$ **dominates** $y$ if:
- $f_i(x) \leq f_i(y)$ for all $i$
- $f_j(x) < f_j(y)$ for at least one $j$

### Evolutionary Algorithms

```python
class NSGA2:
    """Non-dominated Sorting Genetic Algorithm II for multi-objective optimization."""

    def __init__(self, objectives, bounds, pop_size=50):
        self.objectives = objectives
        self.bounds = bounds
        self.pop_size = pop_size

    def non_dominated_sort(self, population, fitnesses):
        """Sort population into non-dominated fronts."""

        n = len(population)
        domination_count = np.zeros(n)
        dominated_by = [[] for _ in range(n)]
        fronts = [[]]

        # Calculate domination relationships
        for i in range(n):
            for j in range(i + 1, n):
                if self.dominates(fitnesses[i], fitnesses[j]):
                    dominated_by[i].append(j)
                    domination_count[j] += 1
                elif self.dominates(fitnesses[j], fitnesses[i]):
                    dominated_by[j].append(i)
                    domination_count[i] += 1

        # Find first front
        for i in range(n):
            if domination_count[i] == 0:
                fronts[0].append(i)

        # Find remaining fronts
        current_front = 0
        while fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front += 1
            fronts.append(next_front)

        return fronts[:-1]  # Remove empty last front

    def dominates(self, f1, f2):
        """Check if f1 dominates f2."""
        return all(f1 <= f2) and any(f1 < f2)

    def crowding_distance(self, fitnesses):
        """Calculate crowding distance for diversity."""

        n, m = fitnesses.shape
        distances = np.zeros(n)

        for obj in range(m):
            # Sort by objective
            sorted_idx = np.argsort(fitnesses[:, obj])

            # Boundary points get infinite distance
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf

            # Calculate distances for interior points
            obj_range = fitnesses[sorted_idx[-1], obj] - fitnesses[sorted_idx[0], obj]
            if obj_range > 0:
                for i in range(1, n - 1):
                    distances[sorted_idx[i]] += (
                        fitnesses[sorted_idx[i + 1], obj] -
                        fitnesses[sorted_idx[i - 1], obj]
                    ) / obj_range

        return distances

    def optimize(self, n_generations=100):
        """Run NSGA-II optimization."""

        # Initialize population
        population = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            (self.pop_size, len(self.bounds))
        )

        for generation in range(n_generations):
            # Evaluate objectives
            fitnesses = np.array([
                [obj(ind) for obj in self.objectives]
                for ind in population
            ])

            # Non-dominated sorting
            fronts = self.non_dominated_sort(population, fitnesses)

            # Create offspring
            offspring = self.create_offspring(population)

            # Combine parent and offspring
            combined_pop = np.vstack([population, offspring])
            combined_fit = np.vstack([
                fitnesses,
                np.array([[obj(ind) for obj in self.objectives]
                         for ind in offspring])
            ])

            # Select next generation
            new_population = []
            new_fitnesses = []

            for front in fronts:
                if len(new_population) + len(front) <= self.pop_size:
                    new_population.extend(combined_pop[front])
                    new_fitnesses.extend(combined_fit[front])
                else:
                    # Use crowding distance for selection
                    remaining = self.pop_size - len(new_population)
                    front_fit = combined_fit[front]
                    distances = self.crowding_distance(front_fit)
                    selected_idx = np.argsort(distances)[-remaining:]

                    for idx in selected_idx:
                        new_population.append(combined_pop[front[idx]])
                        new_fitnesses.append(combined_fit[front[idx]])
                    break

            population = np.array(new_population)
            fitnesses = np.array(new_fitnesses)

        # Return Pareto frontier
        final_fronts = self.non_dominated_sort(population, fitnesses)
        pareto_set = population[final_fronts[0]]
        pareto_front = fitnesses[final_fronts[0]]

        return pareto_set, pareto_front

    def create_offspring(self, population):
        """Generate offspring through crossover and mutation."""

        offspring = []

        for _ in range(self.pop_size):
            # Select parents (tournament selection)
            parents_idx = np.random.choice(len(population), 2, replace=False)
            parent1, parent2 = population[parents_idx]

            # Crossover (SBX)
            child = self.sbx_crossover(parent1, parent2)

            # Mutation (polynomial)
            child = self.polynomial_mutation(child)

            # Ensure bounds
            child = np.clip(child, self.bounds[:, 0], self.bounds[:, 1])

            offspring.append(child)

        return np.array(offspring)

    def sbx_crossover(self, parent1, parent2, eta=20):
        """Simulated binary crossover."""

        child = np.empty_like(parent1)

        for i in range(len(parent1)):
            if np.random.rand() < 0.5:
                # Perform crossover
                if abs(parent1[i] - parent2[i]) > 1e-10:
                    if parent1[i] < parent2[i]:
                        y1, y2 = parent1[i], parent2[i]
                    else:
                        y1, y2 = parent2[i], parent1[i]

                    beta = 1 + (2 * y1 / (y2 - y1))
                    alpha = 2 - beta ** (-(eta + 1))

                    u = np.random.rand()
                    if u <= 1 / alpha:
                        beta_q = (u * alpha) ** (1 / (eta + 1))
                    else:
                        beta_q = (1 / (2 - u * alpha)) ** (1 / (eta + 1))

                    child[i] = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
                else:
                    child[i] = parent1[i]
            else:
                child[i] = parent1[i]

        return child

    def polynomial_mutation(self, individual, eta=20, mutation_prob=0.1):
        """Polynomial mutation."""

        mutated = individual.copy()

        for i in range(len(individual)):
            if np.random.rand() < mutation_prob:
                y = individual[i]
                yl, yu = self.bounds[i]

                delta1 = (y - yl) / (yu - yl)
                delta2 = (yu - y) / (yu - yl)

                u = np.random.rand()

                if u <= 0.5:
                    delta_q = (2 * u + (1 - 2 * u) *
                              (1 - delta1) ** (eta + 1)) ** (1 / (eta + 1)) - 1
                else:
                    delta_q = 1 - (2 * (1 - u) + 2 * (u - 0.5) *
                                  (1 - delta2) ** (eta + 1)) ** (1 / (eta + 1))

                mutated[i] = y + delta_q * (yu - yl)
                mutated[i] = np.clip(mutated[i], yl, yu)

        return mutated

# Example: Three-objective insurance optimization
def premium_objective(x):
    """Minimize premium cost."""
    retention, primary_limit, excess_limit = x
    return 0.02 * primary_limit + 0.01 * excess_limit + 0.005 / (1 + retention)

def risk_objective(x):
    """Minimize retained risk."""
    retention, primary_limit, excess_limit = x
    return retention + max(0, 1 - primary_limit - excess_limit)

def volatility_objective(x):
    """Minimize earnings volatility."""
    retention, primary_limit, excess_limit = x
    return 0.5 * retention - 0.3 * primary_limit - 0.2 * excess_limit + 0.8

# Optimize
objectives = [premium_objective, risk_objective, volatility_objective]
bounds = np.array([
    [0, 1],    # retention
    [0, 2],    # primary_limit
    [0, 3]     # excess_limit
])

nsga2 = NSGA2(objectives, bounds, pop_size=100)
pareto_set, pareto_front = nsga2.optimize(n_generations=50)

# Visualize 3D Pareto frontier
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2])
ax.set_xlabel('Premium Cost')
ax.set_ylabel('Retained Risk')
ax.set_zlabel('Volatility')
ax.set_title('3D Pareto Frontier - Insurance Optimization')
plt.show()
```

## Hamilton-Jacobi-Bellman Equations

### Optimal Control Problem

$$V(t, x) = \max_{u \in U} \left\{ \int_t^T L(s, x(s), u(s)) ds + \Phi(x(T)) \right\}$$

### HJB Equation

$$\frac{\partial V}{\partial t} + \max_{u \in U} \left\{ L(t, x, u) + \nabla V \cdot f(t, x, u) + \frac{1}{2} \text{tr}(\sigma \sigma^T \nabla^2 V) \right\} = 0$$

with boundary condition: $V(T, x) = \Phi(x)$

### Insurance Application

```python
class HJBSolver:
    """Solve HJB equation for optimal insurance control."""

    def __init__(self, state_bounds, control_bounds, params):
        self.state_bounds = state_bounds
        self.control_bounds = control_bounds
        self.params = params

    def solve_hjb(self, nx=50, nu=20, nt=100):
        """Solve HJB using finite differences."""

        # Discretize state and control spaces
        x = np.linspace(self.state_bounds[0], self.state_bounds[1], nx)
        u = np.linspace(self.control_bounds[0], self.control_bounds[1], nu)
        t = np.linspace(0, self.params['T'], nt)

        dx = x[1] - x[0]
        dt = t[1] - t[0]

        # Initialize value function
        V = np.zeros((nt, nx))
        policy = np.zeros((nt, nx), dtype=int)

        # Terminal condition
        V[-1, :] = self.terminal_value(x)

        # Backward iteration
        for i in range(nt - 2, -1, -1):
            for j in range(nx):
                # Evaluate Hamiltonian for each control
                H = np.zeros(nu)

                for k in range(nu):
                    # Drift term
                    drift = self.drift(t[i], x[j], u[k])

                    # Diffusion term
                    diffusion = self.diffusion(t[i], x[j], u[k])

                    # Running cost
                    cost = self.running_cost(t[i], x[j], u[k])

                    # Finite difference approximations
                    if j > 0 and j < nx - 1:
                        V_x = (V[i + 1, j + 1] - V[i + 1, j - 1]) / (2 * dx)
                        V_xx = (V[i + 1, j + 1] - 2 * V[i + 1, j] +
                               V[i + 1, j - 1]) / (dx ** 2)
                    elif j == 0:
                        V_x = (V[i + 1, j + 1] - V[i + 1, j]) / dx
                        V_xx = 0
                    else:
                        V_x = (V[i + 1, j] - V[i + 1, j - 1]) / dx
                        V_xx = 0

                    # Hamiltonian
                    H[k] = cost + drift * V_x + 0.5 * diffusion**2 * V_xx

                # Optimal control and value update
                policy[i, j] = np.argmax(H)
                V[i, j] = V[i + 1, j] + dt * H[policy[i, j]]

        return V, policy, x, u, t

    def drift(self, t, x, u):
        """State drift under control u."""
        return self.params['mu'] * x - self.params['premium'](u)

    def diffusion(self, t, x, u):
        """State diffusion under control u."""
        return self.params['sigma'] * x * (1 - u)  # u reduces volatility

    def running_cost(self, t, x, u):
        """Instantaneous cost/reward."""
        return np.log(x) - self.params['risk_aversion'] * u**2

    def terminal_value(self, x):
        """Terminal value function."""
        return np.log(x)

    def plot_solution(self, V, policy, x, u, t):
        """Visualize HJB solution."""

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Value function surface
        T, X = np.meshgrid(t, x)
        axes[0, 0].contourf(T, X, V.T, levels=20, cmap='viridis')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Wealth')
        axes[0, 0].set_title('Value Function')
        axes[0, 0].colorbar = plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0])

        # Optimal policy
        axes[0, 1].contourf(T, X, u[policy].T, levels=20, cmap='RdYlBu')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Wealth')
        axes[0, 1].set_title('Optimal Insurance Level')
        axes[0, 1].colorbar = plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1])

        # Value function at different times
        times_to_plot = [0, len(t)//4, len(t)//2, 3*len(t)//4, -1]
        for idx in times_to_plot:
            axes[1, 0].plot(x, V[idx, :], label=f't = {t[idx]:.1f}')
        axes[1, 0].set_xlabel('Wealth')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_title('Value Function Evolution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Optimal control at different times
        for idx in times_to_plot:
            axes[1, 1].plot(x, u[policy[idx, :]], label=f't = {t[idx]:.1f}')
        axes[1, 1].set_xlabel('Wealth')
        axes[1, 1].set_ylabel('Optimal Insurance')
        axes[1, 1].set_title('Optimal Policy Evolution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# Solve HJB for insurance control
params = {
    'T': 10,
    'mu': 0.08,
    'sigma': 0.20,
    'risk_aversion': 0.5,
    'premium': lambda u: 0.02 * u  # Linear premium
}

hjb = HJBSolver(
    state_bounds=[1e5, 1e7],
    control_bounds=[0, 1],
    params=params
)

V, policy, x, u, t = hjb.solve_hjb()
hjb.plot_solution(V, policy, x, u, t)
```

## Numerical Methods

### Gradient-Based Methods

#### Gradient Descent

$$x_{k+1} = x_k - \alpha_k \nabla f(x_k)$$

#### Newton's Method

$$x_{k+1} = x_k - H_f(x_k)^{-1} \nabla f(x_k)$$

#### Quasi-Newton (BFGS)

$$x_{k+1} = x_k - \alpha_k B_k^{-1} \nabla f(x_k)$$

where $B_k$ approximates the Hessian.

### Derivative-Free Methods

```python
class OptimizationMethods:
    """Compare different optimization methods for insurance problems."""

    def __init__(self, objective, bounds):
        self.objective = objective
        self.bounds = bounds

    def gradient_descent(self, x0, learning_rate=0.01, max_iter=1000):
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

    def simulated_annealing(self, x0, temp=1.0, cooling=0.95, max_iter=1000):
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
            if delta < 0 or np.random.rand() < np.exp(-delta / temp):
                x = neighbor

                if f_neighbor < best_f:
                    best_x = neighbor.copy()
                    best_f = f_neighbor

            # Cool down
            temp *= cooling

            history.append(x.copy())

        return best_x, history

    def particle_swarm(self, n_particles=30, max_iter=100):
        """Particle swarm optimization."""

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

        g_best = p_best[np.argmin(p_best_scores)].copy()
        g_best_score = np.min(p_best_scores)

        history = [g_best.copy()]

        # PSO parameters
        w = 0.7  # Inertia
        c1 = 1.5  # Cognitive parameter
        c2 = 1.5  # Social parameter

        for _ in range(max_iter):
            for i in range(n_particles):
                # Update velocity
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (w * velocities[i] +
                                c1 * r1 * (p_best[i] - particles[i]) +
                                c2 * r2 * (g_best - particles[i]))

                # Update position
                particles[i] = particles[i] + velocities[i]
                particles[i] = np.clip(particles[i],
                                      self.bounds[:, 0],
                                      self.bounds[:, 1])

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

        methods = {
            'Gradient Descent': lambda: self.gradient_descent(x0),
            'Simulated Annealing': lambda: self.simulated_annealing(x0),
            'Particle Swarm': lambda: self.particle_swarm()
        }

        results = {}

        for name, method in methods.items():
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
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
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
    [0, 2],  # retention
    [0, 10],  # limit
    [0, 1]   # deductible
])

opt_methods = OptimizationMethods(insurance_objective, bounds)
x0 = np.array([1, 5, 0.5])

results = opt_methods.compare_methods(x0)
opt_methods.plot_convergence(results)

# Print results
for name, result in results.items():
    print(f"{name}:")
    print(f"  Solution: {result['solution']}")
    print(f"  Value: {result['value']:.6f}")
```

## Stochastic Control

### Stochastic Differential Equation

State dynamics:
$$dx_t = f(t, x_t, u_t)dt + \sigma(t, x_t, u_t)dW_t$$

### Dynamic Programming Principle

$$V(t, x) = \sup_{u \in \mathcal{U}} E\left[\int_t^{t+h} L(s, x_s, u_s)ds + V(t+h, x_{t+h}) \mid x_t = x\right]$$

### Implementation

```python
class StochasticControl:
    """Stochastic control for dynamic insurance decisions."""

    def __init__(self, dynamics, cost_function, terminal_cost):
        self.dynamics = dynamics
        self.cost_function = cost_function
        self.terminal_cost = terminal_cost

    def monte_carlo_control(self, x0, T, dt, n_sims=1000):
        """Monte Carlo approach to stochastic control."""

        n_steps = int(T / dt)
        times = np.linspace(0, T, n_steps + 1)

        # Control grid
        controls = np.linspace(0, 1, 11)

        # Initialize value function
        V = np.zeros((n_steps + 1, len(controls)))
        optimal_control = np.zeros(n_steps)

        # Terminal condition
        V[-1, :] = self.terminal_cost(x0)

        # Backward iteration
        for t in range(n_steps - 1, -1, -1):
            values = []

            for u in controls:
                # Simulate paths from this point
                total_cost = 0

                for _ in range(n_sims):
                    x = x0
                    cost = 0

                    # Simulate forward
                    for s in range(t, n_steps):
                        # Apply control
                        drift, diffusion = self.dynamics(times[s], x, u)

                        # Euler-Maruyama step
                        dW = np.random.randn() * np.sqrt(dt)
                        x = x + drift * dt + diffusion * dW

                        # Accumulate cost
                        cost += self.cost_function(times[s], x, u) * dt

                    # Add terminal cost
                    cost += self.terminal_cost(x)
                    total_cost += cost

                values.append(total_cost / n_sims)

            # Find optimal control
            optimal_idx = np.argmin(values)
            optimal_control[t] = controls[optimal_idx]
            V[t, :] = values

        return optimal_control, V, times

    def forward_simulation(self, x0, control_policy, T, dt, n_paths=100):
        """Simulate controlled system forward."""

        n_steps = int(T / dt)
        times = np.linspace(0, T, n_steps + 1)

        paths = np.zeros((n_paths, n_steps + 1))
        costs = np.zeros(n_paths)

        for i in range(n_paths):
            x = x0
            paths[i, 0] = x
            total_cost = 0

            for t in range(n_steps):
                # Get control
                if callable(control_policy):
                    u = control_policy(times[t], x)
                else:
                    u = control_policy[t]

                # Dynamics
                drift, diffusion = self.dynamics(times[t], x, u)

                # Step
                dW = np.random.randn() * np.sqrt(dt)
                x = x + drift * dt + diffusion * dW

                paths[i, t + 1] = x
                total_cost += self.cost_function(times[t], x, u) * dt

            costs[i] = total_cost + self.terminal_cost(x)

        return paths, costs, times

# Example: Dynamic retention control
def insurance_dynamics(t, x, u):
    """Wealth dynamics with insurance."""
    drift = 0.08 * x - 0.02 * u * x  # Growth minus premium
    diffusion = 0.15 * x * (1 - 0.5 * u)  # Insurance reduces volatility
    return drift, diffusion

def running_cost(t, x, u):
    """Cost function."""
    return -np.log(x) + 0.1 * u**2  # Maximize log wealth, penalize high insurance

def terminal_value(x):
    """Terminal value."""
    return -np.log(max(x, 1e-6))

# Solve control problem
controller = StochasticControl(
    dynamics=insurance_dynamics,
    cost_function=running_cost,
    terminal_cost=terminal_value
)

optimal_control, V, times = controller.monte_carlo_control(
    x0=1e6, T=5, dt=0.1, n_sims=500
)

# Simulate with optimal control
paths, costs, _ = controller.forward_simulation(
    x0=1e6, control_policy=optimal_control, T=5, dt=0.1
)

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Optimal control
axes[0, 0].plot(times[:-1], optimal_control)
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Optimal Insurance Level')
axes[0, 0].set_title('Optimal Control Policy')
axes[0, 0].grid(True, alpha=0.3)

# Sample paths
for i in range(min(20, len(paths))):
    axes[0, 1].plot(times, paths[i, :], alpha=0.3)
axes[0, 1].plot(times, np.mean(paths, axis=0), 'r-', linewidth=2, label='Mean')
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Wealth')
axes[0, 1].set_title('Controlled Wealth Paths')
axes[0, 1].legend()
axes[0, 1].set_yscale('log')

# Cost distribution
axes[1, 0].hist(costs, bins=30, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(np.mean(costs), color='r', linestyle='--', label=f'Mean: {np.mean(costs):.2f}')
axes[1, 0].set_xlabel('Total Cost')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Cost Distribution')
axes[1, 0].legend()

# Value function
axes[1, 1].imshow(V.T, aspect='auto', origin='lower', cmap='viridis')
axes[1, 1].set_xlabel('Time Step')
axes[1, 1].set_ylabel('Control Level')
axes[1, 1].set_title('Value Function')
axes[1, 1].colorbar = plt.colorbar(axes[1, 1].images[0], ax=axes[1, 1])

plt.tight_layout()
plt.show()
```

## Convergence Criteria

### Numerical Convergence

1. **Gradient norm**: $\|\nabla f(x_k)\| < \epsilon$
2. **Step size**: $\|x_{k+1} - x_k\| < \epsilon$
3. **Function value**: $|f(x_{k+1}) - f(x_k)| < \epsilon$
4. **Relative change**: $\frac{|f(x_{k+1}) - f(x_k)|}{|f(x_k)|} < \epsilon$

### Statistical Convergence

```python
def check_convergence(history, window=10, threshold=1e-4):
    """Check various convergence criteria."""

    if len(history) < window:
        return False, {}

    recent = history[-window:]

    # Calculate metrics
    mean_change = np.mean(np.diff(recent))
    std_change = np.std(np.diff(recent))
    trend = np.polyfit(range(window), recent, 1)[0]

    # Convergence criteria
    criteria = {
        'mean_change': abs(mean_change) < threshold,
        'std_change': std_change < threshold,
        'trend': abs(trend) < threshold,
        'plateau': np.std(recent) / np.mean(recent) < 0.01
    }

    converged = all(criteria.values())

    return converged, criteria
```

## Practical Implementation

### Complete Insurance Optimizer

```python
class CompleteInsuranceOptimizer:
    """Production-ready insurance optimization system."""

    def __init__(self, company_profile):
        self.profile = company_profile
        self.results = {}

    def optimize(self):
        """Run complete optimization workflow."""

        print("Starting insurance optimization...")

        # Step 1: Risk assessment
        self.assess_risks()

        # Step 2: Generate Pareto frontier
        self.generate_pareto_frontier()

        # Step 3: Select optimal point
        self.select_optimal_solution()

        # Step 4: Validate solution
        self.validate_solution()

        # Step 5: Generate recommendations
        self.generate_recommendations()

        return self.results

    def assess_risks(self):
        """Assess company risk profile."""

        print("Assessing risks...")

        # Simulate loss scenarios
        losses = self.simulate_losses(n_years=10, n_sims=10000)

        self.results['risk_metrics'] = {
            'expected_annual_loss': np.mean(losses),
            'var_95': np.percentile(losses, 95),
            'cvar_95': np.mean(losses[losses > np.percentile(losses, 95)]),
            'max_loss': np.max(losses)
        }

    def simulate_losses(self, n_years, n_sims):
        """Simulate loss scenarios."""

        annual_losses = []

        for _ in range(n_sims):
            total = 0
            for year in range(n_years):
                # Frequency
                n_claims = np.random.poisson(self.profile['claim_frequency'])

                # Severity
                if n_claims > 0:
                    claims = np.random.lognormal(
                        np.log(self.profile['claim_severity_mean']),
                        self.profile['claim_severity_std'],
                        n_claims
                    )
                    total += np.sum(claims)

            annual_losses.append(total / n_years)

        return np.array(annual_losses)

    def generate_pareto_frontier(self):
        """Generate multi-objective Pareto frontier."""

        print("Generating Pareto frontier...")

        # Define objectives
        def cost_objective(x):
            retention, limit = x
            return 0.01 * limit + 0.02 * max(0, limit - retention)

        def risk_objective(x):
            retention, limit = x
            return retention - 0.5 * limit

        # Generate frontier
        pareto_points = []
        for weight in np.linspace(0, 1, 20):
            def combined(x):
                return weight * cost_objective(x) + (1 - weight) * risk_objective(x)

            result = minimize(
                combined,
                x0=[self.profile['assets'] * 0.01, self.profile['assets'] * 0.1],
                bounds=[
                    (0, self.profile['assets'] * 0.05),
                    (0, self.profile['assets'] * 0.5)
                ],
                method='L-BFGS-B'
            )

            if result.success:
                pareto_points.append({
                    'retention': result.x[0],
                    'limit': result.x[1],
                    'cost': cost_objective(result.x),
                    'risk': risk_objective(result.x),
                    'weight': weight
                })

        self.results['pareto_frontier'] = pareto_points

    def select_optimal_solution(self):
        """Select optimal point from Pareto frontier."""

        print("Selecting optimal solution...")

        # Use utility function or business rules
        frontier = self.results['pareto_frontier']

        # Example: Minimize cost subject to risk constraint
        valid_points = [p for p in frontier if p['risk'] < self.profile['risk_tolerance']]

        if valid_points:
            optimal = min(valid_points, key=lambda p: p['cost'])
        else:
            # Fallback to minimum risk
            optimal = min(frontier, key=lambda p: p['risk'])

        self.results['optimal_solution'] = optimal

    def validate_solution(self):
        """Validate optimal solution through simulation."""

        print("Validating solution...")

        solution = self.results['optimal_solution']

        # Run detailed simulation
        n_sims = 10000
        outcomes = []

        for _ in range(n_sims):
            wealth = self.profile['assets']

            for year in range(10):
                # Growth
                wealth *= np.random.lognormal(0.08, 0.15)

                # Losses
                loss = self.simulate_losses(1, 1)[0]
                retained = min(loss, solution['retention'])

                # Premium
                premium = solution['cost'] * wealth

                wealth = max(0, wealth - retained - premium)

                if wealth == 0:
                    break

            outcomes.append(wealth)

        self.results['validation'] = {
            'survival_rate': np.mean(np.array(outcomes) > 0),
            'median_final_wealth': np.median([w for w in outcomes if w > 0]),
            'growth_rate': np.mean([np.log(w/self.profile['assets'])/10
                                   for w in outcomes if w > 0])
        }

    def generate_recommendations(self):
        """Generate actionable recommendations."""

        print("Generating recommendations...")

        optimal = self.results['optimal_solution']
        validation = self.results['validation']

        recommendations = []

        # Primary recommendation
        recommendations.append({
            'priority': 'HIGH',
            'action': f"Set retention at ${optimal['retention']:,.0f}",
            'rationale': 'Optimal balance of cost and risk'
        })

        recommendations.append({
            'priority': 'HIGH',
            'action': f"Purchase coverage up to ${optimal['limit']:,.0f}",
            'rationale': f"Ensures {validation['survival_rate']:.1%} survival probability"
        })

        # Additional recommendations based on analysis
        if validation['growth_rate'] < 0.05:
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'Consider increasing coverage',
                'rationale': 'Current growth rate below target'
            })

        self.results['recommendations'] = recommendations

    def plot_results(self):
        """Visualize optimization results."""

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Pareto frontier
        frontier = self.results['pareto_frontier']
        costs = [p['cost'] for p in frontier]
        risks = [p['risk'] for p in frontier]

        axes[0, 0].plot(costs, risks, 'b-o')
        optimal = self.results['optimal_solution']
        axes[0, 0].plot(optimal['cost'], optimal['risk'], 'r*', markersize=15)
        axes[0, 0].set_xlabel('Cost')
        axes[0, 0].set_ylabel('Risk')
        axes[0, 0].set_title('Pareto Frontier')
        axes[0, 0].grid(True, alpha=0.3)

        # Risk metrics
        metrics = self.results['risk_metrics']
        labels = list(metrics.keys())
        values = list(metrics.values())

        axes[0, 1].bar(range(len(labels)), values)
        axes[0, 1].set_xticks(range(len(labels)))
        axes[0, 1].set_xticklabels(labels, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].set_title('Risk Metrics')

        # Validation results
        val = self.results['validation']
        axes[1, 0].bar(['Survival\nRate', 'Growth\nRate'],
                      [val['survival_rate'], val['growth_rate']])
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_title('Validation Metrics')

        # Recommendations
        recs = self.results['recommendations']
        rec_text = '\n\n'.join([f"{r['priority']}: {r['action']}\n  → {r['rationale']}"
                                for r in recs[:3]])
        axes[1, 1].text(0.1, 0.5, rec_text, fontsize=10, verticalalignment='center')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Top Recommendations')

        plt.tight_layout()
        plt.show()

# Run complete optimization
company = {
    'assets': 50_000_000,
    'revenue': 100_000_000,
    'claim_frequency': 5,
    'claim_severity_mean': 100_000,
    'claim_severity_std': 2,
    'risk_tolerance': 0.5
}

optimizer = CompleteInsuranceOptimizer(company)
results = optimizer.optimize()
optimizer.plot_results()

# Print summary
print("\n" + "="*50)
print("OPTIMIZATION COMPLETE")
print("="*50)
print(f"Optimal Retention: ${results['optimal_solution']['retention']:,.0f}")
print(f"Optimal Limit: ${results['optimal_solution']['limit']:,.0f}")
print(f"Annual Cost: ${results['optimal_solution']['cost']:,.0f}")
print(f"Survival Rate: {results['validation']['survival_rate']:.1%}")
print(f"Expected Growth: {results['validation']['growth_rate']:.2%}")
```

## Key Takeaways

1. **Constrained optimization**: Balance objectives with real-world constraints
2. **Pareto efficiency**: No single optimal solution for multi-objective problems
3. **HJB equations**: Powerful framework for dynamic optimization
4. **Numerical methods**: Choose appropriate method for problem structure
5. **Stochastic control**: Account for uncertainty in dynamic decisions
6. **Convergence monitoring**: Essential for reliable solutions
7. **Practical implementation**: Combine theory with business constraints

## Next Steps

- [Chapter 5: Statistical Methods](05_statistical_methods.md) - Validation and testing
- [Chapter 3: Insurance Mathematics](03_insurance_mathematics.md) - Insurance-specific models
- [Chapter 1: Ergodic Economics](01_ergodic_economics.md) - Theoretical foundation
