# Implementation of a Genetic Algorithm and Particle Swarm Optimization algorithm to solve the Traveling Salesman Problem

We created a pure python/numpy/pandas package implementing two gradient-free optimization algorithms, the Genetic Algorithm (GA) and Particle Swarm Optimization (PSO), to solve a version of the Traveling Salesman Problem (TSP), a well-known NP-hard problem in optimization and computer science.

## Why solve the TSP?

The TSP is an important problem in computational mathematics and has applications in a myriad of fields including transportation, electronics and genetics.

## Why create this package?

1. One major advantage of gradient-free optimization techniques like the GA and PSO to solve a problem like TSP is that they can identify many permutations of cities that are close to the ideal solution while converging on to the global optimum solution. These close-to-optimal solutions can be used as starting points for further refinement or as alternative solutions in situations when the ideal solution is not needed or is too costly. 

2. By containing both algorithms, this package allows users to quickly and conveniently compare/contrast the performance of each algorithm with various parameter conditions to determine which method is the best option for a given context.

3. The need for comprehensive problem-solving packages that also allow for instructional utility is growing. By generating cities on a unit circle where the optimal solution is always known, providing built-in visualizations and results analysis,  as well as having a variety of methods and parameters to play around with, this package becomes a powerful pedagogical tool for teaching students about the field of optimization. 

4. We wrote this package in a highly modular fashion to allow users to adapt it to handle similar problems such as the vehicle routing problem or to intake non-synthetic, real-world data.

## How to use this package?


```python
import opt_TSP
```


```python
GA_test1 = GA(num_generations = 1000, pop_size = 500, num_cities = 80, elitism_frac = 0.2, selection_method = "Roulette", mating_method = "Random", mutation_rate = 1)
GA_test1.run_GA()
```


```python
GA_test1.plot_results()
```

```python
GA_test1.plot_best_path()
```


```python
GA_test1.plot_error(num_simulations = 20)
```
