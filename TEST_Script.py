#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Team #2
Authors: Daniel Gordin & Kyle Mikami
"""

import opt_TSP

#CityData Class
test_CityData = opt_TSP.CityData(num_cities = 30)
test_CityData.plot_cities(test_CityData.optimal_solution)

#Run Genetic Algorithm
test_GA = opt_TSP.GA(num_generations = 200, pop_size = 200, num_cities = 30, elitism_frac = 0.25, selection_method = "Roulette", mating_method = "Random", mutation_rate = 1)
test_GA.run_GA()
test_GA.plot_results()
test_GA.plot_best_path()
test_GA.plot_error(num_simulations = 5, print_progress = True)

#Run Particle Swarm Optimization algorithm
test_PSO = opt_TSP.PSO(num_iterations = 200, num_particles = 200, num_cities = 30, current_weight = 0.3, pbest_weight = 0.5, gbest_weight = 0.9)
test_PSO.run_PSO()
test_PSO.plot_results()
test_PSO.plot_best_path()
test_PSO.plot_error(num_simulations = 5, print_progress = True)
