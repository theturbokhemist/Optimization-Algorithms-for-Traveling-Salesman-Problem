#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Team #2
Authors: Daniel Gordin & Kyle Mikami
"""

import opt_TSP

z = GA()        
z.plot_error(5)
z.plot_best_path()

GA_test1 = GA(num_generations = 1000, pop_size = 500, num_cities = 80, elitism_frac = 0.2, selection_method = "Roulette", mating_method = "Random", mutation_rate = 1)
GA_test1.run_GA()
GA_test1.plot_results()
GA_test1.CityData.plot_cities(GA_test1.best_path)


GA_test2 = GA(num_generations = 1000, pop_size = 500, num_cities = 80, elitism_frac = 0.2, selection_method = "Fittest Half", mating_method = "Random", mutation_rate = 1)
GA_test2.run_GA()
GA_test2.plot_results()
GA_test2.CityData.plot_cities(GA_test2.best_path)
GA_test2.CityData.plot_cities(GA_test2.CityData.optimal_solution)
round(GA_test2.CityData.optimal_solution_dist, 5)
GA_test2.CityData.total_dist(path = GA_test2.best_path)



GA_test3 = GA(num_generations = 1000, pop_size = 500, num_cities = 80, elitism_frac = 0.2, selection_method = "Roulette", mating_method = "Fittest Paired", mutation_rate = 1)
GA_test3.run_GA()
GA_test3.plot_results()
GA_test3.CityData.plot_cities(GA_test3.best_path)


GA_test4 = GA(num_generations = 1000, pop_size = 500, num_cities = 80, elitism_frac = 0.2, selection_method = "Fittest Half", mating_method = "Fittest Paired", mutation_rate = 1)
GA_test4.run_GA()
GA_test4.plot_results()
GA_test4.CityData.plot_cities(GA_test4.best_path)


GA_test5 = GA(num_generations = 1000, pop_size = 500, num_cities = 80, elitism_frac = 0.2, selection_method = "Roulette", mating_method = "Fittest Paired", mutation_rate = 4)
GA_test5.run_GA()
GA_test5.plot_results()
GA_test5.CityData.plot_cities(GA_test5.best_path)

GA_test6 = GA(num_generations = 1000, pop_size = 500, num_cities = 80, elitism_frac = 0.2, selection_method = "Roulette", mating_method = "Fittest Paired", mutation_rate = 0)
GA_test6.run_GA()
GA_test6.plot_results()
GA_test6.CityData.plot_cities(GA_test6.best_path)

GA_test7 = GA(num_generations = 1000, pop_size = 500, num_cities = 80, elitism_frac = 0.2, selection_method = "Roulette", mating_method = "Fittest Paired", mutation_rate = 2)
GA_test7.run_GA()
GA_test7.plot_results()
GA_test7.CityData.plot_cities(GA_test7.best_path)

GA_test8 = GA(num_generations = 1000, pop_size = 500, num_cities = 80, elitism_frac = 0.4, selection_method = "Roulette", mating_method = "Fittest Paired", mutation_rate = 1)
GA_test8.run_GA()
GA_test8.plot_results()
GA_test8.CityData.plot_cities(GA_test8.best_path)



#Run GA 20 times and record error
error_list = []
for i in range(1,21):
    GA_test1 = GA(num_generations = 500, pop_size = 400, num_cities = 50, elitism_frac = 0.25, selection_method = "Roulette", mating_method = "Random", mutation_rate = 1)
    GA_test1.run_GA()
    error_list.append(GA_test1.error)
    print(i)
    
avg = sum(error_list)/len(error_list)
print(avg)
GA_test1.mean = avg
GA_test1.error = error_list
GA_test1.plot_results()
#Plot error

for error in error_list:
    x = plt.plot(error_list)
    x = plt.xlabel('Iteration')
    x = plt.ylabel('Error')
    x = plt.title("GA Error by # of Runs")
    x = plt.axhline(y = avg, color='r', linestyle='-')

plt.show()



error_list = []
for i in range(1,21):
    GA_test2 = GA(num_generations = 500, pop_size = 400, num_cities = 50, elitism_frac = 0.25, selection_method = "Fittest Half", mating_method = "Random", mutation_rate = 1)
    GA_test2.run_GA()
    error_list.append(GA_test2.error)
    print(i)

    
avg = sum(error_list)/len(error_list)
print(avg)
GA_test2.mean = avg
GA_test2.error = error_list
GA_test2.plot_results()
#Plot error

for error in error_list:
    x = plt.plot(error_list)
    x = plt.xlabel('Iteration')
    x = plt.ylabel('Error')
    x = plt.title("GA Error by # of Runs")
    x = plt.axhline(y = avg, color='r', linestyle='-')

plt.show()



error_list = []
for i in range(1,21):
    GA_test3 = GA(num_generations = 500, pop_size = 400, num_cities = 50, elitism_frac = 0.25, selection_method = "Roulette", mating_method = "Fittest Paired", mutation_rate = 1)
    GA_test3.run_GA()
    error_list.append(GA_test3.error)
    print(i)
    
avg = sum(error_list)/len(error_list)
print(avg)
GA_test3.mean = avg
GA_test3.error = error_list
GA_test3.plot_results()

#Plot error

for error in error_list:
    x = plt.plot(error_list)
    x = plt.xlabel('Iteration')
    x = plt.ylabel('Error')
    x = plt.title("GA Error by # of Runs")
    x = plt.axhline(y = avg, color='r', linestyle='-')

plt.show()



error_list = []
for i in range(1,21):
    GA_test4 = GA(num_generations = 500, pop_size = 400, num_cities = 50, elitism_frac = 0.25, selection_method = "Fittest Half", mating_method = "Fittest Paired", mutation_rate = 1)
    GA_test4.run_GA()
    error_list.append(GA_test4.error)
    print(i)
    
avg = sum(error_list)/len(error_list)
print(avg)
GA_test4.mean = avg
GA_test4.error = error_list
GA_test4.plot_results()

#Plot error

for error in error_list:
    x = plt.plot(error_list)
    x = plt.xlabel('Iteration')
    x = plt.ylabel('Error')
    x = plt.title("GA Error by # of Runs")
    x = plt.axhline(y = avg, color='r', linestyle='-')

plt.show()


error_list = []
for i in range(1,21):
    GA_test5 = GA(num_generations = 500, pop_size = 400, num_cities = 50, elitism_frac = 0.45, selection_method = "Fittest Half", mating_method = "Random", mutation_rate = 1)
    GA_test5.run_GA()
    error_list.append(GA_test5.error)
    print(i)

    
avg = sum(error_list)/len(error_list)
print(avg)
GA_test5.mean = avg
GA_test5.error = error_list
GA_test5.plot_results()
#Plot error

for error in error_list:
    x = plt.plot(error_list)
    x = plt.xlabel('Iteration')
    x = plt.ylabel('Error')
    x = plt.title("GA Error by # of Runs")
    x = plt.axhline(y = avg, color='r', linestyle='-')

plt.show()



error_list = []
for i in range(1,21):
    GA_test6 = GA(num_generations = 500, pop_size = 400, num_cities = 50, elitism_frac = 0.25, selection_method = "Fittest Half", mating_method = "Random", mutation_rate = 3)
    GA_test6.run_GA()
    error_list.append(GA_test6.error)
    print(i)

    
avg = sum(error_list)/len(error_list)
print(avg)
GA_test6.mean = avg
GA_test6.error = error_list
GA_test6.plot_results()
#Plot error

for error in error_list:
    x = plt.plot(error_list)
    x = plt.xlabel('Iteration')
    x = plt.ylabel('Error')
    x = plt.title("GA Error by # of Runs")
    x = plt.axhline(y = avg, color='r', linestyle='-')

plt.show()
