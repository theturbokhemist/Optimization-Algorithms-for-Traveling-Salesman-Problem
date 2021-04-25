#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Team #2
Authors: Daniel Gordin & Kyle Mikami
"""

#Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random as rnd
import statistics
import itertools
from scipy.spatial import ConvexHull


class CityData:
    
    """
    CityData class generates, cointains, and visualizes the coordinates data
    for the "cities" in the Traveling Salesman Problem. Cities are all contained on a unit circle centered at the origin.
    """

    def __init__(self, num_cities = 10):
        
        """
        Initializes numpy 2D-array of city coordinates
        Param num_cities: Number of cities in the array
        """
                
        #Generate the matrix of coordinates
        self.num_cities = num_cities
		
        x = np.random.rand(num_cities)
        #formula for unit circle center at (0,0)
        y = np.sqrt(1 - x*x)
		
        x = x*(2*np.random.randint(0,2,size=(num_cities))-1)
        y = y*(2*np.random.randint(0,2,size=(num_cities))-1)

        cities_mat = np.vstack((x,y)).T
        self.cities_mat = cities_mat
        
        self.optimal_solution = self.calc_optimal_solution()
        self.optimal_solution_dist = self.total_dist(self.optimal_solution)
        
        self.cities_df = pd.DataFrame(cities_mat, columns= ['X', "Y"])
        
        
    def plot_cities(self, path):
        
        """
        Plots and connects the city coordinates for a given permutation
        Param path: A permutation of the city IDs
        """
        #rearranges cities mat so that it is in the order of the given permutation
        mat = self.cities_mat[path, :]
        
        #appends the coordinates of the first city to the end so route is complete 
        mat = np.vstack((mat, mat[0,]))
            
        df = pd.DataFrame(mat, columns= ['X', "Y"])
            
        plt.scatter('X', 'Y', data = df, color = "red")
        plt.plot('X', 'Y', data = df)
        plt.title('City Coordinates')
        plt.xlabel('X')
        plt.ylabel('Y')

        
    def calc_distance(self, city1, city2):
        
        """
        Calculates the distance between 2 cities
        Param city1: Coordinates of 1 city
        Param city2: Coordinates of another city
        Returns a numeric value which represents the distance between two cities.
        """
        
        #distance formula
        x = self.cities_mat[city2, ][0] - self.cities_mat[city1, ][0]
        y = self.cities_mat[city2, ][1] - self.cities_mat[city1, ][1]
        
        d = math.sqrt(x*x + y*y)
        return d
    
    def total_dist(self, path):
        
        """
        Calculates the distance it takes to travel to each city. The order is determined by the given permutation.
        Param path: A permutation of the city IDs
        Returns a numeric value which represents the distance it takes to travel to each city.
        """
        
        perm_array = np.append(path, path[0])
        
        current_sum = 0
        for i in range(len(perm_array) - 1):
    
            current_sum = current_sum + self.calc_distance(city1 = perm_array[i], city2 = perm_array[i + 1])
            
        return current_sum
    
    def calc_optimal_solution(self):
        
        """
        Calculates the "optinal solution" which is the permutation of cities which yields the lowest total distance.
        Returns a list of city IDs
        """
        
        #Convex Hull algorithm
        hull = ConvexHull(self.cities_mat)
        return hull.vertices    


#GA Class      
class GA:
    
    """
    GA class contains the methods and attributes that employ a genetic
    algorithm to optimize the Traveling Salesman Problem
    """
    
    def __init__(self, num_generations = 10, pop_size = 20, num_cities = 10, elitism_frac = 0.25, selection_method = "Roulette", mating_method = "Random", mutation_rate = 1):
        
        """
        Creates an instance of a CityData object well as all combinations of ranges for a list of length num_cities
        Param num_cities: Number of cities to optimize for TSP
        Param num_generations: Number of generations to create
        Param pop_size: Number of individuals in a generation
        Param elitism_frac: The percent of individuals with the best fitness scores that will be included in the mating pool and subsequent generation
        Param selection_method: The way the individuals are selected for each mating pool. 
        If "Roulette", the individuals are selected randomly with a probability determined by their fitness scores. If "Fittest Half", the top half of fittest individuals are chosen.
        Param mating_method: The way the individuals are selected to mate.
        If "Random", individuals are chosen to mate at random. If "Fittest Paired", the fittest individuals are paired up to mate.
        Param mutation_rate: The amount of random times two cities swap places for all children produced from a mating 
        """
        
        self.num_generations = num_generations    
        self.pop_size = pop_size 
        self.elitism_frac = elitism_frac 
        self.selection_method = selection_method   
        self.mating_method = mating_method
        self.CityData = CityData(num_cities = num_cities)         
        self.range_combos = list(itertools.combinations(range(0,self.CityData.num_cities), 2)) + [(self.CityData.num_cities - 1, self.CityData.num_cities)]
        self.mutation_rate = mutation_rate
        self.mean_distances = []
        self.min_score = []
            
    def run_GA(self):
        
        """
        Runs the Genetic Algorithm
        """
        
        initial_pop = self.initialize_population()
        self.current_generation = initial_pop

        for i in range(0, self.num_generations):  
            self.next_generation(current_generation = self.current_generation) 
        
        self.ranked_final_population = self.rank_population(population = self.current_generation)
        self.best_path = self.ranked_final_population.Path[0]
        
        self.error = round(self.CityData.total_dist(path = self.best_path), 5) - round(self.CityData.optimal_solution_dist,5)

                 
    def initialize_population(self):
        
        """
        Creates the 1st generation of individuals. Each individual is randomly generated.
        """
        
        population = []
        
        for i in range(self.pop_size):  
            population.append(np.array(rnd.sample(range(self.CityData.num_cities), self.CityData.num_cities)))

        return population
     
    def rank_population(self, population):
        
        """
        Ranks each individual of a population based on their fitness score (the total distance of their permutation)
        Param population: A list of each individual in the population/generation. Each individual is a permutation list.
        Returns a sorted pandas dataframe that includes the individual IDs, their fitness scores, their inverse scores, 
        the fraction each individuals inverse score is of the sum of all the fitness scores (probability), and their permutations.
        """
                
        fitness_scores = []
        
        for i in range(self.pop_size):  
            fitness_scores.append(self.CityData.total_dist(population[i]))
            
        self.mean_distances.append(statistics.mean(fitness_scores))
        df = pd.DataFrame({'Ind':np.argsort(fitness_scores), 'Score':[fitness_scores[i] for i in np.argsort(fitness_scores)]})
        df['Inv_Score'] = 1/df['Score']
        df['Percent'] = df.Inv_Score/df.Inv_Score.sum()
        
        pop_ordered = []
        for i in range(0, len(df)):
            pop_ordered.append(population[df.Ind[i]])
            
        df['Path'] = pop_ordered
        self.min_score.append(df.Score[0])

        return df
    
    def mating_pool(self, ranked_population):
        
        """
        Determines which individuals will be available for mating to create the next generation
        Param ranked_population: A sorted pandas dataframe that includes the ID's and probabilities of being selected for each individual in a population/generation. 
        Returns a list of individual IDs
        """
        
        mating_pool = []
        num_elites = math.ceil(self.elitism_frac*len(ranked_population))

        for i in range(0, num_elites):
            mating_pool.append(ranked_population.Ind[i])
         
        if self.selection_method == "Roulette":
            mating_pool = mating_pool + rnd.choices(population = ranked_population.Ind, weights = ranked_population.Percent, k = (len(ranked_population) - num_elites))
               
        elif self.selection_method == "Fittest Half":
            
            for i in range(0, math.ceil(len(ranked_population)/2)): 
                mating_pool.append(ranked_population.Ind[i])
            
        return mating_pool
    
    
    def mate(self, parent1, parent2):
        
        """
        Performs ordered crossover on the permutations of two "parent" individuals to create two "children" permutations
        Param parent1: A permutation of city IDs
        Param parent2: A permutation of city IDs
        Returns a list of two permutation arrays
        """
        
        #ordered crossover
        start, end = self.range_combos[rnd.randrange(0, len(self.range_combos))]
        
        c1 = [None]*self.CityData.num_cities
        c2 = [None]*self.CityData.num_cities
        
        temp1 = []
        temp2 = []
        
        for i in range(start, end):
            
            c1[i] = parent1[i]
            temp1.append(parent1[i])
            
            c2[i] = parent2[i]
            temp2.append(parent2[i])

        counter1 = 0
        counter2 = 0
        for i in range(0, len(c1)):
            
            if c1[i] == None:
                
                for j in range(i + counter1, len(parent2)):     
                    if parent2[j] in temp1:      
                        counter1 += 1
                    
                    else:  
                        c1[i] = parent2[j]
                        break
                    
                for j in range(i + counter2, len(parent1)): 
                    if parent1[j] in temp2:   
                        counter2 += 1

                    else:       
                        c2[i] = parent1[j]
                        break
                
            else:
                counter1 = counter1 - 1
                counter2 = counter2 - 1
                                 
        return self.mutate(c1) + self.mutate(c2)
    
        
    def mutate(self, individual):
        
        """
        Swaps the positions of two cities in a list of city IDs       
        Param individual: A permutation of city IDs
        Returns a permutation of city IDs
        """
        
        if self.mutation_rate > 0:
            
            for i in range(0, self.mutation_rate):  
                start, end = self.range_combos[rnd.randrange(0, len(self.range_combos)-1)]
                individual[start], individual[end] =  individual[end], individual[start]
        
            return [np.array(individual)]
        
        else:
            
            return [np.array(individual)]
        
           
    def next_generation(self, current_generation):
        
        """
        Pairs up individuals from the current generation to create the next generation of individuals   
        Param current_generation: A list of each individual in the generation. Each individual is a permutation array.
        Returns a list of individuals where each individual is a permutation array.
        """
        
        ranked_pop = self.rank_population(current_generation)
        new_generation = []
        num_elites = math.ceil(self.elitism_frac*len(ranked_pop))
        for i in range(0, num_elites):
            new_generation.append(current_generation[ranked_pop.Ind[i]])
        
        mating_pool = self.mating_pool(ranked_pop)

        if self.mating_method == "Random":
            
            pool = []
            for i in range(0, len(mating_pool)):
                pool.append(current_generation[mating_pool[i]])
             
            while len(new_generation) < self.pop_size:
                  
                p1 = np.random.choice(range(0,len(pool)), 1, replace=False)[0]
                p2 = np.random.choice(range(0,len(pool)), 1, replace=False)[0]
            
                new_generation = new_generation + self.mate(parent1 = pool[p1], parent2 = pool[p2])
            
        elif self.mating_method == "Fittest Paired":
            
            mating_pool = list(set(mating_pool))
            
            order = list(ranked_pop.Ind)
            mating_pool = 2*sorted(mating_pool, key=lambda mating_pool: order.index(mating_pool))
            
            pool = []
            for i in range(0, len(mating_pool)):
                pool.append(current_generation[mating_pool[i]])
            
            k = 0
            while len(new_generation) < self.pop_size:
                                
                p1 = k 
                p2 = k + 1
            
                new_generation = new_generation + self.mate(parent1 = pool[p1], parent2 = pool[p2])
                k = k + 2
                
        self.current_generation = new_generation
        return new_generation[0:self.pop_size]
    
    def plot_results(self):
        
        """
        Creates two subplots of the results of the genetic algorithm. One subplot shows the mean total distance per generation. 
        Another shows the total_distance of the fittest individual per generation.
        """
        
        x = list(range(1, self.num_generations + 2))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey = 'row')
        fig.suptitle('Optimization Results')
        
        ax1.plot(x, self.mean_distances)
        ax1.set_title("Mean Path Length")
        ax1.set(xlabel = "Generation", ylabel = "Path Length")
        ax1.axhline(y= self.CityData.optimal_solution_dist, color='r', linestyle='-')

        ax2.plot(x, self.min_score)
        ax2.set_title("Minimum Path Length")
        ax2.set(xlabel = "Generation")
        fig.subplots_adjust(hspace=0.05)
        ax2.axhline(y= self.CityData.optimal_solution_dist, color='r', linestyle='-')
        
        
    def plot_error(self, num_simulations, print_progress = True):
        
        """
        Simulates the optimization algorithm multiple times and plots the error per simulation as well as the average error.
        """
        
        self.num_simulations = num_simulations
        error_list = []
        
        for i in range(num_simulations):
            self.run_GA()
            error_list.append(self.error)
            
            if print_progress:
                print(i)
    
        avg = sum(error_list)/len(error_list)
        self.mean_error = avg
        
        mat = np.vstack((list(range(1, num_simulations + 1)),error_list)).T
        
        df = pd.DataFrame(mat, columns= ['X', "Y"])
        plt.scatter('X', 'Y', data = df, color = "green")
        plt.plot('X', 'Y', data = df)
        plt.title('Error per Simulation')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.axhline(y = avg, color='r', linestyle='-')
        
        
    def plot_best_path(self):
        
        """
        Plots the path of the best solution found
        """
        
            
        self.CityData.plot_cities(self.best_path)


        

#PSO#
#Particle Class
class Particle:
    
    """Initializes particle (permutation)"""
    
    def __init__(self, permutation):
        """
        Initializes particle (permutation)
        Param permutation: Possible route of cities
        """
        self.current = permutation     # current particle permutation
        self.pbest= permutation          # best individual particle permutation
        
class PSO:
    
    def __init__(self, num_iterations = 100, num_cities = 10, num_particles = 100, current_weight = 0.3, pbest_weight = 0.5, gbest_weight = 0.9):  
        """
        Creates the swarm
        Params:
        Num_iterations, how many times you want to run algorithm
        Num_cities, number of cities in TSP
        Num_particles, number of permutations of routes
        Current_weight, how much to weigh current permutation
        Pbest_weight, how much to weigh particle's best permutation
        Gbest_weight, how much to weigh the global best permutation
        """
        
        self.num_cities = num_cities
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        
        self.CityData = CityData(num_cities = num_cities)
        
        self.current_weight = current_weight
        self.pbest_weight = pbest_weight
        self.gbest_weight = gbest_weight
    
        self.current_mean = []
        self.gbest_list = []
        
    def initialize_swarm(self):
        
        """Initializes the swarm"""
        swarm = []
        dist = []
        
        #Iterates through each permutation and appends each permutation and total distance to swarm and dist lists
        for i in range(self.num_particles):
                            
            swarm.append(Particle(permutation = np.array(rnd.sample(range(self.CityData.num_cities), self.CityData.num_cities))))
            
            temp_dist = self.CityData.total_dist(path = swarm[i].current)
            
            dist.append(temp_dist)
            swarm[i].current_dist = temp_dist
            swarm[i].pbest_dist = temp_dist
            
        self.swarm = swarm
        # self.dist = dist
        
        #Sets gbest_distance and gbest permutation
        self.gbest_dist = dist[np.argmin(dist)]
        self.gbest = swarm[np.argmin(dist)].current
        
        #Appends mean distance to current and gbest lists
        self.current_mean.append(statistics.mean(dist))
        self.gbest_list.append(self.gbest_dist)
        
    def crossover(self, particle):
        
        """
        Crossover algithm
        Param particle: permutation
        """
        new_perm = []
        
        #Sets weights for the current permutation, pbest, and gbest
        w_current = self.current_weight
        w_pbest = self.pbest_weight
        w_gbest = self.gbest_weight
        
        #Sets length and start and end of current permutation you want to select
        l_current = math.ceil((w_current*self.num_cities))
        l_current = l_current - int(rnd.random()*(l_current-1))
        start = rnd.sample(range(self.num_cities - l_current), 1)[0]
        end = start + l_current
                
        part_current = particle.current[start:end]
        new_perm = new_perm + list(part_current)
        
        #Repeats process for pbest
        l_pbest = math.ceil((w_pbest*self.num_cities))
        l_pbest = l_pbest - int(rnd.random()*(l_pbest-1))
        start = rnd.sample(range(self.num_cities - l_pbest), 1)[0]
        end = start + l_pbest
        part_pbest = particle.pbest[start:end]
        
        #Checks if cities selected from pbest overlap with what was already selected from current
        for i in range(len(part_pbest)):
            
            if part_pbest[i] in new_perm: 
                continue
            
            else:   
                new_perm.append(part_pbest[i])
                
        #Repeats process for gbest            
        l_gbest = math.ceil((w_gbest*self.num_cities))
        l_gbest = l_gbest - int(rnd.random()*(l_gbest-1))
        start = rnd.sample(range(self.num_cities - l_gbest), 1)[0]
        end = start + l_gbest
        part_gbest = self.gbest[start:end]
        
        #Checks if cities selected from gbest overlap with what was already selected from current and pbest
        for i in range(len(part_gbest)):
            
            if part_gbest[i] in new_perm:
                continue
            
            else:  
                new_perm.append(part_gbest[i])
        
        #Finally, adds any missing cities
        if len(new_perm) == self.num_cities: 
            pass
        
        else:
            for i in range(self.num_cities):
                if i in new_perm:
                    continue
                
                else: 
                    new_perm.append(i)
                    

                            
        particle.current = np.array(new_perm)
        particle.current_dist = self.CityData.total_dist(particle.current)
        
        if particle.current_dist < particle.pbest_dist:
            particle.pbest_dist = particle.current_dist
            particle.pbest = particle.current
                                                          

    def update_swarm(self):    
    
        """Updates the swarm by comparing permutations to global best"""
        
        dist = []        
    
        #Iterate through particles in swarm and evaluate fitness
        for i in range(self.num_particles):
            
            self.crossover(self.swarm[i])
            dist.append(self.swarm[i].current_dist)

            #Checks if current particle is the best globally
            if self.swarm[i].current_dist < self.gbest_dist:
                self.gbest = self.swarm[i].current  #updates gbest
                self.gbest_dist = self.swarm[i].current_dist #updates gbest_dist
                     
        self.current_mean.append(statistics.mean(dist))
        self.gbest_list.append(self.gbest_dist)
                
           
    def run_PSO(self):
        
        """Runs the algorithm"""
        
        #Initializes the swarm
        self.initialize_swarm()
        
        #Updates swarm for each iteration
        for i in range(self.num_iterations):
            self.update_swarm()
            
        #Calculates error 
        self.error = self.gbest_dist - self.CityData.optimal_solution_dist
        
    def plot_results(self):
        
        """Plots mean path length and global best length in relation to optimal distance"""
        
        x = list(range(1, self.num_iterations + 2))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey = 'row')
        fig.suptitle('Optimization Results')
        
        ax1.plot(x, self.current_mean)
        ax1.set_title("Mean Path Length")
        ax1.set(xlabel = "Iteration", ylabel = "Path Length")
        ax1.axhline(y= self.CityData.optimal_solution_dist, color='r', linestyle='-')

        ax2.plot(x, self.gbest_list)
        ax2.set_title("Global Best Path Length")
        ax2.set(xlabel = "Iteration", ylabel = "Path Length")
        fig.subplots_adjust(hspace=0.05)
        ax2.axhline(y= self.CityData.optimal_solution_dist, color='r', linestyle='-')
        
        
    def plot_error(self, num_simulations, print_progress = True):
        
        self.num_simulations = num_simulations
        error_list = []
        
        for i in range(num_simulations):
            self.run_PSO()
            error_list.append(self.error)
            
            if print_progress:
                print(i)
    
        avg = sum(error_list)/len(error_list)
        self.mean_error = avg
        
        mat = np.vstack((list(range(1, num_simulations + 1)),error_list)).T
        
        df = pd.DataFrame(mat, columns= ['X', "Y"])
        plt.scatter('X', 'Y', data = df, color = "green")
        plt.plot('X', 'Y', data = df)
        plt.title('Error per Simulation')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.axhline(y = avg, color='r', linestyle='-')
        
        
            
    def plot_best_path(self):
            
        self.CityData.plot_cities(self.gbest)
        

"""
#Nelder-Mead Algorithm
#Did not use

class Location:
    
    #Location class initalizes an x,y coordinate grid and allows for mathematical operations on the locations
    #May not need this depending on data setup
    
    def __init__(self, x, y):
        # Initializes an x,y coordinate like loc = Location(1,1)
        self.x = x
        self.y = y
        
    def __add__(self, new):
        #Addition method
        x = self.x + new.x
        y = self.y + new.y
        return Location(x, y)
    
    def __sub__(self, new):
        #Subtraction method
        x = self.x - new.x
        y = self.y - new.y
        return Location(x, y)
    
    def __rmul__(self, new):
        #Multiplication method
        x = self.x * new
        y = self.y * new
        return Location(x, y)
    
    def __truediv__(self, new):
        #Division method
        x = self.x / new
        y = self.y / new
        return Location(x, y)
    
    def getcoordinates(self):
        #Returns coordinates in x,y
        return (self.x, self.y)
    
    def __str__(self):
        #Returns a printable string representation of coordinates
        return str((self.x, self.y))
    
"""
"""
Nelder-Mead uses reflection, expansion, and contraction methods
1.) Reflection: xr = centroid + alpha(centroid - worst)
2.) Expansion: xe = centroid + gamma(xr - centroid)
3.) Contraction: xc = centroid + rho(worst - centroid)
4.) Did not use shrink as it is not necessary with contraction
"""   
"""
    
# Function you want to find optimal solution for (circle)
def func (coordinates):
    x, y = coordinates
    return (x-5)**2 + (y-5)**2 #circle with center at (5,5)

def nelder_mead (alpha=1, gamma=2, rho=0.5, num_iter=100):
    """
"""
    Params:
        alpha is the reflection paramater/coefficient, usually 1
        gamma is the expansion paramater/coefficient, usually 2
        rho is the contraction paramater/coefficient, usually equal to 0.5
        num_iter is the number of iterations you want to run through the algorithm, default is 100
    """
"""
    # Initialize the simplex using 3 random points
    v1 = Location(0, 0)
    v2 = Location(10, 0)
    v3 = Location(-15, 10)
    
    for i in range(num_iter): #runs through loop for number of iterations

        #Puts results into dictionary with each location and values from the test function
        results = {v1: func(v1.getcoordinates()), v2: func(v2.getcoordinates()), v3: func(v3.getcoordinates())}   
        #Sorts results based on the lowest function value
        sorted_results = sorted(results.items(), key = lambda coordinates: coordinates[1])
        
        #Creates best, second best, and worst points
        best = sorted_results[0][0] #best point = coordinates of first value
        second_best = sorted_results[1][0] #second best point = coordinates of second value
        worst = sorted_results[2][0] #worst point = coordinates of third value

        centroid = (second_best + best)/2 #centroid is geometric center of points besides worst
        
        #1. Reflection
        #checks if value of xr is better than the second best value but not better than the best
        xr = centroid + alpha * (centroid - worst) #formula for reflected point
        if func(xr.getcoordinates()) < func(second_best.getcoordinates()) and func(best.getcoordinates()) <= func(xr.getcoordinates()): 
            worst = xr #replace the worst point with the reflected point
            
        #2. Expansion
        if func(xr.getcoordinates()) < func(best.getcoordinates()): #if reflected point is best so far
            xe = centroid + gamma * (xr - centroid) #formula for expanded point
            if func(xe.getcoordinates()) < func(xr.getcoordinates()): #if expanded point is better than the reflected point
                worst = xe #replace worst point with expanded point
            else:
                worst = xr #else replace worst point with reflected point
                
        #3. Contraction       
        if func(xr.getcoordinates()) > func(second_best.getcoordinates()): #checks if reflected point is better than second best point
            xc = centroid + rho * (worst - centroid) #formula for contracted point
            if func(xc.getcoordinates()) < func(worst.getcoordinates()): #if contracted point is better than the worst point
                worst = xc #replace worst point with contracted point
                
        #Update points based on loop
        v1 = best
        v2 = second_best
        v3 = worst
        #print("v1: ", v1, " v2: ", v2, " v3: ", v3)
        
    return v1 #returns best point

x = nelder_mead()
print(x)
x.getcoordinates()

"""
