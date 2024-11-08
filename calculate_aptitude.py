import numpy as np

"""
@breaf: This function is used to calculate the aptitude of the individuals in the population using any Objective Function passed as a parameter
@context: General approach to evaluate the individuals in the population
@param population: np.ndarray -> All population to be evaluated -> F.E: [[2.3, 4.5], [1.4, -0.2]]
@param objective_function: function -> Objective function to evaluate the individuals
@return: np.ndarray -> Aptitude of the individuals in the population (Vector) [9, 5]
"""
def evaluate_population(population: np.ndarray, objective_function) -> np.ndarray:
    # get the number of individuals and variables
    n_individuals, n_var = population.shape
    # create the aptitude vector
    aptitude = np.zeros(n_individuals)
    # loop over the population
    for i in range(n_individuals):
        # get the individual
        individual = population[i]
        # evaluate the individual
        aptitude[i] = objective_function(individual)
    return aptitude

def stock_return(individual: np.ndarray) -> float:
    # toma en cuenta que mi individuo tiene la forma [x1, x2, x3, x4, x5, x6]
    x1, x2, x3, x4, x5, x6 = individual
    # the fixed values for the stock return are:
    r1 = 0.20
    r2 = 0.42
    r3 = 1
    r4 = 0.5
    r5 = 0.46
    r6 = 0.30
    
    return r1*x1 + r2*x2 + r3*x3 + r4*x4 + r5*x5 + r6*x6


if __name__ == '__main__':
    # JUST FOR TESTING
    # GENERAL CONFIGURATIONS
    population_zize = 10
    num_values = 6
    population = np.random.uniform(0, 0.4, (population_zize, num_values))
    
    # evaluate the population
    aptitude = evaluate_population(population, stock_return)
    # get the index of the minimum aptitude
    best_index = np.argmax(aptitude)
    
    print(f'Population: \n{population}\n')
    print(f'Aptitude: \n{aptitude}')
    print(f'Best index: {best_index}')
    print(f'Best individual: {population[best_index]}')   