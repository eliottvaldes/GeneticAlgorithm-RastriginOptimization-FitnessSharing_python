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


# define the rastrigin function
def rastrigin_function(individual: np.ndarray) -> float:
    x, y = individual
    term_1 = np.power(x, 2) - 10 * np.cos(10 * np.pi * x)
    term_2 = np.power(y, 2) - 10 * np.cos(2 * np.pi * y)
    return 20 + (term_1 + term_2)
    
    


if __name__ == '__main__':
    # JUST FOR TESTING
    # GENERAL CONFIGURATIONS
    population = np.array([[20 , 4.5], [0, 0]])
    
    # evaluate the population
    aptitude = evaluate_population(population, rastrigin_function)
    # get the index of the minimum aptitude
    best_index = np.argmin(aptitude)
    
    print(f'Population: \n{population}\n')
    print(f'Aptitude: {aptitude}')
    print(f'Best index: {best_index}')
    print(f'Best individual: {population[best_index]}')   