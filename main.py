
# import libraries
import numpy as np
# import personal functions
from generate_population import generate_initial_population
from calculate_aptitude import evaluate_population, rastrigin_function
from tournament_parent_selection import tournament_selection
from sbx_crossover import sbx
from polynomial_mutation import apply_polinomial_mutation


"""
@brief: Function to solve the Rastrigin function using a genetic algorithm
@context: This function run the full GA process using the independent functions created before
@param config: dict -> Configuration object for the GA
@return: dict -> Best individual found by the GA and its aptitude
"""
def run_ga_optimization(config: dict) -> dict:
    # --------------------------------------------------------------- 
    # Get all configurations
    # ---------------------------------------------------------------
    population_size = ga_config.get('population_size', 100)
    generations = ga_config.get('generations', 50)
    variables = ga_config.get('variables', 2)
    limits = ga_config.get('limits', np.array([[0, 10], [0, 10]]))
    sbx_prob = ga_config.get('sbx_prob', 0.8)
    sbx_dispersion_param = ga_config.get('sbx_dispersion_param', 3)
    mutation_probability_param = ga_config.get('mutation_probability_param', 0.7)
    distribution_index_param = ga_config.get('distribution_index_param', 50)
    activate_dynamic_sbx_increasing = ga_config.get('dynamic_sbx_increasing', False)
    if activate_dynamic_sbx_increasing:
        increment_rate = (20 - sbx_dispersion_param) / (generations+(population_size/variables**variables)) if sbx_dispersion_param < 20 else 0
    else:
        increment_rate = 0
       
    # ---------------------------------------------------------------         
    # ALGORITHM EXECUTION
    # ---------------------------------------------------------------
    # 0. Generate initial population
    population = generate_initial_population(population_size, variables, limits)
    
    # iterate over the number of generations
    for i in range(generations):
        # 1. Calculate aptitude vector of the population  
        aptitude = evaluate_population(population, rastrigin_function)
        # 1.1 Get the best individual of the current population
        best_individual = population[np.argmin(aptitude)].copy()
        
        # 2. Select the parents using tournament selection
        population = tournament_selection(population, aptitude)
        
        # 3. SBX crossover
        sbx(population, limits, sbx_prob, sbx_dispersion_param)
        # 4. Apply Polynomial Mutation
        apply_polinomial_mutation(population, limits, mutation_probability_param, distribution_index_param)
        
        # 5. Calculate the new aptitude vector for the population after crossover
        aptitude = evaluate_population(population, rastrigin_function)
        # 5.1 Get the worst individual index of the population (child population)
        worst_after_crossover = np.argmax(aptitude)
        
        # 6. ELITISM
        # replace the worst individual after crossover (children) with the best individual before crossover (parent)
        population[worst_after_crossover] = best_individual
        
        # 7. EXTRA (dynamic configurations)
        sbx_dispersion_param += increment_rate if sbx_dispersion_param < 20 else 0
        
        
    # get the final aptitude vector
    aptitude = evaluate_population(population, rastrigin_function)
    # get the best individual index = min(aptitude)
    best_individual_index = np.argmin(aptitude)
    
    
    return {'individual': population[best_individual_index], 'aptitude': aptitude[best_individual_index]}



# ===============================================================
# GA CONFIGURATIONS
# ===============================================================
ga_config = {
    'population_size': 100,
    'generations': 5000,
    'variables': 2,
    'limits': np.array([[-5.12, 5.12], [-5.12, 5.12]]),
    'sbx_prob': 0.9,
    'sbx_dispersion_param': 2,
    'mutation_probability_param': 0,
    'distribution_index_param': 20,
    'alpha': 0.5, # for fitness sharing
    'dynamic_sbx_increasing': False,
    'early_stop': False,
}

# Show the configurations
print(f'{"*"*50}')
print('Algorithm configurations:')
print(f'Number of generations: {ga_config["generations"]}')
print(f'Population size: {ga_config["population_size"]}')
print(f'Number of variables: {ga_config["variables"]}')
print(f'Limits: \n{ga_config["limits"]}')
print(f'SBX Probability (pc): {ga_config["sbx_prob"]}')
print(f'SBX Dispersion Parameter (nc): {ga_config["sbx_dispersion_param"]}')
print(f'Mutation Probability: {ga_config["mutation_probability_param"]}')
print(f'Distribution Index (nm): {ga_config["distribution_index_param"]}')


# ===============================================================
# EXECUTION OF THE MAIN FUNCTION
# ===============================================================
print(f'\n{"*"*50}')
print('Running Algorithm...')
results = [] # list of dictionaries to store the results
for i in range(10):
    # show the current execution, the results of function and then add the results to the list
    print(f'Execution {i+1}')
    result = run_ga_optimization(ga_config)
    print(f'\tIndividual: {result["individual"]}, Aptitude = {result["aptitude"]}')
    results.append(result)
    

# only pass the results.aptitude of the results dictionaries
partial_results = np.array([result['aptitude'] for result in results])
# get the best, median, worst and standard deviation of the results
best = np.min(partial_results)
median = np.median(partial_results)
worst = np.max(partial_results)
std = np.std(partial_results)

print(f'\n{"*"*50}')
print(f'Statistics:')
print(f'Best: {best}')
print(f'Median: {median}')
print(f'Worst: {worst}')
print(f'Standard deviation: {std}')

    