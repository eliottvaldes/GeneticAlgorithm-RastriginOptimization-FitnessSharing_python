import numpy as np

def evaluate_constraints(individual: np.ndarray, constraint_functions: dict) -> list:
    #destructuring the individual values
    Rd = np.array([g(individual) for g in constraint_functions['inequalities']])
    Ri = np.array([h(individual) for h in constraint_functions['equalities']])
    return Rd, Ri

def calculate_penalty(Rd: float, Ri: float) -> float:
    penalty_inequalities = np.sum(np.maximum(Rd, 0)**2)
    penalty_equalities = np.sum(Ri**2)
    return penalty_inequalities + penalty_equalities

def penalized_objective_function(population: np.ndarray, objective_function, constraint_functions: dict, lmbda: float):
    n_individuals, n_var = population.shape
    aptitude_vector = np.zeros(n_individuals)
    
    for i in range(n_individuals):
        # get the individual
        individual = population[i]
        f_xy = objective_function(individual)
        Rd, Ri = evaluate_constraints(individual, constraint_functions)
        P_xy = calculate_penalty(Rd, Ri)
        Fp_xy = f_xy + lmbda * P_xy
        
        # add to the aptitude vector
        aptitude_vector[i] = Fp_xy           
        
    return aptitude_vector


if __name__ == '__main__':
    # Define an example objective function
    def objective_function(individual: np.ndarray) -> float:
        x1, x2, x3, x4, x5, x6 = individual
        r1 = 0.20
        r2 = 0.42
        r3 = 1
        r4 = 0.5
        r5 = 0.46
        r6 = 0.30
        return r1*x1 + r2*x2 + r3*x3 + r4*x4 + r5*x5 + r6*x6

    # Define some example constraint functions
    constraint_functions = {
        'inequalities': [
            # sould be less or equals to 0.
            lambda individual: sum(individual) - 1,
        ],
        'equalities': []
    }

    # Example individual coordinates
    np.random.seed(42) 
    population_size = 200
    num_values = 6
    population = np.random.uniform(0, 0.4, (population_size, num_values))
    # define the penalization parameter
    lmbda = 10**3

    # Calculate penalized objective function
    penalized_value = penalized_objective_function(population, objective_function, constraint_functions, lmbda)
    print("Penalized Objective Function Value:", penalized_value)
