import numpy as np

"""
@breaf: Modifies the aptitude of individuals in the population based on their proximity to other individuals.
@context: This function is used to apply fitness sharing to the population.
@param: population (np.ndarray): The population matrix where each row is an individual and columns are the features.
@param: aptitudes (np.ndarray): The initial aptitudes of the population calculated with the objective function.
@param: alpha (float): The alpha parameter controlling the shape of the sharing function.
@param: niche_radius (float): The niche radius defining how close two individuals should be to affect each other.
@return: np.ndarray: The modified aptitudes after applying fitness sharing.
"""
def fitness_sharing_vectorized(population: np.ndarray, aptitudes: np.ndarray, alpha: float, niche_radius: float) -> np.ndarray:
    # Calculate the matrix of distances between all pairs of individuals
    diff = population[:, np.newaxis, :] - population[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)

    # Compute the sharing values for each pair of individuals
    sharing_matrix = np.where(distances < niche_radius, 1 - (distances / niche_radius) ** alpha, 0)

    # Ensure that individuals do not affect their own aptitude
    np.fill_diagonal(sharing_matrix, 0)

    # Sum the sharing effects for each individual and adjust the aptitudes
    sharing_effect = np.sum(sharing_matrix, axis=1)
    modified_aptitudes = aptitudes + sharing_effect

    return modified_aptitudes


# @breaf: Modifies the aptitude of individuals in the population based on their proximity to other individuals.
# @context: This function is used to apply fitness sharing to the population.
#! Function not vectorized, so it is slower than the vectorized version due to the use of loops.
def fitness_sharing(population, aptitudes, alpha, radio_nicho):
    n_individuals = population.shape[0]
    shared_aptitudes = np.copy(aptitudes)
    
    for i in range(n_individuals):
        for j in range(n_individuals):
            if i != j:
                distance = np.linalg.norm(population[i] - population[j])
                if distance < radio_nicho:
                    sharing_value = 1 - (distance / radio_nicho) ** alpha
                    shared_aptitudes[i] += sharing_value
                    print(f"i: {i}, j: {j}, distance: {distance:.4f}, sharing_value: {sharing_value:.4f}")
    
    return shared_aptitudes


if __name__ == '__main__':
    # Test the fitness sharing function with a sample population
    np.random.seed(42)  # For reproducibility if always the same random values are needed
    population_size = 5
    dimensions = 2
    niche_radius = 0.01
    alpha = 0.5

    # Generate a random population within the range [-5.12, 5.12]
    population = np.random.uniform(-5.12, 5.12, (population_size, dimensions))
    # Evaluate the initial aptitudes using a sample objective function (Rastrigin here)
    aptitudes = np.array([np.sum(ind ** 2 - 10 * np.cos(2 * np.pi * ind)) + 10 * len(ind) for ind in population])

    print("Initial Population:")
    print(population)
    print("Initial Aptitudes:")
    print(aptitudes)
    
    # Apply fitness sharing
    modified_aptitudes = fitness_sharing_vectorized(population, aptitudes, alpha, niche_radius)
    
    print("Modified Aptitudes After Fitness Sharing:")
    print(modified_aptitudes)