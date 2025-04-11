import numpy as np
import random
import time
from math import exp

# Create matrix for cities
def create_distance_matrix_from_coordinates(coordinates):
    num_cities = len(coordinates)
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[j]
            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
    return dist_matrix

# Calculate total distance
def calculate_tsp_cost(tour, dist_matrix):
    cost = 0
    for i in range(len(tour)):
        cost += dist_matrix[tour[i]][tour[(i + 1) % len(tour)]]
    return cost

# Main simulated annealing
def simulated_annealing_tsp(num_cities, coordinates):
    """
    Solve TSP using Simulated Annealing

    Args:
        num_cities (int): Number of cities
        coordinates (list): List of (x, y) coordinates

    Returns:
        tuple: (best_cost, best_tour)
    """
    print("[SA] Starting Simulated Annealing for TSP...")

    dist_matrix = create_distance_matrix_from_coordinates(coordinates)

    # Setting Parameters
    annealing_initial_temperature = 1000.0
    annealing_final_temperature = 1e-3
    annealing_cooling_rate = 0.995
    max_iter = 1000

    # Initialize solution
    current_solution = list(range(num_cities))
    random.shuffle(current_solution)
    current_cost = calculate_tsp_cost(current_solution, dist_matrix)

    best_solution = list(current_solution)
    best_cost = current_cost

    annealing_temperature = annealing_initial_temperature
    start_time = time.time()

    while annealing_temperature > annealing_final_temperature:
        for _ in range(max_iter):
            # Swapping 2 cities
            i, j = random.sample(range(num_cities), 2)
            neighbor = list(current_solution)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbor_cost = calculate_tsp_cost(neighbor, dist_matrix)

            # Decide whether to accept the new solution
            delta = neighbor_cost - current_cost
            if delta < 0 or random.random() < exp(-delta / annealing_temperature):
                current_solution = neighbor
                current_cost = neighbor_cost

                if current_cost < best_cost:
                    best_solution = list(current_solution)
                    best_cost = current_cost

        annealing_temperature *= annealing_cooling_rate

    total_time = time.time() - start_time
    print(f"[SA] Best tour found: {best_solution}")
    print(f"[SA] Tour distance: {best_cost:.2f}")
    print(f"[SA] Total execution time: {total_time:.2f} seconds")

    return best_cost, best_solution
