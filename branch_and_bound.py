import numpy as np
import itertools
import time
from scipy.spatial.distance import cdist

def create_distance_matrix_from_coordinates(coordinates):
    """Create a distance matrix from city coordinates using Euclidean distance"""
    return cdist(coordinates, coordinates, metric='euclidean')


def calculate_tsp_cost(tour, dist_matrix):
    """Calculate the total distance of a TSP tour"""
    cost = 0
    for i in range(len(tour) - 1):
        cost += dist_matrix[tour[i], tour[i + 1]]
    cost += dist_matrix[tour[-1], tour[0]]  # Return to start
    return cost


def branch_and_bound_tsp(num_cities, coordinates):
    """
    Solve TSP using a brute-force Branch and Bound method.

    Args:
        num_cities (int): Number of cities
        coordinates (list): List of (x, y) city coordinates

    Returns:
        tuple: (best_distance, best_tour)
    """
    print("[B&B] Starting Branch and Bound for TSP...")

    start_time = time.time()

    # Create the distance matrix
    dist_matrix = create_distance_matrix_from_coordinates(coordinates)

    # Initialize best path and cost
    best_cost = float('inf')
    best_path = None

    # Generate all possible permutations (starting from city 0 for optimization)
    cities = list(range(1, num_cities))  # Exclude city 0 to fix starting point
    all_perms = itertools.permutations(cities)

    for perm in all_perms:
        # Prepend city 0 at start
        path = (0,) + perm
        cost = calculate_tsp_cost(path, dist_matrix)

        if cost < best_cost:
            best_cost = cost
            best_path = path

    total_time = time.time() - start_time

    print(f"[B&B] Best tour found: {best_path}")
    print(f"[B&B] Tour distance: {best_cost:.2f}")
    print(f"[B&B] Total execution time: {total_time:.2f} seconds")

    return best_cost, list(best_path)