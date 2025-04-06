# pip install numpy matplotlib networkx
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# External functions (already implemented elsewhere)
from simulated_annealing import simulated_annealing_tsp
from branch_and_bound import branch_and_bound_tsp
from grovers_algorithm import grovers_algorithm_tsp
from qaoa import qaoa_tsp

def visualize_tsp_results(results, coordinates, best_algorithms, min_distance):
    """Visualize all TSP tours side by side"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Create position dictionary for all plots
    pos = {i: (coordinates[i][0], coordinates[i][1]) for i in range(len(coordinates))}

    for idx, (algo_name, (tour_distance, tour)) in enumerate(results.items()):
        ax = axes[idx]

        # Create a directed graph for the tour
        G = nx.DiGraph()
        num_cities = len(tour)

        # Add nodes
        for i in range(num_cities):
            G.add_node(tour[i])

        # Add edges with weights
        edge_labels = {}
        for i in range(num_cities):
            city = tour[i]
            next_city = tour[(i + 1) % num_cities]
            # Get cost from the returned tour distance
            cost = tour_distance / num_cities  # Approximate cost per edge for display
            G.add_edge(city, next_city, weight=cost)
            edge_labels[(city, next_city)] = f"{cost:.2f}"

        # Determine node color based on whether this algorithm is among the best
        node_color = 'lightgreen' if algo_name in best_algorithms else 'lightblue'

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=500, ax=ax)

        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_weight='bold', ax=ax)

        # Draw edges with arrows
        edge_color = 'green' if algo_name in best_algorithms else 'blue'
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20,
                               edge_color=edge_color, width=2, ax=ax)

        # Add edge labels (costs)
        for (u, v), label in edge_labels.items():
            # Calculate a position above the edge
            edge_center = np.array([(pos[u][0] + pos[v][0]) / 2,
                                    (pos[u][1] + pos[v][1]) / 2])
            # Find the perpendicular direction to offset the label
            dx = pos[v][0] - pos[u][0]
            dy = pos[v][1] - pos[u][1]
            # Normalize and perpendicular
            length = np.sqrt(dx ** 2 + dy ** 2)
            if length > 0:
                dx, dy = dx / length, dy / length
                # Perpendicular vector
                px, py = -dy, dx
                # Offset position for the label
                offset = 0.3
                label_pos = edge_center + np.array([px, py]) * offset
                ax.text(label_pos[0], label_pos[1], label,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=8, color='darkblue')

        # Number the edges to show tour order
        for i in range(num_cities):
            city = tour[i]
            next_city = tour[(i + 1) % num_cities]
            edge_center = np.array([(pos[city][0] + pos[next_city][0]) / 2,
                                    (pos[city][1] + pos[next_city][1]) / 2])
            ax.text(edge_center[0], edge_center[1], f"{i + 1}",
                    bbox=dict(facecolor='white', alpha=0.7),
                    horizontalalignment='center',
                    verticalalignment='center')

        # Add city coordinates to plot
        for i in range(len(coordinates)):
            ax.annotate(f"({coordinates[i][0]:.1f}, {coordinates[i][1]:.1f})",
                        (coordinates[i][0], coordinates[i][1]),
                        textcoords="offset points",
                        xytext=(0, -20),
                        ha='center',
                        fontsize=8)

        # Determine if this is an optimal solution and add appropriate title
        if algo_name in best_algorithms:
            title = f"{algo_name}\nTour: {tour}\nCost: {tour_distance:.2f} (OPTIMAL)"
        else:
            gap_percentage = ((tour_distance - min_distance) / min_distance) * 100
            title = f"{algo_name}\nTour: {tour}\nCost: {tour_distance:.2f}\nGap: +{gap_percentage:.2f}%"

        ax.set_title(title)
        ax.axis('on')
        ax.grid(True, linestyle='--', alpha=0.7)

        # Set axis limits with padding
        x_coords = [coordinates[i][0] for i in range(len(coordinates))]
        y_coords = [coordinates[i][1] for i in range(len(coordinates))]
        padding = 1.0
        ax.set_xlim(min(x_coords) - padding, max(x_coords) + padding)
        ax.set_ylim(min(y_coords) - padding, max(y_coords) + padding)

    plt.tight_layout()
    plt.savefig("tsp_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


def calculate_optimality_gap(results):
    """
    Calculate the optimality gap for each algorithm.
    Returns a dictionary with algorithm names as keys and optimality metrics as values.
    """
    distances = {algo: result[0] for algo, result in results.items()}
    min_distance = min(distances.values())

    gaps = {}
    for algo, distance in distances.items():
        # Calculate absolute gap
        absolute_gap = distance - min_distance
        # Calculate percentage gap
        percentage_gap = (absolute_gap / min_distance) * 100 if min_distance > 0 else 0
        # Calculate relative performance (1.0 means optimal)
        relative_performance = min_distance / distance if distance > 0 else 0

        gaps[algo] = {
            'distance': distance,
            'absolute_gap': absolute_gap,
            'percentage_gap': percentage_gap,
            'relative_performance': relative_performance
        }

    return gaps, min_distance


def main():
    """Main function to run and compare different TSP algorithms"""
    print("=" * 50)
    print("TSP Algorithm Comparison")
    print("=" * 50)

    # Get number of cities from user
    while True:
        try:
            num_cities = int(input("Enter the number of cities: "))
            if num_cities < 3:
                print("Please enter at least 3 cities.")
                continue
            break
        except ValueError:
            print("Please enter a valid integer.")

    # Generate random coordinates
    print(f"\nGenerating random coordinates for {num_cities} cities...")
    coordinates = [(np.random.rand() * 10, np.random.rand() * 10) for _ in range(num_cities)]

    # Display the coordinates
    print("\nCity coordinates:")
    for i, (x, y) in enumerate(coordinates):
        print(f"City {i}: ({x:.2f}, {y:.2f})")

    # Dictionary to store results
    results = {}

    # Call each algorithm (all implemented elsewhere)
    print("\nRunning Simulated Annealing algorithm...")
    sa_distance, sa_tour = simulated_annealing_tsp(num_cities, coordinates)
    results["Simulated Annealing"] = (sa_distance, sa_tour)

    print("\nRunning Branch and Bound algorithm...")
    bb_distance, bb_tour = branch_and_bound_tsp(num_cities, coordinates)
    results["Branch and Bound"] = (bb_distance, bb_tour)

    print("\nRunning Grover's Algorithm simulation...")
    grover_distance, grover_tour = grovers_algorithm_tsp(num_cities, coordinates)
    results["Grover's Algorithm"] = (grover_distance, grover_tour)

    print("\nRunning QAOA simulation...")
    qaoa_distance, qaoa_tour = qaoa_tsp(num_cities, coordinates)
    results["QAOA"] = (qaoa_distance, qaoa_tour)

    # Calculate optimality gaps
    gaps, min_distance = calculate_optimality_gap(results)

    # Determine which algorithms achieved the optimal solution
    best_algorithms = [algo for algo, metrics in gaps.items()
                       if abs(metrics['absolute_gap']) < 1e-10]  # Using small epsilon for floating point comparison

    # Find ties (algorithms that produce the same result)
    unique_distances = {}
    for algo, (distance, _) in results.items():
        # Round to handle floating point precision issues
        rounded_distance = round(distance, 2)
        if rounded_distance in unique_distances:
            unique_distances[rounded_distance].append(algo)
        else:
            unique_distances[rounded_distance] = [algo]

    # Display algorithms with identical results
    print("\nAlgorithms with identical results:")
    print("-" * 50)
    for distance, algos in unique_distances.items():
        if len(algos) > 1:
            print(f"Distance {distance:.2f}: {', '.join(algos)}")

    # Compare results
    print("\nAlgorithm comparison:")
    print("-" * 80)
    print(f"{'Algorithm':<20} {'Distance':<10} {'Gap (%)':<10} {'Relative Perf.':<15}")
    print("-" * 80)

    # Sort by distance for clearer comparison
    sorted_gaps = sorted(gaps.items(), key=lambda x: x[1]['distance'])

    for algo, metrics in sorted_gaps:
        is_optimal = " (OPTIMAL)" if algo in best_algorithms else ""
        print(
            f"{algo:<20} {metrics['distance']:<10.2f} {metrics['percentage_gap']:<10.2f}% {metrics['relative_performance']:<15.3f}{is_optimal}")

    # Report on the best solution(s)
    if len(best_algorithms) == 1:
        print(f"\nBest solution: {best_algorithms[0]} with distance {min_distance:.2f}")
    else:
        print(f"\nMultiple algorithms achieved the optimal distance of {min_distance:.2f}:")
        for algo in best_algorithms:
            print(f"- {algo}")

    # Visualize the results
    print("\nGenerating visualizations...")
    visualize_tsp_results(results, coordinates, best_algorithms, min_distance)
    print("Visualization saved to 'tsp_comparison.png'")


if __name__ == "__main__":
    main()