# pip install numpy matplotlib networkx
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# External functions (already implemented elsewhere)
from simulated_annealing import simulated_annealing_tsp
from branch_and_bound import branch_and_bound_tsp
from grovers_algorithm import grovers_algorithm_tsp
from qaoa import qaoa_tsp


def visualize_tsp_results(results, coordinates):
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

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightgreen', node_size=500, ax=ax)

        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_weight='bold', ax=ax)

        # Draw edges with arrows
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20,
                               edge_color='blue', width=2, ax=ax)

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

        ax.set_title(f"{algo_name}\nTour: {tour}\nCost: {tour_distance:.2f}")
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

    # Compare results
    print("\nAlgorithm comparison:")
    print("-" * 50)
    print(f"{'Algorithm':<20} {'Distance':<10}")
    print("-" * 50)
    print(f"{'Simulated Annealing':<20} {sa_distance:<10.2f}")
    print(f"{'Branch and Bound':<20} {bb_distance:<10.2f}")
    print(f"{'Grovers Algorithm':<20} {grover_distance:<10.2f}")
    print(f"{'QAOA':<20} {qaoa_distance:<10.2f}")

    # Find the best solution
    best_algo = min(results.items(), key=lambda x: x[1][0])[0]
    best_distance = min(sa_distance, bb_distance, grover_distance, qaoa_distance)
    print(f"\nBest solution: {best_algo} with distance {best_distance:.2f}")

    # Visualize the results
    print("\nGenerating visualizations...")
    visualize_tsp_results(results, coordinates)
    print("Visualization saved to 'tsp_comparison.png'")


if __name__ == "__main__":
    main()